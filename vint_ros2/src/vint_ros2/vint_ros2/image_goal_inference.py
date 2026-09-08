"""ROS 2 registered-route selector and image-goal ViNT inference node."""

import math
from threading import Lock
import time

from nav_msgs.msg import Path as NavPath
from PIL import Image as PilImage
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from tf2_ros import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformListener

from vint_ros2.bag_conversion import BagConversionError
from vint_ros2.bag_conversion import decode_image
from vint_ros2.inference_common import create_backend
from vint_ros2.inference_common import declare_backend_parameters
from vint_ros2.inference_common import history_is_fresh
from vint_ros2.inference_common import ImageHistory
from vint_ros2.inference_common import prediction_to_path
from vint_ros2.inference_common import stamp_to_nanoseconds
from vint_ros2.model_backend import ModelBackendError
from vint_ros2.route_geometry import RouteGeometryError
from vint_ros2.route_geometry import select_route_goal
from vint_ros2.route_manifest import load_route_manifest


class ImageGoalInferenceNode(Node):
    """Select registered goal images and publish image-goal ViNT paths."""

    def __init__(self, *, parameter_overrides=None) -> None:
        """Initialize registered route, image history, TF, and inference loop."""
        super().__init__(
            'vint_image_goal_inference',
            parameter_overrides=parameter_overrides,
        )
        declare_backend_parameters(self)
        self.declare_parameter('route_manifest', '')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('image_type', 'raw')
        self.declare_parameter(
            'global_plan_topic', '/vint/image_goal/global_plan'
        )
        self.declare_parameter(
            'prediction_topic', '/vint/image_goal/predicted_path'
        )
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('inference_rate_hz', 4.0)
        self.declare_parameter('lookahead_distance_m', 2.0)
        self.declare_parameter('route_corridor_tolerance_m', 1.0)
        self.declare_parameter('maximum_image_age_sec', 0.5)
        self.declare_parameter('maximum_plan_age_sec', 4.0)
        self.declare_parameter('warning_throttle_sec', 2.0)

        route_path = str(self.get_parameter('route_manifest').value)
        if not route_path:
            raise ValueError('route_manifest is required.')
        self._route = load_route_manifest(route_path)
        self._goal_images: dict[int, PilImage.Image] = {}
        self._backend = create_backend(self, 'image_goal')
        self._history = ImageHistory(self._backend.context_size)
        self._robot_frame = str(self.get_parameter('robot_frame').value)
        self._lookahead = float(
            self.get_parameter('lookahead_distance_m').value
        )
        self._corridor = float(
            self.get_parameter('route_corridor_tolerance_m').value
        )
        self._maximum_image_age = float(
            self.get_parameter('maximum_image_age_sec').value
        )
        self._maximum_plan_age = float(
            self.get_parameter('maximum_plan_age_sec').value
        )
        self._warning_throttle = float(
            self.get_parameter('warning_throttle_sec').value
        )
        self._last_warning = float('-inf')
        self._plan_lock = Lock()
        self._inference_lock = Lock()
        self._global_plan = None
        self._previous_goal_index = 0
        self._plan_destination = None

        image_type = str(self.get_parameter('image_type').value).lower()
        if image_type == 'raw':
            message_type = Image
        elif image_type == 'compressed':
            message_type = CompressedImage
        else:
            raise ValueError(
                "image_type must be either 'raw' or 'compressed'."
            )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._image_group = ReentrantCallbackGroup()
        self._inference_group = MutuallyExclusiveCallbackGroup()
        self._image_subscription = self.create_subscription(
            message_type,
            str(self.get_parameter('image_topic').value),
            self._image_received,
            qos_profile_sensor_data,
            callback_group=self._image_group,
        )
        self._plan_subscription = self.create_subscription(
            NavPath,
            str(self.get_parameter('global_plan_topic').value),
            self._plan_received,
            10,
            callback_group=self._image_group,
        )
        self._publisher = self.create_publisher(
            NavPath,
            str(self.get_parameter('prediction_topic').value),
            10,
        )
        rate = float(self.get_parameter('inference_rate_hz').value)
        if rate <= 0:
            raise ValueError('inference_rate_hz must be positive.')
        self._timer = self.create_timer(
            1.0 / rate,
            self._infer,
            callback_group=self._inference_group,
        )
        self.get_logger().info(
            f"Image-goal inference ready for route '{self._route.route_id}' "
            f"with {self.get_parameter('backend').value} backend."
        )

    def _warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning >= self._warning_throttle:
            self.get_logger().warning(message)
            self._last_warning = now

    def _image_received(self, message) -> None:
        try:
            image = decode_image(message)
        except BagConversionError as error:
            self._warn(f'Skipping undecodable image: {error}')
            return
        self._history.append(stamp_to_nanoseconds(message.header.stamp), image)

    def _plan_received(self, plan: NavPath) -> None:
        if not plan.poses:
            return
        destination = (
            plan.poses[-1].pose.position.x,
            plan.poses[-1].pose.position.y,
        )
        expected = self._route.nodes[-1].pose
        if math.dist(destination, (expected.x, expected.y)) > self._corridor:
            self._warn(
                'Ignoring a global plan whose destination does not match the '
                f"configured route '{self._route.route_id}'."
            )
            return
        with self._plan_lock:
            if (self._plan_destination is None or
                    math.dist(destination, self._plan_destination) > 0.25):
                self._previous_goal_index = 0
            self._plan_destination = destination
            self._global_plan = plan

    def _goal_image(self, index: int) -> PilImage.Image:
        image = self._goal_images.get(index)
        if image is None:
            image = PilImage.open(
                self._route.nodes[index].image_path
            ).convert('RGB')
            self._goal_images[index] = image
        return image

    def _infer(self) -> None:
        observations, image_stamp = self._history.snapshot()
        now_ns = self.get_clock().now().nanoseconds
        if not history_is_fresh(
            observations,
            image_stamp,
            required_count=self._history.capacity,
            now_ns=now_ns,
            maximum_age_sec=self._maximum_image_age,
        ):
            return
        with self._plan_lock:
            plan = self._global_plan
            previous_goal = self._previous_goal_index
        if plan is None or plan.header.frame_id != self._route.frame_id:
            return
        plan_stamp_ns = stamp_to_nanoseconds(plan.header.stamp)
        if not 0 <= now_ns - plan_stamp_ns <= self._maximum_plan_age * 1e9:
            self._warn('Global plan is stale; no image-goal prediction published.')
            return
        try:
            transform = self._tf_buffer.lookup_transform(
                self._route.frame_id,
                self._robot_frame,
                Time(nanoseconds=image_stamp),
            )
            robot = (
                transform.transform.translation.x,
                transform.transform.translation.y,
            )
            selection = select_route_goal(
                self._route,
                robot,
                tuple((pose.pose.position.x, pose.pose.position.y)
                      for pose in plan.poses),
                lookahead_distance=self._lookahead,
                corridor_tolerance=self._corridor,
                previous_goal_index=previous_goal,
            )
            with self._inference_lock:
                prediction = self._backend.predict_image_goal(
                    observations,
                    self._goal_image(selection.goal_index),
                )
        except (TransformException, RouteGeometryError,
                ModelBackendError, OSError) as error:
            self._warn(f'Image-goal inference skipped: {error}')
            return
        with self._plan_lock:
            self._previous_goal_index = max(
                self._previous_goal_index,
                selection.goal_index,
            )
        stamp = Time(nanoseconds=image_stamp).to_msg()
        self._publisher.publish(prediction_to_path(
            prediction,
            frame_id=self._robot_frame,
            stamp=stamp,
        ))


def main(args=None) -> None:
    """Run image-goal inference with independent input and model callbacks."""
    rclpy.init(args=args)
    node = ImageGoalInferenceNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
