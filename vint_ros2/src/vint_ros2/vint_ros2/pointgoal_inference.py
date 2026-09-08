"""ROS 2 service for PointGoal ViNT short-horizon path inference."""

from threading import Lock
import time

from nav_msgs.srv import GetPlan
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

from vint_ros2.bag_conversion import BagConversionError
from vint_ros2.bag_conversion import decode_image
from vint_ros2.inference_common import create_backend
from vint_ros2.inference_common import declare_backend_parameters
from vint_ros2.inference_common import history_is_fresh
from vint_ros2.inference_common import ImageHistory
from vint_ros2.inference_common import prediction_to_path
from vint_ros2.inference_common import stamp_to_nanoseconds
from vint_ros2.model_backend import ModelBackendError


class PointGoalInferenceNode(Node):
    """Serve robot-relative predicted prefixes from camera history and goals."""

    def __init__(self, *, parameter_overrides=None) -> None:
        """Initialize the image stream, model backend, and inference service."""
        super().__init__(
            'vint_pointgoal_inference',
            parameter_overrides=parameter_overrides,
        )
        declare_backend_parameters(self)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('image_type', 'raw')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('service_name', '/vint/pointgoal_inference')
        self.declare_parameter('maximum_image_age_sec', 0.5)
        self.declare_parameter('warning_throttle_sec', 2.0)

        self._backend = create_backend(self, 'point_goal')
        self._history = ImageHistory(self._backend.context_size)
        self._robot_frame = str(self.get_parameter('robot_frame').value)
        self._maximum_image_age = float(
            self.get_parameter('maximum_image_age_sec').value
        )
        self._warning_throttle = float(
            self.get_parameter('warning_throttle_sec').value
        )
        self._last_warning = float('-inf')
        self._inference_lock = Lock()

        image_type = str(self.get_parameter('image_type').value).lower()
        if image_type == 'raw':
            message_type = Image
        elif image_type == 'compressed':
            message_type = CompressedImage
        else:
            raise ValueError(
                "image_type must be either 'raw' or 'compressed'."
            )
        self._image_group = ReentrantCallbackGroup()
        self._service_group = MutuallyExclusiveCallbackGroup()
        self._subscription = self.create_subscription(
            message_type,
            str(self.get_parameter('image_topic').value),
            self._image_received,
            qos_profile_sensor_data,
            callback_group=self._image_group,
        )
        self._service = self.create_service(
            GetPlan,
            str(self.get_parameter('service_name').value),
            self._infer,
            callback_group=self._service_group,
        )
        self.get_logger().info(
            f'PointGoal inference ready with '
            f"{self.get_parameter('backend').value} backend."
        )

    def _warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning >= self._warning_throttle:
            self.get_logger().warning(message)
            self._last_warning = now

    def _image_received(self, message) -> None:
        stamp_ns = stamp_to_nanoseconds(message.header.stamp)
        try:
            image = decode_image(message)
        except BagConversionError as error:
            self._warn(f'Skipping undecodable image: {error}')
            return
        if not self._history.append(stamp_ns, image):
            self._warn('Skipping out-of-order camera image.')

    def _infer(self, request, response):
        if request.goal.header.frame_id != self._robot_frame:
            self._warn(
                f"Rejecting PointGoal in frame '{request.goal.header.frame_id}'; "
                f"expected '{self._robot_frame}'."
            )
            return response
        observations, newest_stamp = self._history.snapshot()
        now_ns = self.get_clock().now().nanoseconds
        if not history_is_fresh(
            observations,
            newest_stamp,
            required_count=self._history.capacity,
            now_ns=now_ns,
            maximum_age_sec=self._maximum_image_age,
        ):
            self._warn('Rejecting PointGoal because camera context is incomplete or stale.')
            return response
        goal = (
            float(request.goal.pose.position.x),
            float(request.goal.pose.position.y),
        )
        try:
            with self._inference_lock:
                prediction = self._backend.predict_point_goal(
                    observations,
                    goal,
                )
        except ModelBackendError as error:
            self._warn(f'PointGoal inference failed: {error}')
            return response
        response.plan = prediction_to_path(
            prediction,
            frame_id=self._robot_frame,
            stamp=request.goal.header.stamp,
        )
        return response


def main(args=None) -> None:
    """Run PointGoal inference with independent image and service callbacks."""
    rclpy.init(args=args)
    node = PointGoalInferenceNode()
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
