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
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import String

from vint_ros2.bag_conversion import BagConversionError
from vint_ros2.bag_conversion import decode_image
from vint_ros2.inference_common import create_backend
from vint_ros2.inference_common import declare_backend_parameters
from vint_ros2.inference_common import ImageHistory
from vint_ros2.inference_common import prediction_to_path
from vint_ros2.inference_common import stamp_to_nanoseconds
from vint_ros2.model_backend import ModelPrediction
from vint_ros2.model_backend import ModelBackendError
from vint_ros2.route_manifest import load_route_directory
from vint_ros2.route_manifest import RouteManifest


def select_topomap_prediction(
    predictions: tuple[ModelPrediction, ...],
    *,
    start_index: int,
    goal_index: int,
    close_threshold: float,
) -> tuple[ModelPrediction, int]:
    """Apply the original ViNT local-topomap localization and advance rule."""
    if not predictions:
        raise ModelBackendError('Topomap inference returned no predictions.')
    minimum_offset = min(
        range(len(predictions)),
        key=lambda index: predictions[index].distance,
    )
    localized_index = start_index + minimum_offset
    if predictions[minimum_offset].distance > close_threshold:
        return predictions[minimum_offset], localized_index
    selected_offset = min(minimum_offset + 1, len(predictions) - 1)
    return predictions[selected_offset], min(localized_index + 1, goal_index)


class RouteSelectionError(ValueError):
    """Raised when plan geometry cannot identify one route unambiguously."""


def select_route_for_plan(
    routes: tuple[RouteManifest, ...],
    plan: NavPath,
    *,
    map_id: str,
    source_tolerance_m: float,
    destination_tolerance_m: float,
    ambiguity_tolerance_m: float,
) -> RouteManifest | None:
    """Select a directional route using map, frame, and plan endpoints."""
    if not plan.poses or not map_id:
        return None
    source = plan.poses[0].pose.position
    destination = plan.poses[-1].pose.position
    matches = []
    for route in routes:
        if route.map_id != map_id or route.frame_id != plan.header.frame_id:
            continue
        route_source = route.nodes[0].pose
        route_destination = route.nodes[-1].pose
        source_error = math.dist(
            (source.x, source.y),
            (route_source.x, route_source.y),
        )
        destination_error = math.dist(
            (destination.x, destination.y),
            (route_destination.x, route_destination.y),
        )
        if (source_error <= source_tolerance_m and
                destination_error <= destination_tolerance_m):
            matches.append((source_error + destination_error, route))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1].route_id))
    if (len(matches) > 1 and
            matches[1][0] - matches[0][0] <= ambiguity_tolerance_m):
        raise RouteSelectionError(
            'Plan endpoints ambiguously match routes '
            f"'{matches[0][1].route_id}' and '{matches[1][1].route_id}'."
        )
    return matches[0][1]


class ImageGoalInferenceNode(Node):
    """Select registered goal images and publish image-goal ViNT paths."""

    def __init__(self, *, parameter_overrides=None) -> None:
        """Initialize registered route, image history, and inference loop."""
        super().__init__(
            'vint_image_goal_inference',
            parameter_overrides=parameter_overrides,
        )
        declare_backend_parameters(self)
        self.declare_parameter('route_directory', '')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('image_type', 'raw')
        self.declare_parameter('map_name_topic', '/map_name')
        self.declare_parameter(
            'global_plan_topic', '/vint/image_goal/global_plan'
        )
        self.declare_parameter(
            'prediction_topic', '/vint/image_goal/predicted_path'
        )
        self.declare_parameter(
            'selected_route_topic', '/vint/image_goal/selected_route'
        )
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('inference_rate_hz', 4.0)
        self.declare_parameter('topomap_radius', 4)
        self.declare_parameter('close_threshold', 3.0)
        self.declare_parameter('route_source_tolerance_m', 3.0)
        self.declare_parameter('route_destination_tolerance_m', 3.0)
        self.declare_parameter('route_selection_ambiguity_m', 0.25)
        self.declare_parameter('warning_throttle_sec', 2.0)

        route_directory = str(self.get_parameter('route_directory').value)
        if not route_directory:
            raise ValueError('route_directory is required.')
        self._routes = load_route_directory(route_directory)
        self._goal_images: dict[tuple[str, int], PilImage.Image] = {}
        self._backend = create_backend(self, 'image_goal')
        self._history = ImageHistory(self._backend.context_size)
        self._robot_frame = str(self.get_parameter('robot_frame').value)
        self._topomap_radius = int(
            self.get_parameter('topomap_radius').value
        )
        self._close_threshold = float(
            self.get_parameter('close_threshold').value
        )
        self._source_tolerance = float(
            self.get_parameter('route_source_tolerance_m').value
        )
        self._destination_tolerance = float(
            self.get_parameter('route_destination_tolerance_m').value
        )
        if self._topomap_radius < 0 or self._close_threshold < 0.0:
            raise ValueError(
                'topomap_radius and close_threshold must be non-negative.'
            )
        self._ambiguity_tolerance = float(
            self.get_parameter('route_selection_ambiguity_m').value
        )
        if self._source_tolerance <= 0.0:
            raise ValueError('route_source_tolerance_m must be positive.')
        if self._destination_tolerance <= 0.0:
            raise ValueError('route_destination_tolerance_m must be positive.')
        if self._ambiguity_tolerance < 0.0:
            raise ValueError(
                'route_selection_ambiguity_m must be non-negative.'
            )
        self._warning_throttle = float(
            self.get_parameter('warning_throttle_sec').value
        )
        self._last_warning = float('-inf')
        self._plan_lock = Lock()
        self._global_plan = None
        self._closest_node = 0
        self._plan_destination = None
        self._current_map_id = ''
        self._active_route: RouteManifest | None = None

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
        self._inference_group = MutuallyExclusiveCallbackGroup()
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._map_subscription = self.create_subscription(
            String,
            str(self.get_parameter('map_name_topic').value),
            self._map_received,
            map_qos,
            callback_group=self._image_group,
        )
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
        self._selected_route_publisher = self.create_publisher(
            String,
            str(self.get_parameter('selected_route_topic').value),
            map_qos,
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
            f'Image-goal inference loaded {len(self._routes)} directional '
            f'route(s) from {route_directory} with '
            f'{self.get_parameter("backend").value} backend.'
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

    def _map_received(self, message: String) -> None:
        map_id = message.data.strip()
        if not map_id:
            return
        with self._plan_lock:
            if map_id == self._current_map_id:
                return
            self._current_map_id = map_id
            self._active_route = None
            self._global_plan = None
            self._closest_node = 0
            self._plan_destination = None
        self._selected_route_publisher.publish(String(data=''))
        self.get_logger().info(
            f"ViNT route selection now uses map '{map_id}'."
        )

    def _plan_received(self, plan: NavPath) -> None:
        if not plan.poses:
            return
        destination = (
            plan.poses[-1].pose.position.x,
            plan.poses[-1].pose.position.y,
        )
        with self._plan_lock:
            current_map_id = self._current_map_id
            active_route = self._active_route
            destination_changed = (
                self._plan_destination is None or
                math.dist(destination, self._plan_destination) > 0.25
            )
        if not current_map_id:
            self._warn(
                'Ignoring the global plan until /map_name identifies the '
                'active occupancy map.'
            )
            return
        selected_route = active_route
        if destination_changed or selected_route is None:
            try:
                selected_route = select_route_for_plan(
                    self._routes,
                    plan,
                    map_id=current_map_id,
                    source_tolerance_m=self._source_tolerance,
                    destination_tolerance_m=self._destination_tolerance,
                    ambiguity_tolerance_m=self._ambiguity_tolerance,
                )
            except RouteSelectionError as error:
                self._warn(f'Ignoring global plan: {error}')
                selected_route = None
            if selected_route is None:
                had_active_route = False
                with self._plan_lock:
                    had_active_route = self._active_route is not None
                    self._active_route = None
                    self._global_plan = None
                    self._closest_node = 0
                    self._plan_destination = destination
                if had_active_route:
                    self._selected_route_publisher.publish(String(data=''))
                self._warn(
                    'Ignoring a global plan that does not match a registered '
                    f'directional route on map {current_map_id}.'
                )
                return
        with self._plan_lock:
            if current_map_id != self._current_map_id:
                return
            route_changed = selected_route is not self._active_route
            if destination_changed or route_changed:
                self._closest_node = 0
            self._active_route = selected_route
            self._plan_destination = destination
            self._global_plan = plan
        if route_changed:
            self._selected_route_publisher.publish(String(
                data=selected_route.route_id,
            ))
            self.get_logger().info(
                f"Selected ViNT route '{selected_route.route_id}': "
                f'{selected_route.source_waypoint_id} -> '
                f'{selected_route.destination_waypoint_id}.'
            )

    def _goal_image(
        self,
        route: RouteManifest,
        index: int,
    ) -> PilImage.Image:
        key = (str(route.manifest_path), index)
        image = self._goal_images.get(key)
        if image is None:
            image = PilImage.open(
                route.nodes[index].image_path
            ).convert('RGB')
            self._goal_images[key] = image
        return image

    def _infer(self) -> None:
        observations, image_stamp = self._history.snapshot()
        if (len(observations) != self._history.capacity or
                image_stamp is None):
            return
        with self._plan_lock:
            plan = self._global_plan
            closest_node = self._closest_node
            route = self._active_route
        if (plan is None or route is None or
                plan.header.frame_id != route.frame_id):
            return
        goal_index = len(route.nodes) - 1
        start = max(closest_node - self._topomap_radius, 0)
        end = min(
            closest_node + self._topomap_radius + 1,
            goal_index,
        )
        candidate_indices = tuple(range(start, end + 1))
        try:
            predictions = self._backend.predict_image_goals(
                observations,
                tuple(
                    self._goal_image(route, index)
                    for index in candidate_indices
                ),
            )
            if len(predictions) != len(candidate_indices):
                raise ModelBackendError(
                    'Topomap inference result count does not match its goals.'
                )
            prediction, closest_node = select_topomap_prediction(
                predictions,
                start_index=start,
                goal_index=goal_index,
                close_threshold=self._close_threshold,
            )
        except (ModelBackendError, OSError) as error:
            self._warn(f'Image-goal inference skipped: {error}')
            return
        with self._plan_lock:
            if route is self._active_route:
                self._closest_node = closest_node
        # The ROS 1 node publishes a newly received waypoint after every
        # inference, even when its camera context has not changed.  Stamp the
        # prediction at inference time so the controller's one-second timeout
        # describes loss of inference output, as its ROS 1 subscriber timeout
        # did, rather than adding an image-age safety policy.
        stamp = self.get_clock().now().to_msg()
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
        try:
            executor.shutdown()
            node.destroy_node()
        except KeyboardInterrupt:
            # ros2 launch can forward a second SIGINT while cleanup is active.
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
