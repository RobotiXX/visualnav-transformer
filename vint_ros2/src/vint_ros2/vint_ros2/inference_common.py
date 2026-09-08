"""Shared ROS-independent state and ROS path conversion for inference nodes."""

from collections import deque
import math
from threading import Lock
from typing import Sequence

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from PIL import Image

from vint_ros2.model_backend import InferenceBackend
from vint_ros2.model_backend import MockInferenceBackend
from vint_ros2.model_backend import ModelPrediction
from vint_ros2.model_backend import TorchViNTBackend


class ImageHistory:
    """Thread-safe, ordered camera context with out-of-order rejection."""

    def __init__(self, context_size: int) -> None:
        """Allocate storage for the current image plus temporal context."""
        if context_size < 0:
            raise ValueError('Context size must be non-negative.')
        self._images = deque(maxlen=context_size + 1)
        self._stamps = deque(maxlen=context_size + 1)
        self._lock = Lock()

    def append(self, stamp_ns: int, image: Image.Image) -> bool:
        """Append a newer observation, returning false for old timestamps."""
        with self._lock:
            if self._stamps and stamp_ns <= self._stamps[-1]:
                return False
            self._stamps.append(stamp_ns)
            self._images.append(image)
            return True

    def snapshot(self) -> tuple[tuple[Image.Image, ...], int | None]:
        """Return immutable current history and newest timestamp."""
        with self._lock:
            newest = self._stamps[-1] if self._stamps else None
            return tuple(self._images), newest

    @property
    def capacity(self) -> int:
        """Return the required number of images for inference."""
        return self._images.maxlen


def stamp_to_nanoseconds(stamp: Time) -> int:
    """Convert a ROS time message to integer nanoseconds."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def prediction_to_path(
    prediction: ModelPrediction,
    *,
    frame_id: str,
    stamp: Time,
    include_origin: bool = True,
) -> Path:
    """Convert a robot-relative prediction into a stamped ROS path."""
    path = Path()
    path.header.frame_id = frame_id
    path.header.stamp = stamp
    points = list(prediction.trajectory)
    if include_origin:
        origin = PoseStamped()
        origin.header = path.header
        origin.pose.orientation.w = 1.0
        path.poses.append(origin)
    previous = (0.0, 0.0)
    for index, point in enumerate(points):
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x = point.x
        pose.pose.position.y = point.y
        if point.heading_x is not None and point.heading_y is not None:
            yaw = math.atan2(point.heading_y, point.heading_x)
        else:
            dx = point.x - previous[0]
            dy = point.y - previous[1]
            if abs(dx) + abs(dy) > 1.0e-9:
                yaw = math.atan2(dy, dx)
            elif index + 1 < len(points):
                yaw = math.atan2(
                    points[index + 1].y - point.y,
                    points[index + 1].x - point.x,
                )
            else:
                yaw = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        path.poses.append(pose)
        previous = point.x, point.y
    return path


def create_backend(node, mode: str) -> InferenceBackend:
    """Create a mock or lazy PyTorch backend from declared node parameters."""
    backend_name = str(node.get_parameter('backend').value).lower()
    if backend_name == 'mock':
        return MockInferenceBackend(
            context_size=int(node.get_parameter('context_size').value),
            horizon_points=int(node.get_parameter('horizon_points').value),
            horizon_distance_m=float(
                node.get_parameter('horizon_distance_m').value
            ),
        )
    if backend_name == 'torch':
        return TorchViNTBackend(
            mode=mode,
            checkpoint_path=str(node.get_parameter('checkpoint_path').value),
            config_path=str(node.get_parameter('model_config_path').value),
            device=str(node.get_parameter('device').value),
        )
    raise ValueError("backend must be either 'mock' or 'torch'.")


def declare_backend_parameters(node) -> None:
    """Declare the parameters shared by both inference roles."""
    node.declare_parameter('backend', 'mock')
    node.declare_parameter('context_size', 5)
    node.declare_parameter('horizon_points', 5)
    node.declare_parameter('horizon_distance_m', 1.25)
    node.declare_parameter('checkpoint_path', '')
    node.declare_parameter('model_config_path', '')
    node.declare_parameter('device', 'cuda')


def history_is_fresh(
    history: Sequence[Image.Image],
    newest_stamp_ns: int | None,
    *,
    required_count: int,
    now_ns: int,
    maximum_age_sec: float,
) -> bool:
    """Return true only for complete, non-future, recent image context."""
    if len(history) != required_count or newest_stamp_ns is None:
        return False
    age_ns = now_ns - newest_stamp_ns
    return 0 <= age_ns <= maximum_age_sec * 1_000_000_000
