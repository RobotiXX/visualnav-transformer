"""Convert timestamped ROS 2 bag messages into ViNT route data."""

from bisect import bisect_left
from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
import pickle
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image as PilImage
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import yaml

from vint_ros2.route_manifest import Pose2D


_TRAJECTORY_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
_SUPPORTED_IMAGE_TYPES = {
    'sensor_msgs/msg/Image',
    'sensor_msgs/msg/CompressedImage',
}
_SUPPORTED_POSE_TYPES = {
    'geometry_msgs/msg/PoseStamped',
    'nav_msgs/msg/Odometry',
}


class BagConversionError(RuntimeError):
    """Raised when recorded ROS data cannot form a valid ViNT trajectory."""


@dataclass(frozen=True)
class StampedMessage:
    """One deserialized ROS message with its effective timestamp."""

    stamp_ns: int
    message: Any


@dataclass(frozen=True)
class AlignedSample:
    """One image aligned to a planar pose."""

    stamp_ns: int
    image: PilImage.Image
    pose: Pose2D


@dataclass(frozen=True)
class ConversionRequest:
    """Inputs and semantic metadata for one bag conversion."""

    bag_uri: Path
    output_directory: Path
    trajectory_name: str
    image_topic: str
    pose_topic: str
    sample_rate_hz: float
    synchronization_tolerance_sec: float
    route_id: str
    map_id: str
    frame_id: str
    source_waypoint_id: str
    destination_waypoint_id: str


def message_stamp_ns(message: Any, fallback_ns: int) -> int:
    """Return a ROS header timestamp or the bag timestamp as a fallback."""
    header = getattr(message, 'header', None)
    stamp = getattr(header, 'stamp', None)
    if stamp is None:
        return fallback_ns
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else fallback_ns


def _storage_identifier(bag_uri: Path) -> str:
    metadata_path = bag_uri / 'metadata.yaml'
    if not metadata_path.is_file():
        raise BagConversionError(
            f'ROS 2 bag metadata does not exist: {metadata_path}'
        )
    try:
        metadata = yaml.safe_load(metadata_path.read_text(encoding='utf-8'))
        identifier = metadata['rosbag2_bagfile_information'][
            'storage_identifier'
        ]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        raise BagConversionError(
            f'Unable to read storage identifier from {metadata_path}.'
        ) from error
    if not isinstance(identifier, str) or not identifier:
        raise BagConversionError('Bag storage identifier is empty.')
    return identifier


def read_bag_topics(
    bag_uri: str | Path,
    topics: Sequence[str],
) -> Mapping[str, tuple[StampedMessage, ...]]:
    """Deserialize selected topics from a ROS 2 bag in timestamp order."""
    uri = Path(bag_uri).expanduser().resolve()
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(uri),
            storage_id=_storage_identifier(uri),
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )
    topic_types = {
        topic.name: topic.type
        for topic in reader.get_all_topics_and_types()
    }
    missing = sorted(set(topics) - topic_types.keys())
    if missing:
        raise BagConversionError(
            'Bag is missing required topics: ' + ', '.join(missing)
        )
    output: dict[str, list[StampedMessage]] = {
        topic: [] for topic in topics
    }
    message_types = {
        topic: get_message(topic_types[topic]) for topic in topics
    }
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    while reader.has_next():
        topic, serialized, bag_stamp = reader.read_next()
        message = deserialize_message(serialized, message_types[topic])
        output[topic].append(StampedMessage(
            stamp_ns=message_stamp_ns(message, bag_stamp),
            message=message,
        ))
    return {
        topic: tuple(sorted(messages, key=lambda item: item.stamp_ns))
        for topic, messages in output.items()
    }


def sample_at_rate(
    messages: Sequence[StampedMessage],
    rate_hz: float,
) -> tuple[StampedMessage, ...]:
    """Keep the first message in each fixed minimum-time interval."""
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise BagConversionError('Sample rate must be positive.')
    period_ns = max(1, round(1_000_000_000 / rate_hz))
    sampled: list[StampedMessage] = []
    previous_stamp: int | None = None
    for message in sorted(messages, key=lambda item: item.stamp_ns):
        if previous_stamp is None or message.stamp_ns - previous_stamp >= period_ns:
            sampled.append(message)
            previous_stamp = message.stamp_ns
    return tuple(sampled)


def align_nearest(
    primary: Sequence[StampedMessage],
    secondary: Sequence[StampedMessage],
    tolerance_sec: float,
) -> tuple[tuple[StampedMessage, StampedMessage], ...]:
    """Pair each primary message to its nearest secondary timestamp."""
    if not math.isfinite(tolerance_sec) or tolerance_sec < 0.0:
        raise BagConversionError(
            'Synchronization tolerance must be non-negative.'
        )
    if not secondary:
        return ()
    tolerance_ns = round(tolerance_sec * 1_000_000_000)
    ordered_secondary = sorted(secondary, key=lambda item: item.stamp_ns)
    stamps = [message.stamp_ns for message in ordered_secondary]
    output = []
    for message in sorted(primary, key=lambda item: item.stamp_ns):
        insertion = bisect_left(stamps, message.stamp_ns)
        candidates = []
        if insertion < len(stamps):
            candidates.append(ordered_secondary[insertion])
        if insertion > 0:
            candidates.append(ordered_secondary[insertion - 1])
        nearest = min(
            candidates,
            key=lambda item: (abs(item.stamp_ns - message.stamp_ns),
                              item.stamp_ns),
        )
        if abs(nearest.stamp_ns - message.stamp_ns) <= tolerance_ns:
            output.append((message, nearest))
    return tuple(output)


def decode_image(message: Any) -> PilImage.Image:
    """Decode a ROS raw or compressed image as an RGB Pillow image."""
    if hasattr(message, 'format') and not hasattr(message, 'encoding'):
        try:
            return PilImage.open(BytesIO(bytes(message.data))).convert('RGB')
        except (OSError, ValueError) as error:
            raise BagConversionError('Unable to decode compressed image.') from error

    encoding = str(getattr(message, 'encoding', '')).lower()
    formats = {
        'rgb8': (3, None),
        'bgr8': (3, (2, 1, 0)),
        'rgba8': (4, (0, 1, 2)),
        'bgra8': (4, (2, 1, 0)),
        'mono8': (1, None),
    }
    if encoding not in formats:
        raise BagConversionError(f"Unsupported raw image encoding '{encoding}'.")
    channels, reorder = formats[encoding]
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    if height <= 0 or width <= 0 or step < width * channels:
        raise BagConversionError('Raw image dimensions or row step are invalid.')
    data = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if data.size < height * step:
        raise BagConversionError('Raw image data is shorter than its metadata.')
    rows = data[:height * step].reshape(height, step)
    pixels = rows[:, :width * channels]
    if channels == 1:
        array = pixels.reshape(height, width)
        return PilImage.fromarray(array, mode='L').convert('RGB')
    array = pixels.reshape(height, width, channels)
    if reorder is not None:
        array = array[:, :, reorder]
    return PilImage.fromarray(array[:, :, :3], mode='RGB')


def _quaternion_to_yaw(orientation: Any) -> float:
    sin_yaw = 2.0 * (
        orientation.w * orientation.z + orientation.x * orientation.y
    )
    cos_yaw = 1.0 - 2.0 * (
        orientation.y * orientation.y + orientation.z * orientation.z
    )
    return math.atan2(sin_yaw, cos_yaw)


def pose_2d_from_message(message: Any) -> Pose2D:
    """Extract a planar pose from PoseStamped or Odometry-like messages."""
    pose = message.pose
    if hasattr(pose, 'pose'):
        pose = pose.pose
    return Pose2D(
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw=_quaternion_to_yaw(pose.orientation),
    )


def make_aligned_samples(
    image_messages: Sequence[StampedMessage],
    pose_messages: Sequence[StampedMessage],
    *,
    sample_rate_hz: float,
    synchronization_tolerance_sec: float,
) -> tuple[AlignedSample, ...]:
    """Decode rate-limited images that have a nearby registered pose."""
    sampled_images = sample_at_rate(image_messages, sample_rate_hz)
    pairs = align_nearest(
        sampled_images,
        pose_messages,
        synchronization_tolerance_sec,
    )
    return tuple(
        AlignedSample(
            stamp_ns=image.stamp_ns,
            image=decode_image(image.message),
            pose=pose_2d_from_message(pose.message),
        )
        for image, pose in pairs
    )


def _validate_request(request: ConversionRequest) -> None:
    if not _TRAJECTORY_NAME.fullmatch(request.trajectory_name):
        raise BagConversionError(
            'Trajectory name may contain letters, numbers, dots, dashes, and '
            'underscores, and must start with a letter or number.'
        )
    for field, value in (
        ('route_id', request.route_id),
        ('map_id', request.map_id),
        ('frame_id', request.frame_id),
        ('image_topic', request.image_topic),
        ('pose_topic', request.pose_topic),
        ('source_waypoint_id', request.source_waypoint_id),
        ('destination_waypoint_id', request.destination_waypoint_id),
    ):
        if not value:
            raise BagConversionError(f'{field} must not be empty.')
    for field, value in (
        ('source_waypoint_id', request.source_waypoint_id),
        ('destination_waypoint_id', request.destination_waypoint_id),
    ):
        components = value.split('.')
        if len(components) < 3 or any(not component for component in components):
            raise BagConversionError(f'{field} must be a full waypoint ID.')
    if request.source_waypoint_id == request.destination_waypoint_id:
        raise BagConversionError(
            'Source and destination waypoint IDs must be different.'
        )
    if not math.isfinite(request.sample_rate_hz) or request.sample_rate_hz <= 0:
        raise BagConversionError('Sample rate must be positive.')
    if (not math.isfinite(request.synchronization_tolerance_sec) or
            request.synchronization_tolerance_sec < 0):
        raise BagConversionError(
            'Synchronization tolerance must be non-negative.'
        )
    if request.output_directory.exists():
        raise BagConversionError(
            f'Output trajectory already exists: {request.output_directory}'
        )


def write_trajectory(
    request: ConversionRequest,
    samples: Sequence[AlignedSample],
) -> Path:
    """Atomically write ViNT training data and its registered route manifest."""
    _validate_request(request)
    if len(samples) < 2:
        raise BagConversionError(
            'At least two synchronized image/pose samples are required.'
        )
    output_parent = request.output_directory.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f'.{request.trajectory_name}.',
        dir=output_parent,
    ))
    try:
        positions = []
        yaws = []
        nodes = []
        previous_stamp = -1
        for index, sample in enumerate(samples):
            if sample.stamp_ns <= previous_stamp:
                raise BagConversionError(
                    'Aligned sample timestamps must be strictly increasing.'
                )
            previous_stamp = sample.stamp_ns
            image_name = f'{index}.jpg'
            sample.image.convert('RGB').save(
                temporary / image_name,
                format='JPEG',
                quality=95,
            )
            positions.append([sample.pose.x, sample.pose.y])
            yaws.append(sample.pose.yaw)
            nodes.append({
                'index': index,
                'image': image_name,
                'stamp_ns': sample.stamp_ns,
                'pose': {
                    'x': sample.pose.x,
                    'y': sample.pose.y,
                    'yaw': sample.pose.yaw,
                },
            })
        with (temporary / 'traj_data.pkl').open('wb') as stream:
            pickle.dump({
                'position': np.asarray(positions, dtype=np.float64),
                'yaw': np.asarray(yaws, dtype=np.float64),
            }, stream)
        manifest = {
            'schema_version': '1.0',
            'route_id': request.route_id,
            'map_id': request.map_id,
            'frame_id': request.frame_id,
            'source_waypoint_id': request.source_waypoint_id,
            'destination_waypoint_id': request.destination_waypoint_id,
            'nodes': nodes,
        }
        (temporary / 'route.yaml').write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding='utf-8',
        )
        temporary.rename(request.output_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return request.output_directory


def convert_bag(request: ConversionRequest) -> tuple[Path, int, int]:
    """Read a bag, synchronize its topics, and write one trajectory."""
    _validate_request(request)
    topics = read_bag_topics(
        request.bag_uri,
        (request.image_topic, request.pose_topic),
    )
    image_messages = topics[request.image_topic]
    samples = make_aligned_samples(
        image_messages,
        topics[request.pose_topic],
        sample_rate_hz=request.sample_rate_hz,
        synchronization_tolerance_sec=request.synchronization_tolerance_sec,
    )
    output = write_trajectory(request, samples)
    return output, len(samples), len(sample_at_rate(
        image_messages,
        request.sample_rate_hz,
    ))
