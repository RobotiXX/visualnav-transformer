"""Tests for ROS 2 bag synchronization and trajectory conversion."""

from io import BytesIO
from pathlib import Path
import pickle

from geometry_msgs.msg import PoseStamped
import numpy as np
from PIL import Image as PilImage
import pytest
import rosbag2_py
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

from vint_ros2.bag_conversion import align_nearest
from vint_ros2.bag_conversion import AlignedSample
from vint_ros2.bag_conversion import BagConversionError
from vint_ros2.bag_conversion import ConversionRequest
from vint_ros2.bag_conversion import decode_image
from vint_ros2.bag_conversion import read_bag_topics
from vint_ros2.bag_conversion import message_stamp_ns
from vint_ros2.bag_conversion import sample_at_rate
from vint_ros2.bag_conversion import StampedMessage
from vint_ros2.bag_conversion import write_trajectory
from vint_ros2.route_manifest import load_route_manifest


def _stamped(stamp_ns: int) -> StampedMessage:
    return StampedMessage(stamp_ns=stamp_ns, message=object())


def _request(tmp_path: Path) -> ConversionRequest:
    return ConversionRequest(
        bag_uri=tmp_path / 'bag',
        output_directory=tmp_path / 'dataset' / 'a_to_b',
        trajectory_name='a_to_b',
        image_topic='/camera/image/compressed',
        pose_topic='/vint/map_pose',
        sample_rate_hz=4.0,
        synchronization_tolerance_sec=0.05,
        route_id='a_to_b',
        map_id='test_map',
        frame_id='map',
        source_waypoint_id='test.floor_1.a',
        destination_waypoint_id='test.floor_1.b',
    )


def test_message_stamp_prefers_valid_header():
    message = PoseStamped()
    message.header.stamp.sec = 2
    message.header.stamp.nanosec = 3

    assert message_stamp_ns(message, 99) == 2_000_000_003


def test_sample_at_rate_uses_minimum_timestamp_interval():
    messages = tuple(_stamped(value) for value in (
        0, 100_000_000, 250_000_000, 499_000_000, 500_000_000,
    ))

    sampled = sample_at_rate(messages, 4.0)

    assert [message.stamp_ns for message in sampled] == [
        0, 250_000_000, 500_000_000,
    ]


def test_nearest_alignment_rejects_pairs_outside_tolerance():
    primary = (_stamped(1_000_000_000), _stamped(2_000_000_000))
    secondary = (_stamped(1_020_000_000), _stamped(2_200_000_000))

    pairs = align_nearest(primary, secondary, tolerance_sec=0.05)

    assert len(pairs) == 1
    assert pairs[0][1].stamp_ns == 1_020_000_000


def test_decodes_raw_bgr_image_with_padded_rows():
    message = Image()
    message.width = 2
    message.height = 1
    message.encoding = 'bgr8'
    message.step = 8
    message.data = bytes([0, 0, 255, 0, 255, 0, 99, 99])

    image = decode_image(message)

    assert image.mode == 'RGB'
    assert image.getpixel((0, 0)) == (255, 0, 0)
    assert image.getpixel((1, 0)) == (0, 255, 0)


def test_decodes_compressed_image():
    source = PilImage.new('RGB', (2, 1), color=(10, 20, 30))
    buffer = BytesIO()
    source.save(buffer, format='PNG')
    message = CompressedImage()
    message.format = 'png'
    message.data = buffer.getvalue()

    image = decode_image(message)

    assert image.getpixel((0, 0)) == (10, 20, 30)


def test_writes_training_layout_and_registered_route(tmp_path):
    samples = (
        AlignedSample(1, PilImage.new('RGB', (2, 2)),
                      pose=_pose(0.0, 0.0, 0.0)),
        AlignedSample(2, PilImage.new('RGB', (2, 2)),
                      pose=_pose(1.0, 0.5, 0.2)),
    )

    output = write_trajectory(_request(tmp_path), samples)

    route = load_route_manifest(output / 'route.yaml')
    with (output / 'traj_data.pkl').open('rb') as stream:
        trajectory = pickle.load(stream)
    assert len(route.nodes) == 2
    assert route.direction == ('test.floor_1.a', 'test.floor_1.b')
    assert trajectory['position'] == pytest.approx(
        np.array([[0.0, 0.0], [1.0, 0.5]])
    )
    assert trajectory['yaw'] == pytest.approx(np.array([0.0, 0.2]))


def test_refuses_to_overwrite_existing_trajectory(tmp_path):
    request = _request(tmp_path)
    request.output_directory.mkdir(parents=True)

    with pytest.raises(BagConversionError, match='already exists'):
        write_trajectory(request, ())


def test_reads_selected_topics_from_real_rosbag2(tmp_path):
    bag = tmp_path / 'bag'
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )
    for name, type_name in (
        ('/camera', 'sensor_msgs/msg/Image'),
        ('/pose', 'geometry_msgs/msg/PoseStamped'),
    ):
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=name,
            type=type_name,
            serialization_format='cdr',
        ))
    image = Image(width=1, height=1, encoding='rgb8', step=3,
                  data=bytes([1, 2, 3]))
    image.header.stamp.sec = 1
    pose = PoseStamped()
    pose.header.stamp.sec = 1
    writer.write('/camera', serialize_message(image), 1_000_000_000)
    writer.write('/pose', serialize_message(pose), 1_000_000_010)
    del writer

    messages = read_bag_topics(bag, ('/camera', '/pose'))

    assert messages['/camera'][0].stamp_ns == 1_000_000_000
    assert bytes(messages['/camera'][0].message.data) == bytes([1, 2, 3])
    assert messages['/pose'][0].stamp_ns == 1_000_000_000


def _pose(x: float, y: float, yaw: float):
    from vint_ros2.route_manifest import Pose2D
    return Pose2D(x=x, y=y, yaw=yaw)
