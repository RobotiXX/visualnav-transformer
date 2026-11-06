"""Tests for timestamp-aligned map-pose publication helpers."""

from threading import Event
from threading import Thread
import time

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from tf2_ros import TransformBroadcaster

from vint_ros2.map_pose_sampler import create_sampler_executor
from vint_ros2.map_pose_sampler import MapPoseSampler
from vint_ros2.map_pose_sampler import pose_from_transform


class _ObservedMapPoseSampler(MapPoseSampler):
    """Expose when the image callback starts for concurrency testing."""

    def __init__(self, *args, **kwargs):
        self.image_callback_started = Event()
        super().__init__(*args, **kwargs)

    def _image_received(self, message) -> None:
        self.image_callback_started.set()
        super()._image_received(message)


def test_pose_from_transform_preserves_stamp_frame_and_transform():
    transform = TransformStamped()
    transform.transform.translation.x = 1.0
    transform.transform.translation.y = 2.0
    transform.transform.translation.z = 3.0
    transform.transform.rotation.z = 0.5
    transform.transform.rotation.w = 0.5
    stamp = Time(sec=7, nanosec=8)

    pose = pose_from_transform(transform, stamp, 'map')

    assert pose.header.stamp == stamp
    assert pose.header.frame_id == 'map'
    assert pose.pose.position.x == 1.0
    assert pose.pose.position.y == 2.0
    assert pose.pose.position.z == 3.0
    assert pose.pose.orientation.z == 0.5
    assert pose.pose.orientation.w == 0.5


def test_sampler_executor_allows_concurrent_tf_callbacks():
    context = Context()
    context.init()
    executor = create_sampler_executor(context=context)

    assert isinstance(executor, MultiThreadedExecutor)

    executor.shutdown()
    context.shutdown()


def test_transform_arriving_during_image_lookup_publishes_pose():
    context = Context()
    context.init()
    image_topic = '/test_vint/image'
    pose_topic = '/test_vint/map_pose'
    sampler = _ObservedMapPoseSampler(
        context=context,
        parameter_overrides=[
            Parameter('image_topic', value=image_topic),
            Parameter('image_type', value='compressed'),
            Parameter('output_topic', value=pose_topic),
            Parameter('fixed_frame', value='test_map'),
            Parameter('robot_frame', value='test_base_link'),
            Parameter('transform_timeout_sec', value=0.5),
        ],
    )
    source = Node('test_vint_pose_source', context=context)
    image_publisher = source.create_publisher(
        CompressedImage,
        image_topic,
        qos_profile_sensor_data,
    )
    transform_broadcaster = TransformBroadcaster(source)
    pose_received = Event()
    received_poses = []

    def receive_pose(message):
        received_poses.append(message)
        pose_received.set()

    source.create_subscription(PoseStamped, pose_topic, receive_pose, 10)
    executor = create_sampler_executor(context=context)
    executor.add_node(sampler)
    executor.add_node(source)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        deadline = time.monotonic() + 2.0
        while (
            image_publisher.get_subscription_count() == 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert image_publisher.get_subscription_count() > 0

        stamp = source.get_clock().now().to_msg()
        image = CompressedImage()
        image.header.stamp = stamp
        image_publisher.publish(image)
        assert sampler.image_callback_started.wait(1.0)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = 'test_map'
        transform.child_frame_id = 'test_base_link'
        transform.transform.translation.x = 4.0
        transform.transform.rotation.w = 1.0
        transform_broadcaster.sendTransform(transform)

        assert pose_received.wait(1.0)
        assert received_poses[-1].header.stamp == stamp
        assert received_poses[-1].header.frame_id == 'test_map'
        assert received_poses[-1].pose.position.x == 4.0
    finally:
        executor.shutdown(timeout_sec=1.0)
        spin_thread.join(timeout=1.0)
        sampler.destroy_node()
        source.destroy_node()
        context.shutdown()
