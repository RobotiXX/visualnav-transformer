"""Tests for timestamp-aligned map-pose publication helpers."""

from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped

from vint_ros2.map_pose_sampler import pose_from_transform


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
