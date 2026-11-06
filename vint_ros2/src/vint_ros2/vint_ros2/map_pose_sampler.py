"""Publish a map-frame robot pose at each retained camera timestamp."""

import time

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from tf2_ros import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformListener


def pose_from_transform(transform, stamp, frame_id: str) -> PoseStamped:
    """Create a pose from the translation and rotation of a transform."""
    pose = PoseStamped()
    pose.header.stamp = stamp
    pose.header.frame_id = frame_id
    pose.pose.position.x = transform.transform.translation.x
    pose.pose.position.y = transform.transform.translation.y
    pose.pose.position.z = transform.transform.translation.z
    pose.pose.orientation = transform.transform.rotation
    return pose


def create_sampler_executor(context=None) -> MultiThreadedExecutor:
    """Create an executor that can receive TF while an image lookup waits."""
    return MultiThreadedExecutor(num_threads=2, context=context)


class MapPoseSampler(Node):
    """Look up and publish the robot map pose when each image arrives."""

    def __init__(self, **node_kwargs) -> None:
        """Initialize parameters, TF access, and the camera subscription."""
        super().__init__('vint_map_pose_sampler', **node_kwargs)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('image_type', 'raw')
        self.declare_parameter('output_topic', '/vint/map_pose')
        self.declare_parameter('fixed_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('transform_timeout_sec', 0.1)
        self.declare_parameter('warning_throttle_sec', 2.0)

        self._fixed_frame = str(self.get_parameter('fixed_frame').value)
        self._robot_frame = str(self.get_parameter('robot_frame').value)
        self._transform_timeout = Duration(
            seconds=float(self.get_parameter('transform_timeout_sec').value)
        )
        self._warning_throttle = float(
            self.get_parameter('warning_throttle_sec').value
        )
        self._last_warning_time = float('-inf')

        output_topic = str(self.get_parameter('output_topic').value)
        self._publisher = self.create_publisher(PoseStamped, output_topic, 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        image_topic = str(self.get_parameter('image_topic').value)
        image_type = str(self.get_parameter('image_type').value).lower()
        if image_type == 'raw':
            message_type = Image
        elif image_type == 'compressed':
            message_type = CompressedImage
        else:
            raise ValueError(
                "image_type must be either 'raw' or 'compressed'."
            )
        self._subscription = self.create_subscription(
            message_type,
            image_topic,
            self._image_received,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f'Sampling {self._fixed_frame} -> {self._robot_frame} at '
            f'{image_topic} timestamps onto {output_topic}.'
        )

    def _image_received(self, message) -> None:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._fixed_frame,
                self._robot_frame,
                Time.from_msg(message.header.stamp),
                timeout=self._transform_timeout,
            )
        except TransformException as error:
            now = time.monotonic()
            if now - self._last_warning_time >= self._warning_throttle:
                self.get_logger().warning(
                    'Skipping image without a timestamp-aligned transform: '
                    f'{error}'
                )
                self._last_warning_time = now
            return
        self._publisher.publish(pose_from_transform(
            transform,
            message.header.stamp,
            self._fixed_frame,
        ))


def main(args=None) -> None:
    """Run the timestamp-aligned map-pose sampler."""
    rclpy.init(args=args)
    node = MapPoseSampler()
    executor = create_sampler_executor()
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
