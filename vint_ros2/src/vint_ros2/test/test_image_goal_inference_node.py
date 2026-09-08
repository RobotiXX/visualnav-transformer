"""ROS integration test for registered-route mock image-goal inference."""

from io import BytesIO
from threading import Thread
import time

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Path
from PIL import Image as PilImage
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import yaml

from vint_ros2.image_goal_inference import ImageGoalInferenceNode


def _write_route(tmp_path):
    nodes = []
    for index, x in enumerate((0.0, 2.0)):
        image_name = f'{index}.jpg'
        PilImage.new('RGB', (2, 2), color=(index, 0, 0)).save(
            tmp_path / image_name
        )
        nodes.append({
            'index': index,
            'image': image_name,
            'stamp_ns': index + 1,
            'pose': {'x': x, 'y': 0.0, 'yaw': 0.0},
        })
    manifest = tmp_path / 'route.yaml'
    manifest.write_text(yaml.safe_dump({
        'schema_version': '1.0',
        'route_id': 'test_a_to_b',
        'map_id': 'test_map',
        'frame_id': 'map',
        'source_waypoint_id': 'test.floor_1.a',
        'destination_waypoint_id': 'test.floor_1.b',
        'nodes': nodes,
    }), encoding='utf-8')
    return manifest


def _compressed_image(stamp):
    buffer = BytesIO()
    PilImage.new('RGB', (2, 2), color=(5, 10, 15)).save(
        buffer, format='PNG'
    )
    message = CompressedImage()
    message.header.stamp = stamp
    message.header.frame_id = 'camera'
    message.format = 'png'
    message.data = buffer.getvalue()
    return message


def test_registered_plan_produces_robot_frame_prediction(tmp_path):
    rclpy.init()
    suffix = str(time.monotonic_ns())
    image_topic = f'/test_vint_image_goal_camera_{suffix}'
    plan_topic = f'/test_vint_image_goal_plan_{suffix}'
    prediction_topic = f'/test_vint_image_goal_prediction_{suffix}'
    inference = ImageGoalInferenceNode(parameter_overrides=[
        Parameter('route_manifest', value=str(_write_route(tmp_path))),
        Parameter('context_size', value=0),
        Parameter('image_topic', value=image_topic),
        Parameter('image_type', value='compressed'),
        Parameter('global_plan_topic', value=plan_topic),
        Parameter('prediction_topic', value=prediction_topic),
        Parameter('inference_rate_hz', value=20.0),
        Parameter('maximum_image_age_sec', value=2.0),
    ])
    client = rclpy.create_node(f'test_vint_image_goal_client_{suffix}')
    image_publisher = client.create_publisher(
        CompressedImage, image_topic, qos_profile_sensor_data
    )
    plan_publisher = client.create_publisher(Path, plan_topic, 10)
    predictions = []
    subscription = client.create_subscription(
        Path, prediction_topic, predictions.append, 10
    )
    broadcaster = StaticTransformBroadcaster(client)
    transform = TransformStamped()
    transform.header.frame_id = 'map'
    transform.child_frame_id = 'base_link'
    transform.transform.rotation.w = 1.0
    broadcaster.sendTransform(transform)

    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(inference)
    executor.add_node(client)
    thread = Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 2.0
        while (plan_publisher.get_subscription_count() == 0 or
               image_publisher.get_subscription_count() == 0) and (
                   time.monotonic() < deadline):
            time.sleep(0.01)
        assert plan_publisher.get_subscription_count() == 1
        assert image_publisher.get_subscription_count() == 1

        stamp = client.get_clock().now().to_msg()
        plan = Path()
        plan.header.frame_id = 'map'
        plan.header.stamp = stamp
        for x in (0.0, 1.0, 2.0):
            pose = PoseStamped()
            pose.header = plan.header
            pose.pose.position.x = x
            pose.pose.orientation.w = 1.0
            plan.poses.append(pose)
        plan_publisher.publish(plan)
        image_publisher.publish(_compressed_image(stamp))

        deadline = time.monotonic() + 3.0
        while not predictions and time.monotonic() < deadline:
            time.sleep(0.01)

        assert predictions
        assert predictions[-1].header.frame_id == 'base_link'
        assert len(predictions[-1].poses) == 6
        assert predictions[-1].poses[0].pose.position.x == 0.0
    finally:
        del subscription
        executor.shutdown(timeout_sec=2.0)
        thread.join(timeout=2.0)
        client.destroy_node()
        inference.destroy_node()
        rclpy.shutdown()
