"""ROS integration test for mock PointGoal inference."""

from threading import Thread
import time

from nav_msgs.srv import GetPlan
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from vint_ros2.pointgoal_inference import PointGoalInferenceNode


def test_mock_service_returns_correlated_robot_frame_path():
    rclpy.init()
    suffix = str(time.monotonic_ns())
    image_topic = f'/test_vint_camera_{suffix}'
    service_name = f'/test_vint_pointgoal_{suffix}'
    inference = PointGoalInferenceNode(parameter_overrides=[
        Parameter('context_size', value=2),
        Parameter('image_topic', value=image_topic),
        Parameter('service_name', value=service_name),
        Parameter('maximum_image_age_sec', value=1.0),
    ])
    client_node = rclpy.create_node(f'test_vint_client_{suffix}')
    publisher = client_node.create_publisher(
        Image, image_topic, qos_profile_sensor_data
    )
    client = client_node.create_client(GetPlan, service_name)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(inference)
    executor.add_node(client_node)
    thread = Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        assert client.wait_for_service(timeout_sec=2.0)
        time.sleep(0.1)
        for _ in range(3):
            image = Image(width=1, height=1, encoding='rgb8', step=3,
                          data=bytes([1, 2, 3]))
            image.header.stamp = client_node.get_clock().now().to_msg()
            publisher.publish(image)
            time.sleep(0.03)

        request = GetPlan.Request()
        request.goal.header.frame_id = 'base_link'
        request.goal.header.stamp = client_node.get_clock().now().to_msg()
        request.goal.pose.position.x = 3.0
        request.goal.pose.position.y = 4.0
        future = client.call_async(request)
        deadline = time.monotonic() + 2.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)

        assert future.done()
        response = future.result()
        assert response.plan.header.frame_id == 'base_link'
        assert response.plan.header.stamp == request.goal.header.stamp
        assert len(response.plan.poses) == 6
        assert response.plan.poses[-1].pose.position.x == 0.75
        assert response.plan.poses[-1].pose.position.y == 1.0
    finally:
        executor.shutdown(timeout_sec=2.0)
        thread.join(timeout=2.0)
        client_node.destroy_node()
        inference.destroy_node()
        rclpy.shutdown()
