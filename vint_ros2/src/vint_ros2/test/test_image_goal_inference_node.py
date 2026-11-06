"""ROS integration test for registered-route mock image-goal inference."""

from io import BytesIO
from threading import Thread
import time

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from PIL import Image as PilImage
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import yaml

import vint_ros2.image_goal_inference as image_goal_module
from vint_ros2.image_goal_inference import ImageGoalInferenceNode
from vint_ros2.image_goal_inference import RouteSelectionError
from vint_ros2.image_goal_inference import select_route_for_plan
from vint_ros2.image_goal_inference import select_topomap_prediction
from vint_ros2.model_backend import ModelPrediction
from vint_ros2.model_backend import TrajectoryPoint
from vint_ros2.route_manifest import load_route_manifest


def test_main_does_not_repeat_shutdown_after_signal_handler(monkeypatch):
    events = []

    class FakeNode:
        def destroy_node(self):
            events.append('destroy_node')
            raise KeyboardInterrupt

    class FakeExecutor:
        def __init__(self, *, num_threads):
            assert num_threads == 2

        def add_node(self, node):
            assert isinstance(node, FakeNode)

        def spin(self):
            raise KeyboardInterrupt

        def shutdown(self):
            events.append('executor_shutdown')

    monkeypatch.setattr(image_goal_module.rclpy, 'init', lambda args: None)
    monkeypatch.setattr(image_goal_module.rclpy, 'ok', lambda: False)
    monkeypatch.setattr(
        image_goal_module.rclpy,
        'shutdown',
        lambda: events.append('rclpy_shutdown'),
    )
    monkeypatch.setattr(image_goal_module, 'ImageGoalInferenceNode', FakeNode)
    monkeypatch.setattr(image_goal_module, 'MultiThreadedExecutor', FakeExecutor)

    image_goal_module.main(args=[])

    assert events == ['executor_shutdown', 'destroy_node']


def _prediction(distance, x):
    return ModelPrediction(
        distance=distance,
        trajectory=(TrajectoryPoint(x=x, y=0.0),),
    )


def test_topomap_selection_matches_original_distance_and_advance_rule():
    predictions = (
        _prediction(5.0, 0.0),
        _prediction(2.0, 1.0),
        _prediction(4.0, 2.0),
    )

    selected, closest = select_topomap_prediction(
        predictions,
        start_index=7,
        goal_index=12,
        close_threshold=3.0,
    )

    assert selected is predictions[2]
    assert closest == 9


def test_topomap_selection_stays_on_distant_best_match():
    predictions = (
        _prediction(5.0, 0.0),
        _prediction(4.0, 1.0),
        _prediction(6.0, 2.0),
    )

    selected, closest = select_topomap_prediction(
        predictions,
        start_index=7,
        goal_index=12,
        close_threshold=3.0,
    )

    assert selected is predictions[1]
    assert closest == 8


def _write_route(
    tmp_path,
    *,
    route_id='test_a_to_b',
    source_id='test.floor_1.a',
    destination_id='test.floor_1.b',
    source_x=0.0,
    destination_x=2.0,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    nodes = []
    for index, x in enumerate((source_x, destination_x)):
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
        'route_id': route_id,
        'map_id': 'test_map',
        'frame_id': 'map',
        'source_waypoint_id': source_id,
        'destination_waypoint_id': destination_id,
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


def _plan(start_x, destination_x):
    plan = Path()
    plan.header.frame_id = 'map'
    for x in (start_x, destination_x):
        pose = PoseStamped()
        pose.header = plan.header
        pose.pose.position.x = x
        pose.pose.orientation.w = 1.0
        plan.poses.append(pose)
    return plan


def test_route_selection_uses_current_map_and_both_plan_endpoints(tmp_path):
    first = load_route_manifest(_write_route(
        tmp_path / 'first',
        route_id='entrance_1_to_goal',
        source_id='test.floor_1.entrance_1',
        source_x=0.0,
        destination_x=20.0,
    ))
    second = load_route_manifest(_write_route(
        tmp_path / 'second',
        route_id='entrance_2_to_goal',
        source_id='test.floor_1.entrance_2',
        source_x=10.0,
        destination_x=20.0,
    ))

    selected = select_route_for_plan(
        (first, second),
        _plan(10.2, 20.5),
        map_id='test_map',
        source_tolerance_m=2.0,
        destination_tolerance_m=2.0,
        ambiguity_tolerance_m=0.25,
    )

    assert selected is second
    assert select_route_for_plan(
        (first, second),
        _plan(10.2, 20.5),
        map_id='another_map',
        source_tolerance_m=2.0,
        destination_tolerance_m=2.0,
        ambiguity_tolerance_m=0.25,
    ) is None


def test_route_selection_rejects_ambiguous_endpoint_matches(tmp_path):
    first = load_route_manifest(_write_route(
        tmp_path / 'first',
        route_id='first',
        source_id='test.floor_1.first',
        source_x=0.0,
        destination_x=2.0,
    ))
    second = load_route_manifest(_write_route(
        tmp_path / 'second',
        route_id='second',
        source_id='test.floor_1.second',
        source_x=0.1,
        destination_x=2.0,
    ))

    with pytest.raises(RouteSelectionError, match='ambiguously'):
        select_route_for_plan(
            (first, second),
            _plan(0.05, 2.0),
            map_id='test_map',
            source_tolerance_m=2.0,
            destination_tolerance_m=2.0,
            ambiguity_tolerance_m=0.25,
        )


def test_registered_plan_produces_robot_frame_prediction(tmp_path):
    rclpy.init()
    _write_route(tmp_path / 'route_a')
    _write_route(
        tmp_path / 'route_b',
        route_id='test_c_to_d',
        source_id='test.floor_1.c',
        destination_id='test.floor_1.d',
        source_x=10.0,
        destination_x=12.0,
    )
    suffix = str(time.monotonic_ns())
    image_topic = f'/test_vint_image_goal_camera_{suffix}'
    plan_topic = f'/test_vint_image_goal_plan_{suffix}'
    prediction_topic = f'/test_vint_image_goal_prediction_{suffix}'
    selected_route_topic = f'/test_vint_selected_route_{suffix}'
    inference = ImageGoalInferenceNode(parameter_overrides=[
        Parameter('route_directory', value=str(tmp_path)),
        Parameter('context_size', value=0),
        Parameter('image_topic', value=image_topic),
        Parameter('image_type', value='compressed'),
        Parameter('global_plan_topic', value=plan_topic),
        Parameter('prediction_topic', value=prediction_topic),
        Parameter('selected_route_topic', value=selected_route_topic),
        Parameter('inference_rate_hz', value=20.0),
    ])
    client = rclpy.create_node(f'test_vint_image_goal_client_{suffix}')
    image_publisher = client.create_publisher(
        CompressedImage, image_topic, qos_profile_sensor_data
    )
    plan_publisher = client.create_publisher(Path, plan_topic, 10)
    map_qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    map_publisher = client.create_publisher(String, '/map_name', map_qos)
    predictions = []
    selected_routes = []
    subscription = client.create_subscription(
        Path, prediction_topic, predictions.append, 10
    )
    selected_subscription = client.create_subscription(
        String,
        selected_route_topic,
        lambda message: selected_routes.append(message.data),
        map_qos,
    )
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(inference)
    executor.add_node(client)
    thread = Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 2.0
        while (plan_publisher.get_subscription_count() == 0 or
               image_publisher.get_subscription_count() == 0 or
               map_publisher.get_subscription_count() == 0) and (
                   time.monotonic() < deadline):
            time.sleep(0.01)
        assert plan_publisher.get_subscription_count() == 1
        assert image_publisher.get_subscription_count() == 1
        assert map_publisher.get_subscription_count() == 1

        stamp = client.get_clock().now().to_msg()
        map_publisher.publish(String(data='test_map'))
        time.sleep(0.1)
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
        old_image_stamp = CompressedImage().header.stamp
        image_publisher.publish(_compressed_image(old_image_stamp))

        deadline = time.monotonic() + 3.0
        while not predictions and time.monotonic() < deadline:
            time.sleep(0.01)

        assert predictions
        assert 'test_a_to_b' in selected_routes
        assert predictions[-1].header.frame_id == 'base_link'
        assert predictions[-1].header.stamp != old_image_stamp
        assert len(predictions[-1].poses) == 6
        assert predictions[-1].poses[0].pose.position.x == 0.0

        plan_publisher.publish(_plan(10.0, 12.0))
        deadline = time.monotonic() + 2.0
        while ('test_c_to_d' not in selected_routes and
               time.monotonic() < deadline):
            time.sleep(0.01)
        assert 'test_c_to_d' in selected_routes
    finally:
        del subscription
        del selected_subscription
        executor.shutdown(timeout_sec=2.0)
        thread.join(timeout=2.0)
        client.destroy_node()
        inference.destroy_node()
        rclpy.shutdown()
