"""Tests for reproducible navigation experiment metadata."""

import json

import pytest
import yaml

from vint_ros2.experiment_context import build_experiment_context
from vint_ros2.experiment_context import ExperimentContextError
from vint_ros2.experiment_context import write_experiment_context
from vint_ros2.write_experiment_context import main


def _parameters(path, node_name, **parameters):
    path.write_text(
        yaml.safe_dump({node_name: {'ros__parameters': parameters}}),
        encoding='utf-8',
    )
    return path


def test_classical_context_has_no_learned_components():
    context = build_experiment_context('classical')

    assert context['navigation_profile'] == 'classical'
    assert context['global_planner'] == 'theta_star'
    assert context['local_controller'] == 'dwb'
    assert context['inference'] == {}
    assert context['nav2_plugin_configurations'] == {}


def test_full_context_hashes_models_route_and_plugin_configs(tmp_path):
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'checkpoint')
    model_config = tmp_path / 'model.yaml'
    model_config.write_text('model_type: pointgoal_vint\n', encoding='utf-8')
    images = tmp_path / 'images'
    images.mkdir()
    (images / '000000.jpg').write_bytes(b'image zero')
    (images / '000001.jpg').write_bytes(b'image one')
    route = tmp_path / 'route.yaml'
    route.write_text(
        yaml.safe_dump({
            'schema_version': '1.0',
            'route_id': 'test_route',
            'map_id': 'test_map',
            'frame_id': 'map',
            'source_waypoint_id': 'test.floor_1.source',
            'destination_waypoint_id': 'test.floor_1.destination',
            'nodes': [
                {
                    'index': 0,
                    'image': 'images/000000.jpg',
                    'stamp_ns': 1,
                    'pose': {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
                },
                {
                    'index': 1,
                    'image': 'images/000001.jpg',
                    'stamp_ns': 2,
                    'pose': {'x': 1.0, 'y': 0.0, 'yaw': 0.0},
                },
            ],
        }),
        encoding='utf-8',
    )
    controller = tmp_path / 'controller.yaml'
    controller.write_text('controller: test\n', encoding='utf-8')
    planner = tmp_path / 'planner.yaml'
    planner.write_text('planner: test\n', encoding='utf-8')
    image_params = _parameters(
        tmp_path / 'image.yaml',
        'vint_image_goal_inference',
        backend='mock',
        checkpoint_path='',
        model_config_path='',
        device='cpu',
    )
    pointgoal_params = _parameters(
        tmp_path / 'pointgoal.yaml',
        'vint_pointgoal_inference',
        backend='torch',
        checkpoint_path=str(checkpoint),
        model_config_path=str(model_config),
        device='cuda',
    )

    context = build_experiment_context(
        'vint_full',
        image_goal_params=image_params,
        pointgoal_params=pointgoal_params,
        route_directory=tmp_path,
        image_goal_controller_config=controller,
        pointgoal_planner_config=planner,
    )

    assert context['navigation_profile'] == 'vint_full'
    pointgoal = context['inference']['pointgoal']
    assert pointgoal['checkpoint']['path'] == str(checkpoint.resolve())
    assert len(pointgoal['checkpoint']['sha256']) == 64
    route_directory = context['inference']['image_goal']['route_directory']
    assert route_directory['route_count'] == 1
    route_descriptor = route_directory['routes'][0]
    assert route_descriptor['image_count'] == 2
    assert route_descriptor['route_id'] == 'test_route'
    assert len(route_descriptor['content_sha256']) == 64
    assert set(context['nav2_plugin_configurations']) == {
        'image_goal_controller', 'pointgoal_planner'
    }


def test_torch_context_rejects_missing_checkpoint(tmp_path):
    parameters = _parameters(
        tmp_path / 'pointgoal.yaml',
        'vint_pointgoal_inference',
        backend='torch',
        checkpoint_path='',
        model_config_path='',
    )

    with pytest.raises(ExperimentContextError, match='requires checkpoint_path'):
        build_experiment_context(
            'pointgoal_global',
            pointgoal_params=parameters,
            pointgoal_planner_config=parameters,
        )


def test_context_writer_is_non_overwriting(tmp_path):
    output = write_experiment_context(
        tmp_path / 'context.json',
        build_experiment_context('classical'),
    )
    assert json.loads(output.read_text(encoding='utf-8'))[
        'navigation_profile'
    ] == 'classical'

    with pytest.raises(ExperimentContextError, match='already exists'):
        write_experiment_context(output, build_experiment_context('classical'))


def test_hardware_context_cli_rejects_mock_inference(tmp_path):
    """The study launcher cannot accidentally evaluate the mock backend."""
    image_params = _parameters(
        tmp_path / 'image.yaml',
        'vint_image_goal_inference',
        backend='mock',
        checkpoint_path='',
        model_config_path='',
        device='cpu',
    )
    images = tmp_path / 'images'
    images.mkdir()
    (images / '000000.jpg').write_bytes(b'image zero')
    (images / '000001.jpg').write_bytes(b'image one')
    route = tmp_path / 'route.yaml'
    route.write_text(
        yaml.safe_dump({
            'schema_version': '1.0',
            'route_id': 'test_route',
            'map_id': 'test_map',
            'frame_id': 'map',
            'source_waypoint_id': 'test.floor_1.source',
            'destination_waypoint_id': 'test.floor_1.destination',
            'nodes': [
                {
                    'index': 0,
                    'image': 'images/000000.jpg',
                    'stamp_ns': 1,
                    'pose': {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
                },
                {
                    'index': 1,
                    'image': 'images/000001.jpg',
                    'stamp_ns': 2,
                    'pose': {'x': 1.0, 'y': 0.0, 'yaw': 0.0},
                },
            ],
        }),
        encoding='utf-8',
    )

    with pytest.raises(SystemExit):
        main([
            '--mode', 'image_goal_local',
            '--image-goal-params', str(image_params),
            '--route-directory', str(tmp_path),
            '--image-goal-controller-config', str(image_params),
            '--require-torch',
            '--output', str(tmp_path / 'context.json'),
        ])
