"""Tests for portable robot and simulation topic profiles."""

from pathlib import Path

import pytest
import yaml


CONFIG_DIRECTORY = Path(__file__).parents[1] / 'config'


def _load(name: str) -> dict:
    return yaml.safe_load(
        (CONFIG_DIRECTORY / name).read_text(encoding='utf-8')
    )


def test_robot_profiles_use_adapter_owned_camera_topics():
    collection = _load('scoutmini_route_collection.yaml')
    image_goal = _load('image_goal_inference.yaml')[
        'vint_image_goal_inference'
    ]['ros__parameters']
    pointgoal = _load('pointgoal_inference.yaml')[
        'vint_pointgoal_inference'
    ]['ros__parameters']

    assert collection['image_topic'] == '/camera/image/compressed'
    assert '/camera/camera_info' in collection['topics']
    assert '/odom' in collection['topics']
    assert '/cmd_vel' not in collection['topics']
    assert image_goal['image_topic'] == '/camera/image/compressed'
    assert image_goal['inference_rate_hz'] == 4.0
    assert image_goal['topomap_radius'] == 4
    assert image_goal['close_threshold'] == 3.0
    assert image_goal['route_source_tolerance_m'] == 3.0
    assert image_goal['route_destination_tolerance_m'] == 3.0
    assert image_goal['map_name_topic'] == '/map_name'
    assert 'lookahead_distance_m' not in image_goal
    assert 'route_corridor_tolerance_m' not in image_goal
    assert 'maximum_image_age_sec' not in image_goal
    assert 'maximum_plan_age_sec' not in image_goal
    assert pointgoal['image_topic'] == '/camera/image/compressed'


def test_simulation_profiles_use_raw_simulated_camera_and_clock():
    collection = _load('simulation_route_collection.yaml')
    image_goal = _load('image_goal_inference_sim.yaml')[
        'vint_image_goal_inference'
    ]['ros__parameters']
    pointgoal = _load('pointgoal_inference_sim.yaml')[
        'vint_pointgoal_inference'
    ]['ros__parameters']

    assert collection['image_topic'] == '/camera/image'
    assert collection['image_type'] == 'raw'
    assert image_goal['image_topic'] == '/camera/image'
    assert image_goal['image_type'] == 'raw'
    assert image_goal['use_sim_time'] is True
    assert image_goal['route_source_tolerance_m'] == 3.0
    assert image_goal['route_destination_tolerance_m'] == 3.0
    assert pointgoal['image_topic'] == '/camera/image'
    assert pointgoal['image_type'] == 'raw'
    assert pointgoal['use_sim_time'] is True
    assert pointgoal['backend'] == 'mock'
    assert pointgoal['horizon_distance_m'] <= 0.05


def test_release_model_profile_matches_official_checkpoint_architecture():
    profile = _load('vint_release_model.yaml')

    assert profile['model_type'] == 'vint'
    assert profile['context_size'] == 5
    assert profile['len_traj_pred'] == 5
    assert profile['image_size'] == [85, 64]
    assert profile['metric_waypoint_spacing_m'] == 0.20


def test_scoutmini_action_scale_matches_controller_speed():
    model = _load('vint_release_model.yaml')
    controller = yaml.safe_load(
        (
            CONFIG_DIRECTORY.parents[1]
            / 'vint_nav2_plugins'
            / 'config'
            / 'image_goal_controller.yaml'
        ).read_text(encoding='utf-8')
    )['controller_server']['ros__parameters']['ViNTImageGoal']

    assert controller['waypoint_index'] == 2
    assert controller['maximum_linear_speed'] == 0.8
    assert model['metric_waypoint_spacing_m'] * 4.0 == pytest.approx(
        controller['maximum_linear_speed']
    )
