"""Tests for controlled Nav2 study-profile composition."""

from copy import deepcopy

import pytest

from vint_ros2.nav2_profile import compose_nav2_profile
from vint_ros2.nav2_profile import load_plugin_parameters
from vint_ros2.nav2_profile import Nav2ProfileError
from vint_ros2.nav2_profile import NavigationMode
from vint_ros2.nav2_profile import write_nav2_yaml


def _base():
    return {
        'controller_server': {
            'ros__parameters': {
                'controller_plugins': ['FollowPath', 'DoorTraversal'],
                'FollowPath': {'plugin': 'dwb_core::DWBLocalPlanner'},
                'DoorTraversal': {'plugin': 'dwb_core::DWBLocalPlanner'},
            },
        },
        'planner_server': {
            'ros__parameters': {
                'planner_plugins': ['GridBased'],
                'GridBased': {
                    'plugin': 'nav2_theta_star_planner/ThetaStarPlanner',
                },
            },
        },
    }


def test_classical_profile_does_not_mutate_base():
    base = _base()
    original = deepcopy(base)

    output = compose_nav2_profile(base, NavigationMode.CLASSICAL)

    assert output == original
    assert output is not base


def test_image_goal_replaces_only_stable_follow_path_id():
    output = compose_nav2_profile(_base(), 'image_goal_local')
    parameters = output['controller_server']['ros__parameters']

    assert parameters['FollowPath']['plugin'] == (
        'vint_nav2_plugins/ImageGoalController'
    )
    assert parameters['DoorTraversal']['plugin'] == (
        'dwb_core::DWBLocalPlanner'
    )
    assert output['planner_server']['ros__parameters']['GridBased'][
        'plugin'
    ] == 'nav2_theta_star_planner/ThetaStarPlanner'


def test_pointgoal_replaces_only_stable_gridbased_id():
    output = compose_nav2_profile(_base(), 'pointgoal_global')
    planner = output['planner_server']['ros__parameters']['GridBased']

    assert planner['plugin'] == (
        'vint_nav2_plugins/PointGoalGlobalPlanner'
    )
    assert planner['tail']['use_final_approach_orientation'] is False
    assert output['controller_server']['ros__parameters']['FollowPath'][
        'plugin'
    ] == 'dwb_core::DWBLocalPlanner'


def test_full_profile_changes_both_components():
    output = compose_nav2_profile(_base(), 'vint_full')

    assert output['controller_server']['ros__parameters']['FollowPath'][
        'plugin'
    ] == 'vint_nav2_plugins/ImageGoalController'
    assert output['planner_server']['ros__parameters']['GridBased'][
        'plugin'
    ] == 'vint_nav2_plugins/PointGoalGlobalPlanner'


def test_missing_stable_plugin_id_fails_closed():
    base = _base()
    base['planner_server']['ros__parameters']['planner_plugins'] = ['Other']

    with pytest.raises(Nav2ProfileError, match='GridBased'):
        compose_nav2_profile(base, 'pointgoal_global')


def test_written_profile_materializes_shared_yaml_values(tmp_path):
    base = _base()
    shared_critics = ['PathAlign', 'PathDist']
    base['controller_server']['ros__parameters']['FollowPath'][
        'critics'
    ] = shared_critics
    base['controller_server']['ros__parameters']['DoorTraversal'][
        'critics'
    ] = shared_critics

    output = write_nav2_yaml(tmp_path / 'profile.yaml', base)
    text = output.read_text(encoding='utf-8')

    assert '&id' not in text
    assert '*id' not in text


def test_explicit_plugin_config_controls_composed_parameters(tmp_path):
    snippet = tmp_path / 'controller.yaml'
    snippet.write_text(
        'controller_server:\n'
        '  ros__parameters:\n'
        '    ViNTImageGoal:\n'
        '      plugin: vint_nav2_plugins/ImageGoalController\n'
        '      maximum_linear_speed: 0.27\n',
        encoding='utf-8',
    )
    plugin = load_plugin_parameters(
        snippet,
        node_name='controller_server',
        plugin_name='ViNTImageGoal',
    )

    output = compose_nav2_profile(
        _base(),
        'image_goal_local',
        image_goal_controller=plugin,
    )

    assert output['controller_server']['ros__parameters']['FollowPath'][
        'maximum_linear_speed'
    ] == 0.27
