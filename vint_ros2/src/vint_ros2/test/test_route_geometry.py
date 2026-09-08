"""Tests for route selection and PointGoal frame geometry."""

import math
from pathlib import Path

import pytest
import yaml

from vint_ros2.route_geometry import map_point_to_robot
from vint_ros2.route_geometry import point_ahead_on_path
from vint_ros2.route_geometry import project_to_polyline
from vint_ros2.route_geometry import robot_point_to_map
from vint_ros2.route_geometry import RouteGeometryError
from vint_ros2.route_geometry import select_route_goal
from vint_ros2.route_manifest import load_route_manifest
from vint_ros2.route_manifest import Pose2D


def _route(tmp_path: Path):
    data = {
        'schema_version': '1.0',
        'route_id': 'corner_route',
        'map_id': 'test_map',
        'frame_id': 'map',
        'source_waypoint_id': 'test.floor_1.a',
        'destination_waypoint_id': 'test.floor_1.b',
        'nodes': [
            {'index': 0, 'image': '0.jpg', 'stamp_ns': 1,
             'pose': {'x': 0.0, 'y': 0.0, 'yaw': 0.0}},
            {'index': 1, 'image': '1.jpg', 'stamp_ns': 2,
             'pose': {'x': 2.0, 'y': 0.0, 'yaw': 0.0}},
            {'index': 2, 'image': '2.jpg', 'stamp_ns': 3,
             'pose': {'x': 2.0, 'y': 2.0, 'yaw': math.pi / 2.0}},
            {'index': 3, 'image': '3.jpg', 'stamp_ns': 4,
             'pose': {'x': 2.0, 'y': 4.0, 'yaw': math.pi / 2.0}},
        ],
    }
    path = tmp_path / 'route.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    return load_route_manifest(path, require_images=False)


def test_projects_onto_l_shaped_polyline():
    projection = project_to_polyline(
        (2.3, 1.0),
        [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)],
    )

    assert projection.point == pytest.approx((2.0, 1.0))
    assert projection.distance == pytest.approx(0.3)
    assert projection.arc_length == pytest.approx(3.0)


def test_lookahead_is_measured_from_robot_path_projection():
    point = point_ahead_on_path(
        [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)],
        (1.5, 0.2),
        1.0,
    )

    assert point == pytest.approx((2.0, 0.5))


def test_route_goal_selection_is_monotonic(tmp_path):
    route = _route(tmp_path)
    path = [(0.0, 0.0), (2.0, 0.0), (2.0, 4.0)]

    first = select_route_goal(
        route,
        (0.2, 0.1),
        path,
        lookahead_distance=1.0,
        corridor_tolerance=0.5,
    )
    noisy_update = select_route_goal(
        route,
        (0.1, 0.0),
        path,
        lookahead_distance=0.1,
        corridor_tolerance=0.5,
        previous_goal_index=first.goal_index,
    )

    assert first.goal_index == 1
    assert noisy_update.goal_index == first.goal_index


def test_rejects_global_path_outside_registered_corridor(tmp_path):
    route = _route(tmp_path)

    with pytest.raises(RouteGeometryError, match='Global path leaves'):
        select_route_goal(
            route,
            (0.0, 0.0),
            [(0.0, 0.0), (0.0, 2.0), (2.0, 4.0)],
            lookahead_distance=1.0,
            corridor_tolerance=0.5,
        )


def test_pointgoal_frame_transforms_round_trip():
    robot_pose = Pose2D(x=10.0, y=-2.0, yaw=math.pi / 2.0)
    map_point = (8.0, 1.0)

    robot_point = map_point_to_robot(map_point, robot_pose)
    recovered = robot_point_to_map(robot_point, robot_pose)

    assert robot_point == pytest.approx((3.0, 2.0))
    assert recovered == pytest.approx(map_point)
