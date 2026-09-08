"""Dependency-light geometry for visual-route selection and PointGoal use."""

from bisect import bisect_left
from dataclasses import dataclass
import math
from typing import Sequence

from vint_ros2.route_manifest import Pose2D
from vint_ros2.route_manifest import RouteManifest


Point2D = tuple[float, float]
_EPSILON = 1.0e-9


class RouteGeometryError(ValueError):
    """Raised when a route cannot safely support the requested path."""


@dataclass(frozen=True)
class PolylineProjection:
    """The nearest point and arc coordinate on a polyline."""

    point: Point2D
    distance: float
    arc_length: float
    segment_index: int
    segment_fraction: float


@dataclass(frozen=True)
class RouteSelection:
    """A monotonic visual goal selection for one global-plan update."""

    current_index: int
    goal_index: int
    lookahead_point: Point2D
    robot_route_distance: float
    maximum_plan_route_distance: float


def _validate_points(points: Sequence[Point2D], field: str) -> None:
    if len(points) < 2:
        raise RouteGeometryError(f'{field} must contain at least two points.')
    for point in points:
        if len(point) != 2 or not all(math.isfinite(value) for value in point):
            raise RouteGeometryError(f'{field} contains an invalid point.')


def cumulative_lengths(points: Sequence[Point2D]) -> tuple[float, ...]:
    """Return the cumulative arc length at every polyline vertex."""
    _validate_points(points, 'Polyline')
    lengths = [0.0]
    for first, second in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.dist(first, second))
    if lengths[-1] <= _EPSILON:
        raise RouteGeometryError('Polyline has zero total length.')
    return tuple(lengths)


def project_to_polyline(
    point: Point2D,
    polyline: Sequence[Point2D],
) -> PolylineProjection:
    """Project a point onto the nearest non-degenerate polyline segment."""
    if len(point) != 2 or not all(math.isfinite(value) for value in point):
        raise RouteGeometryError('Projection point must be finite.')
    lengths = cumulative_lengths(polyline)
    best: PolylineProjection | None = None
    for index, (first, second) in enumerate(zip(polyline, polyline[1:])):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        squared_length = dx * dx + dy * dy
        if squared_length <= _EPSILON:
            continue
        fraction = max(0.0, min(
            1.0,
            ((point[0] - first[0]) * dx +
             (point[1] - first[1]) * dy) / squared_length,
        ))
        projected = (first[0] + fraction * dx, first[1] + fraction * dy)
        distance = math.dist(point, projected)
        candidate = PolylineProjection(
            point=projected,
            distance=distance,
            arc_length=lengths[index] + fraction * math.sqrt(squared_length),
            segment_index=index,
            segment_fraction=fraction,
        )
        if best is None or candidate.distance < best.distance:
            best = candidate
    if best is None:
        raise RouteGeometryError('Polyline contains no non-degenerate segment.')
    return best


def point_at_arc_length(
    polyline: Sequence[Point2D],
    arc_length: float,
) -> Point2D:
    """Interpolate a point at a clamped arc coordinate on a polyline."""
    if not math.isfinite(arc_length):
        raise RouteGeometryError('Arc length must be finite.')
    lengths = cumulative_lengths(polyline)
    target = max(0.0, min(arc_length, lengths[-1]))
    index = min(bisect_left(lengths, target), len(lengths) - 1)
    if index == 0:
        return polyline[0]
    first_index = index - 1
    segment_length = lengths[index] - lengths[first_index]
    if segment_length <= _EPSILON:
        return polyline[index]
    fraction = (target - lengths[first_index]) / segment_length
    first = polyline[first_index]
    second = polyline[index]
    return (
        first[0] + fraction * (second[0] - first[0]),
        first[1] + fraction * (second[1] - first[1]),
    )


def point_ahead_on_path(
    path: Sequence[Point2D],
    robot_position: Point2D,
    lookahead_distance: float,
) -> Point2D:
    """Return a metric lookahead point measured from the path projection."""
    if not math.isfinite(lookahead_distance) or lookahead_distance < 0.0:
        raise RouteGeometryError('Lookahead distance must be non-negative.')
    projection = project_to_polyline(robot_position, path)
    return point_at_arc_length(
        path,
        projection.arc_length + lookahead_distance,
    )


def maximum_corridor_distance(
    path: Sequence[Point2D],
    route: Sequence[Point2D],
) -> float:
    """Return the maximum sampled path-point distance from a route polyline."""
    _validate_points(path, 'Path')
    _validate_points(route, 'Route')
    return max(project_to_polyline(point, route).distance for point in path)


def select_route_goal(
    route: RouteManifest,
    robot_position: Point2D,
    global_path: Sequence[Point2D],
    *,
    lookahead_distance: float,
    corridor_tolerance: float,
    previous_goal_index: int = 0,
) -> RouteSelection:
    """
    Select a forward-only route image for a registered global path.

    The robot and every sampled global-path pose must remain inside the visual
    route corridor. The returned route index never precedes the previous goal
    index, even when localization noise changes the nearest route segment.
    """
    if not math.isfinite(corridor_tolerance) or corridor_tolerance <= 0.0:
        raise RouteGeometryError('Corridor tolerance must be positive.')
    if previous_goal_index < 0 or previous_goal_index >= len(route.nodes):
        raise RouteGeometryError('Previous route goal index is out of range.')

    route_points = tuple((node.pose.x, node.pose.y) for node in route.nodes)
    route_lengths = cumulative_lengths(route_points)
    robot_projection = project_to_polyline(robot_position, route_points)
    if robot_projection.distance > corridor_tolerance:
        raise RouteGeometryError(
            'Robot is outside the registered visual-route corridor: '
            f'{robot_projection.distance:.3f} m > {corridor_tolerance:.3f} m.'
        )

    maximum_distance = maximum_corridor_distance(global_path, route_points)
    if maximum_distance > corridor_tolerance:
        raise RouteGeometryError(
            'Global path leaves the registered visual-route corridor: '
            f'{maximum_distance:.3f} m > {corridor_tolerance:.3f} m.'
        )

    lookahead_point = point_ahead_on_path(
        global_path,
        robot_position,
        lookahead_distance,
    )
    lookahead_projection = project_to_polyline(lookahead_point, route_points)
    desired_index = bisect_left(route_lengths, lookahead_projection.arc_length)
    desired_index = min(desired_index, len(route.nodes) - 1)
    current_index = bisect_left(route_lengths, robot_projection.arc_length)
    current_index = min(current_index, len(route.nodes) - 1)
    goal_index = max(previous_goal_index, current_index, desired_index)

    return RouteSelection(
        current_index=current_index,
        goal_index=goal_index,
        lookahead_point=lookahead_point,
        robot_route_distance=robot_projection.distance,
        maximum_plan_route_distance=maximum_distance,
    )


def map_point_to_robot(point: Point2D, robot_pose: Pose2D) -> Point2D:
    """Transform a map-frame point into a planar robot frame."""
    dx = point[0] - robot_pose.x
    dy = point[1] - robot_pose.y
    cosine = math.cos(robot_pose.yaw)
    sine = math.sin(robot_pose.yaw)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def robot_point_to_map(point: Point2D, robot_pose: Pose2D) -> Point2D:
    """Transform a planar robot-frame point into the map frame."""
    cosine = math.cos(robot_pose.yaw)
    sine = math.sin(robot_pose.yaw)
    return (
        robot_pose.x + cosine * point[0] - sine * point[1],
        robot_pose.y + sine * point[0] + cosine * point[1],
    )
