"""Command-line validation for registered visual route manifests."""

import argparse
from typing import Sequence

from vint_ros2.route_geometry import cumulative_lengths
from vint_ros2.route_manifest import load_route_manifest


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the route-validation argument parser."""
    parser = argparse.ArgumentParser(
        description='Validate a metric-registered ViNT visual route.',
    )
    parser.add_argument('manifest', help='Path to the route YAML manifest.')
    parser.add_argument(
        '--allow-missing-images',
        action='store_true',
        help='Validate metadata without requiring image files.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Validate one route and print a compact summary."""
    arguments = build_argument_parser().parse_args(argv)
    route = load_route_manifest(
        arguments.manifest,
        require_images=not arguments.allow_missing_images,
    )
    points = tuple((node.pose.x, node.pose.y) for node in route.nodes)
    distance = cumulative_lengths(points)[-1]
    print(
        f"Valid route '{route.route_id}': {len(route.nodes)} nodes, "
        f'{distance:.2f} m, {route.source_waypoint_id} -> '
        f'{route.destination_waypoint_id}'
    )


if __name__ == '__main__':
    main()
