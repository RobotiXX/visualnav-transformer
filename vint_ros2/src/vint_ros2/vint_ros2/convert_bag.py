"""Command-line rosbag2 to ViNT trajectory conversion."""

import argparse
from pathlib import Path
from typing import Sequence

from vint_ros2.bag_conversion import convert_bag
from vint_ros2.bag_conversion import ConversionRequest


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the bag-conversion argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            'Convert one ROS 2 bag into ViNT training data and a registered '
            'visual route.'
        ),
    )
    parser.add_argument('bag', type=Path, help='ROS 2 bag directory.')
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--trajectory-name', required=True)
    parser.add_argument('--image-topic', required=True)
    parser.add_argument('--pose-topic', default='/vint/map_pose')
    parser.add_argument('--sample-rate', type=float, default=4.0)
    parser.add_argument('--sync-tolerance', type=float, default=0.1)
    parser.add_argument('--route-id', required=True)
    parser.add_argument('--map-id', required=True)
    parser.add_argument('--frame-id', default='map')
    parser.add_argument('--source-waypoint-id', required=True)
    parser.add_argument('--destination-waypoint-id', required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Convert one ROS 2 bag and report retained synchronization samples."""
    arguments = build_argument_parser().parse_args(argv)
    request = ConversionRequest(
        bag_uri=arguments.bag,
        output_directory=arguments.output_root / arguments.trajectory_name,
        trajectory_name=arguments.trajectory_name,
        image_topic=arguments.image_topic,
        pose_topic=arguments.pose_topic,
        sample_rate_hz=arguments.sample_rate,
        synchronization_tolerance_sec=arguments.sync_tolerance,
        route_id=arguments.route_id,
        map_id=arguments.map_id,
        frame_id=arguments.frame_id,
        source_waypoint_id=arguments.source_waypoint_id,
        destination_waypoint_id=arguments.destination_waypoint_id,
    )
    output, retained, considered = convert_bag(request)
    print(
        f'Wrote {retained}/{considered} sampled image/pose pairs to {output}'
    )


if __name__ == '__main__':
    main()
