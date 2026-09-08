"""Command-line composition of RWD Nav2 and ViNT study profiles."""

import argparse
from pathlib import Path
from typing import Sequence

from vint_ros2.nav2_profile import compose_nav2_profile
from vint_ros2.nav2_profile import load_image_goal_controller
from vint_ros2.nav2_profile import load_nav2_yaml
from vint_ros2.nav2_profile import load_pointgoal_global_planner
from vint_ros2.nav2_profile import NavigationMode
from vint_ros2.nav2_profile import write_nav2_yaml


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the Nav2 profile composition parser."""
    parser = argparse.ArgumentParser(
        description=(
            'Replace the stable FollowPath/GridBased implementations in a '
            'rendered Nav2 YAML while preserving task-executor plugin IDs.'
        ),
    )
    parser.add_argument('base', type=Path, help='Rendered RWD Nav2 YAML.')
    parser.add_argument(
        '--mode',
        required=True,
        choices=[mode.value for mode in NavigationMode],
    )
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument(
        '--image-goal-controller-config',
        type=Path,
        help='Optional image_goal_controller.yaml override.',
    )
    parser.add_argument(
        '--pointgoal-planner-config',
        type=Path,
        help='Optional pointgoal_planner.yaml override.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Compose and write one non-overwriting Nav2 study profile."""
    arguments = build_argument_parser().parse_args(argv)
    config = compose_nav2_profile(
        load_nav2_yaml(arguments.base),
        arguments.mode,
        image_goal_controller=(
            load_image_goal_controller(arguments.image_goal_controller_config)
            if arguments.image_goal_controller_config else None
        ),
        pointgoal_global_planner=(
            load_pointgoal_global_planner(arguments.pointgoal_planner_config)
            if arguments.pointgoal_planner_config else None
        ),
    )
    output = write_nav2_yaml(arguments.output, config)
    print(f'Wrote {arguments.mode} Nav2 profile to {output}')


if __name__ == '__main__':
    main()
