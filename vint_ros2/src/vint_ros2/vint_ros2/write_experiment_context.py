"""CLI for writing task-associated ViNT experiment metadata."""

import argparse
from pathlib import Path
from typing import Sequence

from ament_index_python.packages import get_package_share_directory

from vint_ros2.experiment_context import build_experiment_context
from vint_ros2.experiment_context import write_experiment_context
from vint_ros2.nav2_profile import NavigationMode


def _default_plugin_config(filename: str) -> Path:
    return (
        Path(get_package_share_directory('vint_nav2_plugins'))
        / 'config'
        / filename
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the experiment-context command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            'Write immutable navigation profile, model, route, and plugin '
            'configuration metadata for RWD task and bag logs.'
        ),
    )
    parser.add_argument(
        '--mode',
        required=True,
        choices=[mode.value for mode in NavigationMode],
    )
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--image-goal-params', type=Path)
    parser.add_argument('--pointgoal-params', type=Path)
    parser.add_argument('--route-directory', type=Path)
    parser.add_argument(
        '--image-goal-controller-config',
        type=Path,
        default=_default_plugin_config('image_goal_controller.yaml'),
    )
    parser.add_argument(
        '--pointgoal-planner-config',
        type=Path,
        default=_default_plugin_config('pointgoal_planner.yaml'),
    )
    parser.add_argument(
        '--require-torch',
        action='store_true',
        help='Reject mock inference profiles for a hardware study run.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Build and write one experiment-context JSON file."""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    context = build_experiment_context(
        arguments.mode,
        image_goal_params=arguments.image_goal_params,
        pointgoal_params=arguments.pointgoal_params,
        route_directory=arguments.route_directory,
        image_goal_controller_config=arguments.image_goal_controller_config,
        pointgoal_planner_config=arguments.pointgoal_planner_config,
    )
    if arguments.require_torch:
        non_torch = [
            name for name, component in context['inference'].items()
            if component['backend'] != 'torch'
        ]
        if non_torch:
            parser.error(
                'Hardware study profiles require torch inference; invalid: '
                + ', '.join(non_torch)
            )
    output = write_experiment_context(arguments.output, context)
    print(f'Wrote {arguments.mode} experiment context to {output}')


if __name__ == '__main__':
    main()
