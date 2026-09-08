"""Command-line validation and splitting of converted ViNT trajectories."""

import argparse
from typing import Sequence

from vint_ros2.dataset_layout import create_splits
from vint_ros2.dataset_layout import discover_trajectories
from vint_ros2.dataset_layout import validate_trajectory
from vint_ros2.dataset_layout import write_splits


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the dataset preparation argument parser."""
    parser = argparse.ArgumentParser(
        description='Validate converted trajectories and create fixed splits.',
    )
    parser.add_argument('dataset_root')
    parser.add_argument('--split-output', required=True)
    parser.add_argument('--test-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Validate a dataset, print spacing, and write deterministic splits."""
    arguments = build_argument_parser().parse_args(argv)
    directories = discover_trajectories(arguments.dataset_root)
    summaries = tuple(validate_trajectory(path) for path in directories)
    splits = create_splits(
        [summary.name for summary in summaries],
        test_fraction=arguments.test_fraction,
        seed=arguments.seed,
    )
    output = write_splits(
        arguments.split_output,
        splits,
        overwrite=arguments.overwrite,
    )
    sample_count = sum(summary.samples for summary in summaries)
    path_length = sum(summary.path_length_m for summary in summaries)
    mean_spacing = sum(
        summary.mean_spacing_m for summary in summaries
    ) / len(summaries)
    print(
        f'Validated {len(summaries)} trajectories, {sample_count} samples, '
        f'{path_length:.2f} m. Mean per-trajectory spacing is '
        f'{mean_spacing:.3f} m.'
    )
    print(
        f'Wrote {len(splits.train)} train and {len(splits.test)} test '
        f'trajectories to {output}'
    )


if __name__ == '__main__':
    main()
