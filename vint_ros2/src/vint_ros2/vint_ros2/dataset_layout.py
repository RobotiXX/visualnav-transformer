"""Validate ViNT trajectories and create deterministic dataset splits."""

from dataclasses import dataclass
import math
from pathlib import Path
import pickle
import random
from typing import Mapping, Sequence

import numpy as np

from vint_ros2.route_manifest import load_route_manifest


class DatasetLayoutError(ValueError):
    """Raised when converted ViNT data has an invalid layout or shape."""


@dataclass(frozen=True)
class TrajectorySummary:
    """Validation and spacing statistics for one ViNT trajectory."""

    name: str
    samples: int
    path_length_m: float
    mean_spacing_m: float
    median_spacing_m: float
    has_registered_route: bool


@dataclass(frozen=True)
class DatasetSplits:
    """Deterministically assigned training and test trajectory names."""

    train: tuple[str, ...]
    test: tuple[str, ...]


def _load_trajectory_data(path: Path) -> Mapping:
    try:
        with path.open('rb') as stream:
            data = pickle.load(stream)
    except (OSError, pickle.PickleError, EOFError) as error:
        raise DatasetLayoutError(
            f'Unable to load trajectory data: {path}'
        ) from error
    if not isinstance(data, Mapping):
        raise DatasetLayoutError(f'Trajectory data is not a mapping: {path}')
    return data


def validate_trajectory(directory: str | Path) -> TrajectorySummary:
    """Validate one converted trajectory and return metric statistics."""
    path = Path(directory).expanduser().resolve()
    data_path = path / 'traj_data.pkl'
    if not data_path.is_file():
        raise DatasetLayoutError(f'Missing trajectory data: {data_path}')
    data = _load_trajectory_data(data_path)
    try:
        positions = np.asarray(data['position'], dtype=np.float64)
        yaws = np.asarray(data['yaw'], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetLayoutError(
            f'Invalid position or yaw arrays in {data_path}'
        ) from error
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise DatasetLayoutError('Trajectory position must have shape [T, 2].')
    if yaws.ndim != 1 or len(yaws) != len(positions):
        raise DatasetLayoutError('Trajectory yaw must have shape [T].')
    if len(positions) < 2:
        raise DatasetLayoutError('Trajectory must contain at least two samples.')
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(yaws)):
        raise DatasetLayoutError('Trajectory positions and yaws must be finite.')

    expected_images = {f'{index}.jpg' for index in range(len(positions))}
    actual_images = {image.name for image in path.glob('*.jpg')}
    if actual_images != expected_images:
        missing = sorted(expected_images - actual_images)
        extra = sorted(actual_images - expected_images)
        detail = []
        if missing:
            detail.append('missing: ' + ', '.join(missing[:5]))
        if extra:
            detail.append('unexpected: ' + ', '.join(extra[:5]))
        raise DatasetLayoutError(
            'Trajectory images must be contiguous from 0.jpg; ' +
            '; '.join(detail)
        )

    route_path = path / 'route.yaml'
    has_route = route_path.is_file()
    if has_route:
        route = load_route_manifest(route_path)
        if len(route.nodes) != len(positions):
            raise DatasetLayoutError(
                'Registered route and trajectory sample counts differ.'
            )

    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return TrajectorySummary(
        name=path.name,
        samples=len(positions),
        path_length_m=float(np.sum(deltas)),
        mean_spacing_m=float(np.mean(deltas)),
        median_spacing_m=float(np.median(deltas)),
        has_registered_route=has_route,
    )


def discover_trajectories(dataset_root: str | Path) -> tuple[Path, ...]:
    """Return sorted direct child directories containing trajectory data."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise DatasetLayoutError(f'Dataset root does not exist: {root}')
    trajectories = tuple(sorted(
        (path for path in root.iterdir()
         if path.is_dir() and (path / 'traj_data.pkl').is_file()),
        key=lambda path: path.name,
    ))
    if not trajectories:
        raise DatasetLayoutError(f'No trajectories found under {root}')
    return trajectories


def create_splits(
    trajectory_names: Sequence[str],
    *,
    test_fraction: float,
    seed: int,
) -> DatasetSplits:
    """Assign whole trajectories to reproducible train and test splits."""
    if not math.isfinite(test_fraction) or not 0.0 < test_fraction < 1.0:
        raise DatasetLayoutError('Test fraction must be between zero and one.')
    unique_names = sorted(set(trajectory_names))
    if len(unique_names) != len(trajectory_names):
        raise DatasetLayoutError('Trajectory names must be unique.')
    if len(unique_names) < 2:
        raise DatasetLayoutError(
            'At least two trajectories are required for a train/test split.'
        )
    generator = random.Random(seed)
    generator.shuffle(unique_names)
    test_count = max(1, min(
        len(unique_names) - 1,
        round(len(unique_names) * test_fraction),
    ))
    test = tuple(sorted(unique_names[:test_count]))
    train = tuple(sorted(unique_names[test_count:]))
    return DatasetSplits(train=train, test=test)


def write_splits(
    output_root: str | Path,
    splits: DatasetSplits,
    *,
    overwrite: bool = False,
) -> Path:
    """Write ViNT-compatible train/test trajectory-name files."""
    root = Path(output_root).expanduser().resolve()
    paths = {
        'train': root / 'train' / 'traj_names.txt',
        'test': root / 'test' / 'traj_names.txt',
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise DatasetLayoutError(
            'Split file already exists; pass overwrite explicitly: ' +
            ', '.join(str(path) for path in existing)
        )
    for split_name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        names = getattr(splits, split_name)
        path.write_text('\n'.join(names) + '\n', encoding='utf-8')
    return root
