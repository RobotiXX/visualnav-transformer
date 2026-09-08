"""Tests for ViNT dataset validation and deterministic splits."""

from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import pytest

from vint_ros2.dataset_layout import create_splits
from vint_ros2.dataset_layout import DatasetLayoutError
from vint_ros2.dataset_layout import validate_trajectory
from vint_ros2.dataset_layout import write_splits


def _trajectory(root: Path, name: str, samples: int = 3) -> Path:
    path = root / name
    path.mkdir(parents=True)
    with (path / 'traj_data.pkl').open('wb') as stream:
        pickle.dump({
            'position': np.array([
                [float(index), 0.0] for index in range(samples)
            ]),
            'yaw': np.zeros(samples),
        }, stream)
    for index in range(samples):
        Image.new('RGB', (2, 2)).save(path / f'{index}.jpg')
    return path


def test_validates_training_layout_and_reports_spacing(tmp_path):
    summary = validate_trajectory(_trajectory(tmp_path, 'route_a'))

    assert summary.name == 'route_a'
    assert summary.samples == 3
    assert summary.path_length_m == pytest.approx(2.0)
    assert summary.median_spacing_m == pytest.approx(1.0)
    assert not summary.has_registered_route


def test_rejects_noncontiguous_image_sequence(tmp_path):
    path = _trajectory(tmp_path, 'route_a')
    (path / '1.jpg').unlink()

    with pytest.raises(DatasetLayoutError, match='contiguous'):
        validate_trajectory(path)


def test_split_is_deterministic_and_keeps_whole_trajectories():
    names = ['route_4', 'route_1', 'route_3', 'route_2']

    first = create_splits(names, test_fraction=0.25, seed=17)
    second = create_splits(list(reversed(names)), test_fraction=0.25, seed=17)

    assert first == second
    assert len(first.train) == 3
    assert len(first.test) == 1
    assert set(first.train).isdisjoint(first.test)


def test_write_splits_refuses_accidental_overwrite(tmp_path):
    splits = create_splits(['a', 'b'], test_fraction=0.5, seed=0)
    output = write_splits(tmp_path / 'splits', splits)

    assert (output / 'train' / 'traj_names.txt').is_file()
    assert (output / 'test' / 'traj_names.txt').is_file()
    with pytest.raises(DatasetLayoutError, match='already exists'):
        write_splits(output, splits)
