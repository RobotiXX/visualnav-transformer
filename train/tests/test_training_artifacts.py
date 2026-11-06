"""Tests for persisted ViNT training artifacts."""

import csv

import pytest

from vint_train.training.training_artifacts import checkpoint_filename
from vint_train.training.training_artifacts import is_best_validation_loss
from vint_train.training.training_artifacts import prepare_run_directory
from vint_train.training.training_artifacts import should_save_numbered_checkpoint
from vint_train.training.training_artifacts import ValidationPoint
from vint_train.training.training_artifacts import write_validation_artifacts


def test_checkpoint_interval_uses_completed_epoch_numbers():
    assert not should_save_numbered_checkpoint(3, 5)
    assert should_save_numbered_checkpoint(4, 5)
    assert should_save_numbered_checkpoint(9, 5)
    assert checkpoint_filename(4) == "epoch_005.pth"
    assert checkpoint_filename(49) == "epoch_050.pth"


def test_prepare_run_directory_creates_folder_before_copying_config(tmp_path):
    config = tmp_path / "input.yaml"
    config.write_text("epochs: 1\n", encoding="utf-8")

    output = prepare_run_directory(
        tmp_path / "logs", "project", "run", config
    )

    assert output == tmp_path / "logs" / "project" / "run"
    assert (output / "training_config.yaml").read_text(
        encoding="utf-8"
    ) == "epochs: 1\n"


def test_checkpoint_interval_rejects_invalid_values():
    with pytest.raises(ValueError, match="frequency"):
        should_save_numbered_checkpoint(0, 0)
    with pytest.raises(ValueError, match="epoch"):
        checkpoint_filename(-1)


def test_best_validation_loss_requires_finite_improvement():
    assert is_best_validation_loss(1.0, None)
    assert is_best_validation_loss(0.9, 1.0)
    assert not is_best_validation_loss(1.0, 1.0)
    assert not is_best_validation_loss(float("nan"), 1.0)


def test_writes_csv_and_png_validation_artifacts(tmp_path):
    history = [ValidationPoint(0, 1.5), ValidationPoint(1, 1.25)]
    csv_path, plot_path = write_validation_artifacts(tmp_path, history)

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows == [
        ["epoch", "validation_loss"],
        ["1", "1.5"],
        ["2", "1.25"],
    ]
    assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
