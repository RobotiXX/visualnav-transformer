"""Checkpoint scheduling and validation-loss artifacts for model training."""

from dataclasses import dataclass
import csv
import math
from pathlib import Path
import shutil
import tempfile
from typing import Sequence


@dataclass(frozen=True)
class ValidationPoint:
    """Validation loss recorded after one zero-indexed training epoch."""

    epoch: int
    validation_loss: float

    @property
    def completed_epochs(self) -> int:
        """Return the human-facing number of completed epochs."""
        return self.epoch + 1


def prepare_run_directory(
    log_root: str | Path,
    project_name: str,
    run_name: str,
    config_path: str | Path,
) -> Path:
    """Create a new run directory and preserve its exact input config."""
    folder = Path(log_root).expanduser() / project_name / run_name
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, folder / "training_config.yaml")
    return folder


def checkpoint_filename(epoch: int) -> str:
    """Return a human-facing checkpoint name for a zero-indexed epoch."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    return f"epoch_{epoch + 1:03d}.pth"


def should_save_numbered_checkpoint(epoch: int, frequency: int) -> bool:
    """Return whether this completed epoch is on the configured interval."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if frequency <= 0:
        raise ValueError("checkpoint frequency must be positive")
    return (epoch + 1) % frequency == 0


def is_best_validation_loss(loss: float, previous_best: float | None) -> bool:
    """Return whether a finite validation loss improves on the previous best."""
    return math.isfinite(loss) and (
        previous_best is None or loss < previous_best
    )


def write_validation_artifacts(
    project_folder: str | Path,
    history: Sequence[ValidationPoint],
    *,
    csv_filename: str = "validation_loss.csv",
    plot_filename: str = "validation_loss.png",
) -> tuple[Path, Path]:
    """Atomically write validation history as CSV and a PNG line plot."""
    if not history:
        raise ValueError("validation history must not be empty")
    folder = Path(project_folder)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / csv_filename
    plot_path = folder / plot_filename

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{csv_path.name}.",
        dir=folder,
        delete=False,
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "validation_loss"))
        writer.writerows(
            (point.completed_epochs, f"{point.validation_loss:.10g}")
            for point in history
        )
        temporary_csv = Path(stream.name)
    temporary_csv.replace(csv_path)

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(figsize=(8.0, 5.0))
    axes.plot(
        [point.completed_epochs for point in history],
        [point.validation_loss for point in history],
        color="tab:blue",
        marker="o",
        linewidth=2.0,
    )
    axes.set_xlabel("Completed epoch")
    axes.set_ylabel("Mean validation loss")
    axes.set_title("ViNT validation loss")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    with tempfile.NamedTemporaryFile(
        prefix=f".{plot_path.stem}.",
        suffix=plot_path.suffix,
        dir=folder,
        delete=False,
    ) as stream:
        temporary_plot = Path(stream.name)
    figure.savefig(temporary_plot, dpi=160)
    plt.close(figure)
    temporary_plot.replace(plot_path)
    return csv_path, plot_path
