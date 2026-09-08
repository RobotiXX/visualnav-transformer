"""Tests for mock inference and checkpoint compatibility contracts."""

import sys

from PIL import Image
import pytest

import vint_ros2.model_backend as model_backend
from vint_ros2.model_backend import checkpoint_model_type
from vint_ros2.model_backend import load_trusted_checkpoint
from vint_ros2.model_backend import MockInferenceBackend
from vint_ros2.model_backend import ModelBackendError


def _images(count: int):
    return tuple(Image.new('RGB', (2, 2)) for _ in range(count))


def test_mock_pointgoal_prediction_is_bounded_and_goal_directed():
    backend = MockInferenceBackend(
        context_size=2,
        horizon_points=5,
        horizon_distance_m=1.0,
    )

    prediction = backend.predict_point_goal(_images(3), (3.0, 4.0))

    assert len(prediction.trajectory) == 5
    assert prediction.distance == pytest.approx(5.0)
    assert prediction.trajectory[-1].x == pytest.approx(0.6)
    assert prediction.trajectory[-1].y == pytest.approx(0.8)


def test_mock_backend_rejects_incomplete_history():
    backend = MockInferenceBackend(context_size=2)

    with pytest.raises(ModelBackendError, match='Expected 3'):
        backend.predict_image_goal(_images(2), Image.new('RGB', (2, 2)))


def test_pointgoal_checkpoint_requires_explicit_model_type_metadata():
    assert checkpoint_model_type({'model_type': 'pointgoal_vint'}) == (
        'pointgoal_vint'
    )
    assert checkpoint_model_type({'model': object()}) == (
        'vint_legacy_image_goal'
    )
    assert checkpoint_model_type({'state_dict': {}}) is None


def test_trusted_loader_shims_training_only_scheduler(monkeypatch, tmp_path):
    real_import = model_backend.importlib.import_module

    def import_without_warmup(name):
        if name == 'warmup_scheduler.scheduler':
            error = ModuleNotFoundError(name)
            error.name = 'warmup_scheduler'
            raise error
        return real_import(name)

    class FakeTorch:
        @staticmethod
        def load(path, *, map_location, weights_only):
            assert path == tmp_path / 'checkpoint.pth'
            assert map_location == 'cpu'
            assert weights_only is False
            scheduler = sys.modules['warmup_scheduler.scheduler']
            return scheduler.GradualWarmupScheduler()

    monkeypatch.setattr(
        model_backend.importlib, 'import_module', import_without_warmup
    )
    result = load_trusted_checkpoint(
        FakeTorch, tmp_path / 'checkpoint.pth', 'cpu'
    )

    assert isinstance(result, model_backend._LegacyWarmupScheduler)
    assert 'warmup_scheduler' not in sys.modules
    assert 'warmup_scheduler.scheduler' not in sys.modules
