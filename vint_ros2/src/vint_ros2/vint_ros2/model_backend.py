"""Lazy, testable inference backends for image-goal and PointGoal ViNT."""

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import time
import types
from typing import Any, Mapping, Protocol, Sequence

from PIL import Image
import yaml


class ModelBackendError(RuntimeError):
    """Raised when a model or prediction violates the deployment contract."""


class _LegacyWarmupScheduler:
    """Unpickle-only stand-in for optimizer metadata in release checkpoints."""


@contextmanager
def _legacy_warmup_scheduler_import():
    """Provide the training-only scheduler name while loading trusted weights."""
    try:
        importlib.import_module('warmup_scheduler.scheduler')
    except ModuleNotFoundError as error:
        if error.name not in {
            'warmup_scheduler', 'warmup_scheduler.scheduler'
        }:
            raise
    else:
        yield
        return

    package = types.ModuleType('warmup_scheduler')
    scheduler = types.ModuleType('warmup_scheduler.scheduler')
    scheduler.GradualWarmupScheduler = _LegacyWarmupScheduler
    package.scheduler = scheduler
    sys.modules['warmup_scheduler'] = package
    sys.modules['warmup_scheduler.scheduler'] = scheduler
    try:
        yield
    finally:
        sys.modules.pop('warmup_scheduler.scheduler', None)
        sys.modules.pop('warmup_scheduler', None)


def load_trusted_checkpoint(torch, path: Path, device):
    """Load a trusted checkpoint without requiring training scheduler code."""
    with _legacy_warmup_scheduler_import():
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )


@dataclass(frozen=True)
class TrajectoryPoint:
    """One robot-relative predicted point with an optional heading vector."""

    x: float
    y: float
    heading_x: float | None = None
    heading_y: float | None = None


@dataclass(frozen=True)
class ModelPrediction:
    """Temporal distance and robot-relative trajectory predicted by ViNT."""

    distance: float
    trajectory: tuple[TrajectoryPoint, ...]


class InferenceBackend(Protocol):
    """Protocol implemented by deterministic and PyTorch model backends."""

    context_size: int

    def predict_image_goal(
        self,
        observations: Sequence[Image.Image],
        goal_image: Image.Image,
    ) -> ModelPrediction:
        """Predict a trajectory conditioned on an image goal."""

    def predict_image_goals(
        self,
        observations: Sequence[Image.Image],
        goal_images: Sequence[Image.Image],
    ) -> tuple[ModelPrediction, ...]:
        """Predict trajectories for a temporal neighborhood of image goals."""

    def predict_point_goal(
        self,
        observations: Sequence[Image.Image],
        point_goal: tuple[float, float],
    ) -> ModelPrediction:
        """Predict a trajectory conditioned on a point goal."""


class MockInferenceBackend:
    """Deterministic straight-path backend for ROS and Nav2 contract tests."""

    def __init__(
        self,
        *,
        context_size: int = 5,
        horizon_points: int = 5,
        horizon_distance_m: float = 1.25,
    ) -> None:
        """Configure deterministic history and path dimensions."""
        if context_size < 0 or horizon_points <= 0 or horizon_distance_m <= 0:
            raise ModelBackendError('Mock backend dimensions must be positive.')
        self.context_size = context_size
        self._horizon_points = horizon_points
        self._horizon_distance = horizon_distance_m

    def _check_history(self, observations: Sequence[Image.Image]) -> None:
        expected = self.context_size + 1
        if len(observations) != expected:
            raise ModelBackendError(
                f'Expected {expected} observation images, got '
                f'{len(observations)}.'
            )

    def predict_image_goal(
        self,
        observations: Sequence[Image.Image],
        goal_image: Image.Image,
    ) -> ModelPrediction:
        """Return a fixed forward trajectory after validating image history."""
        self._check_history(observations)
        if not isinstance(goal_image, Image.Image):
            raise ModelBackendError('Image goal must be a Pillow image.')
        points = tuple(
            TrajectoryPoint(
                x=self._horizon_distance * index / self._horizon_points,
                y=0.0,
                heading_x=1.0,
                heading_y=0.0,
            )
            for index in range(1, self._horizon_points + 1)
        )
        return ModelPrediction(
            distance=float(self._horizon_points),
            trajectory=points,
        )

    def predict_image_goals(
        self,
        observations: Sequence[Image.Image],
        goal_images: Sequence[Image.Image],
    ) -> tuple[ModelPrediction, ...]:
        """Return one deterministic prediction for every candidate goal."""
        if not goal_images:
            raise ModelBackendError('At least one image goal is required.')
        return tuple(
            self.predict_image_goal(observations, goal)
            for goal in goal_images
        )

    def predict_point_goal(
        self,
        observations: Sequence[Image.Image],
        point_goal: tuple[float, float],
    ) -> ModelPrediction:
        """Return a bounded straight interpolation toward a PointGoal."""
        self._check_history(observations)
        x, y = point_goal
        distance = (x * x + y * y) ** 0.5
        scale = min(1.0, self._horizon_distance / max(distance, 1.0e-9))
        endpoint = x * scale, y * scale
        heading_x = x / distance if distance > 1.0e-9 else 1.0
        heading_y = y / distance if distance > 1.0e-9 else 0.0
        points = tuple(
            TrajectoryPoint(
                x=endpoint[0] * index / self._horizon_points,
                y=endpoint[1] * index / self._horizon_points,
                heading_x=heading_x,
                heading_y=heading_y,
            )
            for index in range(1, self._horizon_points + 1)
        )
        return ModelPrediction(distance=distance, trajectory=points)


def _load_config(config_path: str | Path) -> Mapping[str, Any]:
    path = Path(config_path).expanduser().resolve()
    try:
        with path.open('r', encoding='utf-8') as stream:
            config = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ModelBackendError(f'Unable to load model config: {path}') from error
    if not isinstance(config, Mapping):
        raise ModelBackendError('Model config must be a mapping.')
    return config


def checkpoint_model_type(checkpoint: Any) -> str | None:
    """Read explicit model type metadata without guessing PointGoal support."""
    if isinstance(checkpoint, Mapping):
        model_type = checkpoint.get('model_type')
        if isinstance(model_type, str):
            return model_type
        config = checkpoint.get('model_config')
        if isinstance(config, Mapping) and isinstance(
            config.get('model_type'), str
        ):
            return config['model_type']
        if 'model' in checkpoint:
            return 'vint_legacy_image_goal'
    return None


class TorchViNTBackend:
    """Lazy PyTorch backend that enforces checkpoint/model compatibility."""

    def __init__(
        self,
        *,
        mode: str,
        checkpoint_path: str | Path,
        config_path: str | Path,
        device: str = 'cuda',
    ) -> None:
        """Load one trusted image-goal or PointGoal ViNT checkpoint."""
        if mode not in {'image_goal', 'point_goal'}:
            raise ModelBackendError(
                "Backend mode must be 'image_goal' or 'point_goal'."
            )
        try:
            import torch
            from torchvision import transforms
        except ImportError as error:
            raise ModelBackendError(
                'PyTorch and torchvision are required for model inference.'
            ) from error
        self._torch = torch
        self._transforms = transforms
        self._config = _load_config(config_path)
        self.context_size = int(self._config['context_size'])
        self._image_size = tuple(self._config['image_size'])
        self._metric_scale = float(
            self._config.get('metric_waypoint_spacing_m', 0.25)
        )
        if self._metric_scale <= 0:
            raise ModelBackendError('metric_waypoint_spacing_m must be positive.')
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise ModelBackendError(
                f"CUDA device '{device}' requested but CUDA is unavailable."
            )
        self._device = torch.device(device)
        checkpoint_file = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_file.is_file():
            raise ModelBackendError(
                f'Model checkpoint does not exist: {checkpoint_file}'
            )
        # Legacy official ViNT checkpoints pickle a module. Only load trusted
        # official or locally generated checkpoints.
        checkpoint = load_trusted_checkpoint(
            torch, checkpoint_file, torch.device('cpu')
        )
        actual_type = checkpoint_model_type(checkpoint)
        if mode == 'point_goal' and actual_type != 'pointgoal_vint':
            raise ModelBackendError(
                'PointGoal inference requires a checkpoint explicitly marked '
                "model_type: pointgoal_vint; image-goal weights are invalid."
            )
        if mode == 'image_goal' and actual_type not in {
            'vint', 'vint_legacy_image_goal', None,
        }:
            raise ModelBackendError(
                f"Checkpoint type '{actual_type}' is not image-goal ViNT."
            )
        self._mode = mode
        self._model = self._build_model()
        self._load_weights(checkpoint)
        self._model.to(self._device)
        self._model.eval()
        self.warmup_seconds = self._warm_up()

    def _warm_up(self) -> float:
        """Initialize model kernels before a live Nav2 goal needs a prediction."""
        start = time.perf_counter()
        blank = Image.new('RGB', self._image_size)
        observations = tuple(
            blank.copy() for _ in range(self.context_size + 1)
        )
        if self._mode == 'image_goal':
            self.predict_image_goal(observations, blank)
        else:
            self.predict_point_goal(observations, (1.0, 0.0))
        if self._device.type == 'cuda':
            self._torch.cuda.synchronize(self._device)
        return time.perf_counter() - start

    def _build_model(self):
        common = {
            'context_size': self.context_size,
            'len_traj_pred': int(self._config['len_traj_pred']),
            'learn_angle': bool(self._config['learn_angle']),
            'obs_encoder': self._config['obs_encoder'],
            'obs_encoding_size': int(self._config['obs_encoding_size']),
            'mha_num_attention_heads': int(
                self._config['mha_num_attention_heads']
            ),
            'mha_num_attention_layers': int(
                self._config['mha_num_attention_layers']
            ),
            'mha_ff_dim_factor': int(self._config['mha_ff_dim_factor']),
        }
        if self._mode == 'image_goal':
            from vint_train.models.vint.vint import ViNT
            return ViNT(
                **common,
                late_fusion=bool(self._config.get('late_fusion', False)),
            )
        from vint_train.models.vint.pointgoal_vint import PointGoalViNT
        return PointGoalViNT(
            **common,
            pointgoal_latent_size=int(
                self._config.get('pointgoal_latent_size', 32)
            ),
            pointgoal_hidden_size=int(
                self._config.get('pointgoal_hidden_size', 256)
            ),
        )

    def _load_weights(self, checkpoint: Any) -> None:
        torch = self._torch
        state = None
        if isinstance(checkpoint, Mapping):
            state = checkpoint.get('state_dict')
            if state is None:
                state = checkpoint.get('model_state_dict')
            if state is None:
                state = checkpoint.get('model')
        elif isinstance(checkpoint, torch.nn.Module):
            state = checkpoint
        if isinstance(state, torch.nn.DataParallel):
            state = state.module
        if isinstance(state, torch.nn.Module):
            state = state.state_dict()
        if not isinstance(state, Mapping):
            raise ModelBackendError('Checkpoint does not contain model weights.')
        cleaned = {
            (key[7:] if key.startswith('module.') else key): value
            for key, value in state.items()
        }
        try:
            self._model.load_state_dict(cleaned, strict=True)
        except RuntimeError as error:
            raise ModelBackendError(
                'Checkpoint parameters do not match the configured model.'
            ) from error

    def _transform_images(self, images: Sequence[Image.Image]):
        torch = self._torch
        transform = self._transforms.Compose([
            self._transforms.ToTensor(),
            self._transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        tensors = [
            transform(image.convert('RGB').resize(self._image_size)).unsqueeze(0)
            for image in images
        ]
        return torch.cat(tensors, dim=1).to(self._device)

    def _predictions(self, model_output) -> tuple[ModelPrediction, ...]:
        distance, action = model_output
        distances = distance.detach().cpu().reshape(-1)
        actions = action.detach().cpu().numpy()
        if len(distances) != len(actions):
            raise ModelBackendError(
                'ViNT distance and trajectory batch sizes do not match.'
            )
        predictions = []
        for distance_value, values in zip(distances, actions):
            points = []
            for value in values:
                heading_x = float(value[2]) if len(value) >= 4 else None
                heading_y = float(value[3]) if len(value) >= 4 else None
                points.append(TrajectoryPoint(
                    x=float(value[0]) * self._metric_scale,
                    y=float(value[1]) * self._metric_scale,
                    heading_x=heading_x,
                    heading_y=heading_y,
                ))
            predictions.append(ModelPrediction(
                distance=float(distance_value),
                trajectory=tuple(points),
            ))
        return tuple(predictions)

    def _observation_tensor(self, observations: Sequence[Image.Image]):
        if len(observations) != self.context_size + 1:
            raise ModelBackendError(
                f'Expected {self.context_size + 1} observation images.'
            )
        return self._transform_images(observations)

    def predict_image_goal(
        self,
        observations: Sequence[Image.Image],
        goal_image: Image.Image,
    ) -> ModelPrediction:
        """Run trusted image-goal ViNT weights."""
        if self._mode != 'image_goal':
            raise ModelBackendError('Backend is not configured for image goals.')
        return self.predict_image_goals(observations, [goal_image])[0]

    def predict_image_goals(
        self,
        observations: Sequence[Image.Image],
        goal_images: Sequence[Image.Image],
    ) -> tuple[ModelPrediction, ...]:
        """Run the original batched local-topomap image-goal inference."""
        if self._mode != 'image_goal':
            raise ModelBackendError('Backend is not configured for image goals.')
        if not goal_images:
            raise ModelBackendError('At least one image goal is required.')
        observation = self._observation_tensor(observations)
        goal_batch = self._torch.cat([
            self._transform_images([goal]) for goal in goal_images
        ], dim=0)
        with self._torch.inference_mode():
            output = self._model(
                observation.repeat(len(goal_images), 1, 1, 1),
                goal_batch,
            )
        return self._predictions(output)

    def predict_point_goal(
        self,
        observations: Sequence[Image.Image],
        point_goal: tuple[float, float],
    ) -> ModelPrediction:
        """Run PointGoal ViNT with metric goal coordinates normalized once."""
        if self._mode != 'point_goal':
            raise ModelBackendError('Backend is not configured for PointGoals.')
        goal = self._torch.tensor(
            [[point_goal[0] / self._metric_scale,
              point_goal[1] / self._metric_scale]],
            dtype=self._torch.float32,
            device=self._device,
        )
        with self._torch.inference_mode():
            output = self._model(
                self._observation_tensor(observations),
                goal,
            )
        return self._predictions(output)[0]
