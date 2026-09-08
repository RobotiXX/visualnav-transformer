"""Build reproducible navigation-profile metadata for RWD task logs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from vint_ros2.nav2_profile import NavigationMode
from vint_ros2.route_manifest import load_route_manifest


class ExperimentContextError(ValueError):
    """Raised when an experiment profile cannot be described safely."""


_MODE_COMPONENTS = {
    NavigationMode.CLASSICAL: ('theta_star', 'dwb'),
    NavigationMode.IMAGE_GOAL_LOCAL: ('theta_star', 'vint_image_goal'),
    NavigationMode.POINTGOAL_GLOBAL: ('vint_pointgoal_theta_star_tail', 'dwb'),
    NavigationMode.VINT_FULL: (
        'vint_pointgoal_theta_star_tail',
        'vint_image_goal',
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, path)
    return digest.hexdigest()


def _update_digest(digest, path: Path) -> None:
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)


def describe_file(path: str | Path, *, label: str) -> dict[str, Any]:
    """Describe one required file by absolute path, size, and SHA-256."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ExperimentContextError(f'{label} is not a file: {resolved}')
    return {
        'path': str(resolved),
        'size_bytes': resolved.stat().st_size,
        'sha256': _sha256(resolved),
    }


def describe_route(path: str | Path) -> dict[str, Any]:
    """Describe a route manifest and hash every referenced goal image."""
    route = load_route_manifest(path)
    descriptor = describe_file(route.manifest_path, label='Route manifest')
    digest = hashlib.sha256()
    _update_digest(digest, route.manifest_path)
    for node in route.nodes:
        digest.update(b'\0')
        digest.update(node.image.encode('utf-8'))
        digest.update(b'\0')
        _update_digest(digest, node.image_path)
    descriptor.update({
        'content_sha256': digest.hexdigest(),
        'image_count': len(route.nodes),
        'route_id': route.route_id,
    })
    return descriptor


def _load_ros_parameters(path: str | Path, node_name: str) -> tuple[Path, dict]:
    resolved = Path(path).expanduser().resolve()
    try:
        config = yaml.safe_load(resolved.read_text(encoding='utf-8'))
        parameters = config[node_name]['ros__parameters']
    except (OSError, yaml.YAMLError, KeyError, TypeError) as error:
        raise ExperimentContextError(
            f'Unable to load {node_name}.ros__parameters from {resolved}.'
        ) from error
    if not isinstance(parameters, Mapping):
        raise ExperimentContextError(
            f'{node_name}.ros__parameters must be a mapping in {resolved}.'
        )
    return resolved, dict(parameters)


def _inference_component(
    path: str | Path,
    *,
    node_name: str,
    route_manifest: str | Path | None = None,
) -> dict[str, Any]:
    resolved, parameters = _load_ros_parameters(path, node_name)
    backend = str(parameters.get('backend', '')).strip()
    if backend not in {'mock', 'torch'}:
        raise ExperimentContextError(
            f"{node_name}.backend must be 'mock' or 'torch'."
        )

    component: dict[str, Any] = {
        'backend': backend,
        'device': str(parameters.get('device', '')),
        'parameter_file': describe_file(resolved, label='Inference parameter file'),
        'checkpoint': None,
        'model_config': None,
    }
    checkpoint = str(parameters.get('checkpoint_path', '')).strip()
    model_config = str(parameters.get('model_config_path', '')).strip()
    if backend == 'torch' and (not checkpoint or not model_config):
        raise ExperimentContextError(
            f'{node_name} torch backend requires checkpoint_path and model_config_path.'
        )
    if checkpoint:
        component['checkpoint'] = describe_file(
            checkpoint, label='Model checkpoint'
        )
    if model_config:
        component['model_config'] = describe_file(
            model_config, label='Model configuration'
        )
    if route_manifest is not None:
        component['route_manifest'] = describe_route(route_manifest)
    return component


def build_experiment_context(
    mode: NavigationMode | str,
    *,
    image_goal_params: str | Path | None = None,
    pointgoal_params: str | Path | None = None,
    route_manifest: str | Path | None = None,
    image_goal_controller_config: str | Path | None = None,
    pointgoal_planner_config: str | Path | None = None,
) -> dict[str, Any]:
    """Return a self-contained description of one navigation study profile."""
    try:
        selected_mode = NavigationMode(mode)
    except ValueError as error:
        raise ExperimentContextError(
            f"Unsupported navigation profile '{mode}'."
        ) from error

    global_planner, local_controller = _MODE_COMPONENTS[selected_mode]
    context: dict[str, Any] = {
        'schema_version': 1,
        'navigation_profile': selected_mode.value,
        'global_planner': global_planner,
        'local_controller': local_controller,
        'inference': {},
        'nav2_plugin_configurations': {},
    }
    if selected_mode in {
        NavigationMode.IMAGE_GOAL_LOCAL,
        NavigationMode.VINT_FULL,
    }:
        if image_goal_params is None or route_manifest is None:
            raise ExperimentContextError(
                'Image-goal profiles require image-goal parameters and a route manifest.'
            )
        context['inference']['image_goal'] = _inference_component(
            image_goal_params,
            node_name='vint_image_goal_inference',
            route_manifest=route_manifest,
        )
        if image_goal_controller_config is None:
            raise ExperimentContextError(
                'Image-goal profiles require the effective controller configuration.'
            )
        context['nav2_plugin_configurations']['image_goal_controller'] = (
            describe_file(
                image_goal_controller_config,
                label='Image-goal controller configuration',
            )
        )
    if selected_mode in {
        NavigationMode.POINTGOAL_GLOBAL,
        NavigationMode.VINT_FULL,
    }:
        if pointgoal_params is None:
            raise ExperimentContextError(
                'PointGoal profiles require PointGoal inference parameters.'
            )
        context['inference']['pointgoal'] = _inference_component(
            pointgoal_params,
            node_name='vint_pointgoal_inference',
        )
        if pointgoal_planner_config is None:
            raise ExperimentContextError(
                'PointGoal profiles require the effective planner configuration.'
            )
        context['nav2_plugin_configurations']['pointgoal_planner'] = (
            describe_file(
                pointgoal_planner_config,
                label='PointGoal planner configuration',
            )
        )
    return context


def write_experiment_context(
    destination: str | Path,
    context: Mapping[str, Any],
) -> Path:
    """Write one context JSON without overwriting an existing artifact."""
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise ExperimentContextError(
            f'Experiment context already exists: {output}'
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(context), indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return output
