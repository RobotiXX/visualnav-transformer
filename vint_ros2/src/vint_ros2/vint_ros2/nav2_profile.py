"""Compose controlled ViNT study modes into a rendered Nav2 parameter file."""

from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml


class Nav2ProfileError(ValueError):
    """Raised when a base Nav2 file cannot support the requested ViNT mode."""


class NavigationMode(str, Enum):
    """Navigation-system combinations supported by the initial study."""

    CLASSICAL = 'classical'
    IMAGE_GOAL_LOCAL = 'image_goal_local'
    POINTGOAL_GLOBAL = 'pointgoal_global'
    VINT_FULL = 'vint_full'


class _RosParameterDumper(yaml.SafeDumper):
    """Emit Nav2 parameter YAML without unsupported aliases or anchors."""

    def ignore_aliases(self, data):
        """Represent repeated Python objects as independent YAML values."""
        return True


def _ros_parameters(config: dict, node_name: str) -> dict:
    try:
        parameters = config[node_name]['ros__parameters']
    except (KeyError, TypeError) as error:
        raise Nav2ProfileError(
            f"Base Nav2 YAML has no '{node_name}.ros__parameters'."
        ) from error
    if not isinstance(parameters, dict):
        raise Nav2ProfileError(
            f"'{node_name}.ros__parameters' must be a mapping."
        )
    return parameters


def _default_plugin_config(filename: str) -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        return (
            Path(get_package_share_directory('vint_nav2_plugins')) /
            'config' / filename
        )
    except (ImportError, LookupError):
        source_root = Path(__file__).resolve().parents[2]
        return source_root / 'vint_nav2_plugins' / 'config' / filename


def load_plugin_parameters(
    path: str | Path,
    *,
    node_name: str,
    plugin_name: str,
) -> Mapping[str, Any]:
    """Load one named plugin mapping from a ROS parameter YAML snippet."""
    config = load_nav2_yaml(path)
    parameters = _ros_parameters(dict(config), node_name)
    plugin = parameters.get(plugin_name)
    if not isinstance(plugin, Mapping):
        raise Nav2ProfileError(
            f"'{node_name}.{plugin_name}' must be a plugin mapping."
        )
    if not isinstance(plugin.get('plugin'), str):
        raise Nav2ProfileError(
            f"'{node_name}.{plugin_name}.plugin' must be a string."
        )
    return plugin


def load_image_goal_controller(path: str | Path | None = None):
    """Load the configurable image-goal controller parameters."""
    return load_plugin_parameters(
        path or _default_plugin_config('image_goal_controller.yaml'),
        node_name='controller_server',
        plugin_name='ViNTImageGoal',
    )


def load_pointgoal_global_planner(path: str | Path | None = None):
    """Load the configurable PointGoal planner parameters."""
    return load_plugin_parameters(
        path or _default_plugin_config('pointgoal_planner.yaml'),
        node_name='planner_server',
        plugin_name='ViNTPointGoal',
    )


def compose_nav2_profile(
    base_config: Mapping[str, Any],
    mode: NavigationMode | str,
    *,
    image_goal_controller: Mapping[str, Any] | None = None,
    pointgoal_global_planner: Mapping[str, Any] | None = None,
) -> dict:
    """Return a deep-copied Nav2 config for one controlled study mode."""
    try:
        selected_mode = NavigationMode(mode)
    except ValueError as error:
        raise Nav2ProfileError(f"Unsupported navigation mode '{mode}'.") from error
    if not isinstance(base_config, Mapping):
        raise Nav2ProfileError('Base Nav2 YAML must be a mapping.')
    output = deepcopy(dict(base_config))
    if selected_mode == NavigationMode.CLASSICAL:
        return output

    if selected_mode in {
        NavigationMode.IMAGE_GOAL_LOCAL,
        NavigationMode.VINT_FULL,
    }:
        controller = _ros_parameters(output, 'controller_server')
        plugins = controller.get('controller_plugins')
        if not isinstance(plugins, list) or 'FollowPath' not in plugins:
            raise Nav2ProfileError(
                "Base controller_plugins must contain 'FollowPath'."
            )
        plugin = image_goal_controller or load_image_goal_controller()
        controller['FollowPath'] = deepcopy(dict(plugin))

    if selected_mode in {
        NavigationMode.POINTGOAL_GLOBAL,
        NavigationMode.VINT_FULL,
    }:
        planner = _ros_parameters(output, 'planner_server')
        plugins = planner.get('planner_plugins')
        if not isinstance(plugins, list) or 'GridBased' not in plugins:
            raise Nav2ProfileError(
                "Base planner_plugins must contain 'GridBased'."
            )
        plugin = pointgoal_global_planner or load_pointgoal_global_planner()
        planner['GridBased'] = deepcopy(dict(plugin))
    return output


def load_nav2_yaml(path: str | Path) -> Mapping[str, Any]:
    """Load a rendered Nav2 YAML file."""
    source = Path(path).expanduser().resolve()
    try:
        with source.open('r', encoding='utf-8') as stream:
            config = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise Nav2ProfileError(f'Unable to load Nav2 YAML: {source}') from error
    if not isinstance(config, Mapping):
        raise Nav2ProfileError('Base Nav2 YAML must be a mapping.')
    return config


def write_nav2_yaml(path: str | Path, config: Mapping[str, Any]) -> Path:
    """Write a composed Nav2 YAML without overwriting an existing file."""
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise Nav2ProfileError(
            f'Output Nav2 profile already exists: {destination}'
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.dump(
            dict(config),
            Dumper=_RosParameterDumper,
            sort_keys=False,
        ),
        encoding='utf-8',
    )
    return destination
