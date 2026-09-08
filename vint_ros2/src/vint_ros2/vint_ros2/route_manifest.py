"""Load and validate metric-registered visual route manifests."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = '1.0'


class RouteManifestError(ValueError):
    """Raised when a visual route manifest violates its data contract."""


@dataclass(frozen=True)
class Pose2D:
    """A planar pose expressed in the route manifest's map frame."""

    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class RouteNode:
    """One timestamped route observation and its registered map pose."""

    index: int
    image: str
    image_path: Path
    stamp_ns: int
    pose: Pose2D


@dataclass(frozen=True)
class RouteManifest:
    """An ordered, directional visual route registered to an occupancy map."""

    schema_version: str
    route_id: str
    map_id: str
    frame_id: str
    source_waypoint_id: str
    destination_waypoint_id: str
    nodes: tuple[RouteNode, ...]
    manifest_path: Path

    @property
    def direction(self) -> tuple[str, str]:
        """Return the explicit semantic source and destination pair."""
        return self.source_waypoint_id, self.destination_waypoint_id


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RouteManifestError(f"'{field}' must be a mapping.")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RouteManifestError(f"'{field}' must be a sequence.")
    return value


def _require_string(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RouteManifestError(f"'{field}' must be a non-empty string.")
    return value.strip()


def _require_int(data: Mapping[str, Any], field: str) -> int:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RouteManifestError(f"'{field}' must be an integer.")
    return value


def _require_finite_float(data: Mapping[str, Any], field: str) -> float:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RouteManifestError(f"'{field}' must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise RouteManifestError(f"'{field}' must be finite.")
    return result


def _validate_waypoint_id(value: str, field: str) -> None:
    components = value.split('.')
    if len(components) < 3 or any(not component for component in components):
        raise RouteManifestError(
            f"'{field}' must be a full waypoint ID such as "
            "'building.floor.waypoint'."
        )


def _resolve_image_path(root: Path, image: str, index: int) -> Path:
    relative_path = Path(image)
    if relative_path.is_absolute():
        raise RouteManifestError(
            f"Node {index} image path must be relative to the manifest."
        )
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RouteManifestError(
            f"Node {index} image path escapes the route directory."
        ) from error
    return resolved


def load_route_manifest(
    manifest_path: str | Path,
    *,
    require_images: bool = True,
) -> RouteManifest:
    """
    Load a route manifest and enforce its versioned contract.

    ``manifest_path`` identifies the YAML file. When ``require_images`` is
    true, every referenced image must already exist. Invalid YAML or contract
    fields raise :class:`RouteManifestError`.
    """
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise RouteManifestError(f"Route manifest does not exist: {path}")

    try:
        with path.open('r', encoding='utf-8') as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise RouteManifestError(
            f"Unable to read route manifest '{path}': {error}"
        ) from error

    data = _require_mapping(raw, 'document')
    schema_version = _require_string(data, 'schema_version')
    if schema_version != SCHEMA_VERSION:
        raise RouteManifestError(
            f"Unsupported route schema '{schema_version}'; expected "
            f"'{SCHEMA_VERSION}'."
        )

    route_id = _require_string(data, 'route_id')
    map_id = _require_string(data, 'map_id')
    frame_id = _require_string(data, 'frame_id')
    source_id = _require_string(data, 'source_waypoint_id')
    destination_id = _require_string(data, 'destination_waypoint_id')
    _validate_waypoint_id(source_id, 'source_waypoint_id')
    _validate_waypoint_id(destination_id, 'destination_waypoint_id')
    if source_id == destination_id:
        raise RouteManifestError(
            'Source and destination waypoint IDs must be different.'
        )

    raw_nodes = _require_sequence(data.get('nodes'), 'nodes')
    if len(raw_nodes) < 2:
        raise RouteManifestError('A route must contain at least two nodes.')

    root = path.parent.resolve()
    nodes: list[RouteNode] = []
    previous_stamp = -1
    for expected_index, raw_node in enumerate(raw_nodes):
        node = _require_mapping(raw_node, f'nodes[{expected_index}]')
        index = _require_int(node, 'index')
        if index != expected_index:
            raise RouteManifestError(
                'Route node indices must be contiguous and start at zero; '
                f'expected {expected_index}, found {index}.'
            )
        image = _require_string(node, 'image')
        image_path = _resolve_image_path(root, image, index)
        if require_images and not image_path.is_file():
            raise RouteManifestError(
                f"Node {index} image does not exist: {image_path}"
            )
        stamp_ns = _require_int(node, 'stamp_ns')
        if stamp_ns < 0:
            raise RouteManifestError(
                f'Node {index} stamp_ns must be non-negative.'
            )
        if stamp_ns <= previous_stamp:
            raise RouteManifestError(
                'Route node timestamps must be strictly increasing.'
            )
        previous_stamp = stamp_ns
        pose = _require_mapping(node.get('pose'), f'nodes[{index}].pose')
        nodes.append(RouteNode(
            index=index,
            image=image,
            image_path=image_path,
            stamp_ns=stamp_ns,
            pose=Pose2D(
                x=_require_finite_float(pose, 'x'),
                y=_require_finite_float(pose, 'y'),
                yaw=_require_finite_float(pose, 'yaw'),
            ),
        ))

    return RouteManifest(
        schema_version=schema_version,
        route_id=route_id,
        map_id=map_id,
        frame_id=frame_id,
        source_waypoint_id=source_id,
        destination_waypoint_id=destination_id,
        nodes=tuple(nodes),
        manifest_path=path,
    )
