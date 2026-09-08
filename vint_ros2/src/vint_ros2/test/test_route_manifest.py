"""Tests for registered visual route manifests."""

from pathlib import Path

import pytest
import yaml

from vint_ros2.route_manifest import RouteManifestError
from vint_ros2.route_manifest import load_route_manifest


def _manifest() -> dict:
    return {
        'schema_version': '1.0',
        'route_id': 'a_to_b',
        'map_id': 'test_map',
        'frame_id': 'map',
        'source_waypoint_id': 'test.floor_1.a',
        'destination_waypoint_id': 'test.floor_1.b',
        'nodes': [
            {
                'index': 0,
                'image': 'images/000000.jpg',
                'stamp_ns': 1,
                'pose': {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
            },
            {
                'index': 1,
                'image': 'images/000001.jpg',
                'stamp_ns': 2,
                'pose': {'x': 1.0, 'y': 0.0, 'yaw': 0.0},
            },
        ],
    }


def _write_manifest(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / 'route.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    return path


def test_loads_directional_route_with_resolved_image_paths(tmp_path):
    data = _manifest()
    for node in data['nodes']:
        image = tmp_path / node['image']
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b'image')

    route = load_route_manifest(_write_manifest(tmp_path, data))

    assert route.direction == ('test.floor_1.a', 'test.floor_1.b')
    assert route.nodes[1].image_path == (
        tmp_path / 'images/000001.jpg'
    ).resolve()


def test_can_validate_metadata_without_recorded_images(tmp_path):
    route = load_route_manifest(
        _write_manifest(tmp_path, _manifest()),
        require_images=False,
    )

    assert len(route.nodes) == 2


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (lambda data: data.update(schema_version='2.0'), 'Unsupported'),
        (lambda data: data['nodes'][1].update(index=3), 'contiguous'),
        (lambda data: data['nodes'][1].update(stamp_ns=1), 'increasing'),
        (lambda data: data['nodes'][0].update(image='../escape.jpg'), 'escapes'),
        (lambda data: data.update(source_waypoint_id='local'), 'full waypoint'),
        (lambda data: data['nodes'][0]['pose'].update(x=float('nan')), 'finite'),
    ],
)
def test_rejects_invalid_contracts(tmp_path, mutation, message):
    data = _manifest()
    mutation(data)

    with pytest.raises(RouteManifestError, match=message):
        load_route_manifest(
            _write_manifest(tmp_path, data),
            require_images=False,
        )


def test_requires_images_by_default(tmp_path):
    with pytest.raises(RouteManifestError, match='does not exist'):
        load_route_manifest(_write_manifest(tmp_path, _manifest()))
