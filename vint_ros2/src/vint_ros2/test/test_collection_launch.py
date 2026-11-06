"""Tests for the route-collection launch contract."""

from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
from pathlib import Path


LAUNCH_PATH = Path(__file__).parents[1] / 'launch' / 'collect_route.launch.py'


def _load_launch_module():
    spec = spec_from_file_location('collect_route_launch', LAUNCH_PATH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_bag_output_uses_ssd_directory(monkeypatch, tmp_path):
    monkeypatch.setenv('HOME', str(tmp_path))
    module = _load_launch_module()
    description = module.generate_launch_description()
    output_argument = next(
        entity for entity in description.entities
        if getattr(entity, 'name', None) == 'output'
    )

    default_parts = output_argument.default_value
    assert len(default_parts) == 1
    assert default_parts[0].text == '~/ssd/vint_bags/vint_route_bag'
    assert module._resolve_output_directory(default_parts[0].text) == str(
        tmp_path / 'ssd' / 'vint_bags' / 'vint_route_bag'
    )
