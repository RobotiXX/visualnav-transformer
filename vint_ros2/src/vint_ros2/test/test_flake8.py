"""Flake8 lint test."""

from ament_flake8.main import main_with_errors
import pytest


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    """Check code against flake8."""
    rc, errors = main_with_errors(argv=['--exclude', 'build,install,log'])
    assert rc == 0, (
        'Found %d code style errors / warnings:\n' % len(errors) +
        '\n'.join(errors)
    )
