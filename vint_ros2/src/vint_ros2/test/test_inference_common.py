"""Tests for inference history, freshness, and ROS path conversion."""

import math

from builtin_interfaces.msg import Time
from PIL import Image
import pytest

from vint_ros2.inference_common import history_is_fresh
from vint_ros2.inference_common import ImageHistory
from vint_ros2.inference_common import prediction_to_path
from vint_ros2.model_backend import ModelPrediction
from vint_ros2.model_backend import TrajectoryPoint


def test_image_history_rejects_out_of_order_observations():
    history = ImageHistory(context_size=1)
    image = Image.new('RGB', (2, 2))

    assert history.append(10, image)
    assert not history.append(9, image)
    assert history.append(11, image)
    observations, newest = history.snapshot()
    assert len(observations) == 2
    assert newest == 11


def test_freshness_requires_complete_nonfuture_context():
    images = (Image.new('RGB', (2, 2)),) * 2

    assert history_is_fresh(
        images, 900, required_count=2, now_ns=1000,
        maximum_age_sec=1.0,
    )
    assert not history_is_fresh(
        images, 1100, required_count=2, now_ns=1000,
        maximum_age_sec=1.0,
    )
    assert not history_is_fresh(
        images[:1], 900, required_count=2, now_ns=1000,
        maximum_age_sec=1.0,
    )


def test_prediction_path_includes_origin_and_heading():
    prediction = ModelPrediction(
        distance=2.0,
        trajectory=(
            TrajectoryPoint(0.5, 0.0),
            TrajectoryPoint(1.0, 1.0, heading_x=0.0, heading_y=1.0),
        ),
    )

    path = prediction_to_path(
        prediction,
        frame_id='base_link',
        stamp=Time(sec=3),
    )

    assert path.header.frame_id == 'base_link'
    assert len(path.poses) == 3
    assert path.poses[0].pose.position.x == 0.0
    assert path.poses[1].pose.orientation.w == pytest.approx(1.0)
    assert path.poses[2].pose.orientation.z == pytest.approx(
        math.sin(math.pi / 4.0)
    )
