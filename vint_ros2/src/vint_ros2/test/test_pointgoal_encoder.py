"""Tests for the train-time PointGoal token encoder."""

from pathlib import Path
import sys

import pytest
import torch


_TRAIN_ROOT = Path(__file__).resolve().parents[4] / 'train'
sys.path.insert(0, str(_TRAIN_ROOT))
from vint_train.models.vint.pointgoal_encoder import PointGoalTokenEncoder  # noqa: E402,E501


def test_pointgoal_encoder_produces_one_token_and_gradients():
    encoder = PointGoalTokenEncoder(
        embedding_size=16,
        latent_size=4,
        hidden_size=8,
    )
    goals = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    tokens = encoder(goals)
    tokens.sum().backward()

    assert tokens.shape == (2, 1, 16)
    assert encoder.goal_latent.grad is not None


def test_pointgoal_encoder_rejects_wrong_goal_shape():
    encoder = PointGoalTokenEncoder(embedding_size=8)

    with pytest.raises(ValueError, match='shape'):
        encoder(torch.zeros(2, 3))
