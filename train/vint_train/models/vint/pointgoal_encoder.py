"""PointGoal token encoding shared by training and deployment."""

import torch
from torch import nn


class PointGoalTokenEncoder(nn.Module):
    """Encode planar robot-relative goal coordinates as one ViNT token."""

    def __init__(
        self,
        embedding_size: int,
        latent_size: int = 32,
        hidden_size: int = 256,
    ) -> None:
        """Create the learned latent and two-layer coordinate MLP."""
        super().__init__()
        if embedding_size <= 0 or latent_size <= 0 or hidden_size <= 0:
            raise ValueError('PointGoal encoder dimensions must be positive.')
        self.goal_latent = nn.Parameter(torch.zeros(latent_size))
        nn.init.normal_(self.goal_latent, mean=0.0, std=0.02)
        self.encoder = nn.Sequential(
            nn.Linear(latent_size + 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, point_goal: torch.Tensor) -> torch.Tensor:
        """Encode a batch shaped ``[batch, 2]`` into ``[batch, 1, dim]``."""
        if point_goal.ndim != 2 or point_goal.shape[-1] != 2:
            raise ValueError('PointGoal input must have shape [batch, 2].')
        latent = self.goal_latent.unsqueeze(0).expand(point_goal.shape[0], -1)
        goal_input = torch.cat((point_goal, latent), dim=-1)
        return self.encoder(goal_input).unsqueeze(1)
