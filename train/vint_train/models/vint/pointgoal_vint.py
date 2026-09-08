"""PointGoal-conditioned adaptation of the Visual Navigation Transformer."""

from typing import Optional, Tuple

import torch

from vint_train.models.vint.pointgoal_encoder import PointGoalTokenEncoder
from vint_train.models.vint.vint import ViNT


class PointGoalViNT(ViNT):
    """Replace ViNT's goal-image token with robot-relative goal coordinates."""

    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = 'efficientnet-b0',
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 4,
        mha_num_attention_layers: Optional[int] = 4,
        mha_ff_dim_factor: Optional[int] = 4,
        pointgoal_latent_size: int = 32,
        pointgoal_hidden_size: int = 256,
    ) -> None:
        """Initialize ViNT observation encoding and the PointGoal token MLP."""
        super().__init__(
            context_size=context_size,
            len_traj_pred=len_traj_pred,
            learn_angle=learn_angle,
            obs_encoder=obs_encoder,
            obs_encoding_size=obs_encoding_size,
            late_fusion=True,
            mha_num_attention_heads=mha_num_attention_heads,
            mha_num_attention_layers=mha_num_attention_layers,
            mha_ff_dim_factor=mha_ff_dim_factor,
        )
        del self.goal_encoder
        del self.compress_goal_enc
        self.pointgoal_encoder = PointGoalTokenEncoder(
            embedding_size=obs_encoding_size,
            latent_size=pointgoal_latent_size,
            hidden_size=pointgoal_hidden_size,
        )

    def forward(
        self,
        obs_img: torch.Tensor,
        point_goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict temporal distance and a short robot-relative trajectory."""
        goal_encoding = self.pointgoal_encoder(point_goal)

        observations = torch.split(obs_img, 3, dim=1)
        observations = torch.concat(observations, dim=0)
        obs_encoding = self.obs_encoder.extract_features(observations)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.reshape(
            (self.context_size + 1, -1, self.obs_encoding_size)
        )
        obs_encoding = torch.transpose(obs_encoding, 0, 1)

        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)
        final_representation = self.decoder(tokens)
        distance = self.dist_predictor(final_representation)
        action = self.action_predictor(final_representation)
        action = action.reshape(
            (action.shape[0], self.len_trajectory_pred,
             self.num_action_params)
        )
        action[:, :, :2] = torch.cumsum(action[:, :, :2], dim=1)
        if self.learn_angle:
            action[:, :, 2:] = torch.nn.functional.normalize(
                action[:, :, 2:].clone(), dim=-1
            )
        return distance, action
