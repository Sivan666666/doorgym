import math

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.Mish(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.Mish(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.Mish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.Mish(),
        )

    def forward(self, x):
        return self.net(x.float() / 255.0)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.Mish(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.cond = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, channels))
        self.act = nn.Mish()

    def forward(self, x, cond):
        y = self.blocks(x)
        y = y + self.cond(cond).unsqueeze(-1)
        return self.act(x + y)


class DoorDiffusionPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim=10,
        obs_horizon=16,
        pred_horizon=32,
        image_feature_dim=128,
        hidden_dim=256,
        diffusion_step_embed_dim=128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.mask_encoder = ImageEncoder(image_feature_dim)
        self.depth_encoder = ImageEncoder(image_feature_dim)
        self.front_mask_encoder = ImageEncoder(image_feature_dim)
        self.front_depth_encoder = ImageEncoder(image_feature_dim)
        per_obs_dim = state_dim + image_feature_dim * 4
        self.obs_encoder = nn.Sequential(
            nn.Linear(per_obs_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = hidden_dim * obs_horizon + diffusion_step_embed_dim
        self.input_proj = nn.Conv1d(action_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList([ConditionalResidualBlock1D(hidden_dim, cond_dim) for _ in range(6)])
        self.output_proj = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, 1), nn.Mish(), nn.Conv1d(hidden_dim, action_dim, 1))

    def encode_obs(self, state, mask, masked_depth, front_mask=None, front_masked_depth=None):
        b, t = state.shape[:2]
        mask_feat = self.mask_encoder(mask.reshape(b * t, *mask.shape[2:])).reshape(b, t, -1)
        depth_feat = self.depth_encoder(masked_depth.reshape(b * t, *masked_depth.shape[2:])).reshape(b, t, -1)
        if front_mask is None:
            front_mask = torch.zeros_like(mask)
        if front_masked_depth is None:
            front_masked_depth = torch.zeros_like(masked_depth)
        front_mask_feat = self.front_mask_encoder(front_mask.reshape(b * t, *front_mask.shape[2:])).reshape(b, t, -1)
        front_depth_feat = self.front_depth_encoder(front_masked_depth.reshape(b * t, *front_masked_depth.shape[2:])).reshape(b, t, -1)
        obs_feat = torch.cat([state, mask_feat, depth_feat, front_mask_feat, front_depth_feat], dim=-1)
        return self.obs_encoder(obs_feat).reshape(b, -1)

    def forward(self, noisy_action, timesteps, state, mask, masked_depth, front_mask=None, front_masked_depth=None):
        obs_cond = self.encode_obs(state, mask, masked_depth, front_mask, front_masked_depth)
        time_cond = self.timestep_encoder(timesteps)
        cond = torch.cat([obs_cond, time_cond], dim=-1)
        x = self.input_proj(noisy_action.transpose(1, 2))
        for block in self.blocks:
            x = block(x, cond)
        return self.output_proj(x).transpose(1, 2)
