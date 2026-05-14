import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from door_rl.door_asset_rl_env import IMAGE_CHANNELS, IMAGE_H, IMAGE_W, PROPRIO_DIM


class TeacherPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, deterministic=False):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum")
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        self.deterministic = deterministic

    def compute(self, inputs, role):
        actions = self.net(inputs["states"])
        if self.deterministic:
            return actions, torch.full_like(self.log_std_parameter, -20.0), {}
        return actions, self.log_std_parameter, {}


class TeacherValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class StudentPolicy(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.image_dim = IMAGE_CHANNELS * IMAGE_H * IMAGE_W
        self.proprio_dim = PROPRIO_DIM
        self.image_encoder = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 12, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, self.num_actions),
            nn.Tanh(),
        )

    def compute(self, inputs, role):
        states = inputs["states"]
        images = states[:, : self.image_dim].reshape(-1, IMAGE_CHANNELS, IMAGE_H, IMAGE_W)
        proprio = states[:, self.image_dim : self.image_dim + self.proprio_dim]
        x = torch.cat([self.image_encoder(images), self.proprio_encoder(proprio)], dim=-1)
        return self.head(x), {}

