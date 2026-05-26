"""
Gymnasium wrapper for the MuJoCo+Style3D dual Piper deformable manipulation environment.

Used by `lerobot-eval` for policy rollout evaluation in simulation.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.robots.sim_dual_piper_deformable import (
    SimDualPiperDeformable,
    SimDualPiperDeformableConfig,
)

MOTOR_NAMES = SimDualPiperDeformable.MOTOR_NAMES


class DeformableClothEnv(gym.Env):
    """Wraps SimDualPiperDeformable as a standard gym.Env for LeRobot eval."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        task: str = "silk_flatten",
        obs_type: str = "state_pcd",
        render_mode: str = "rgb_array",
        max_episode_steps: int = 500,
        xml_path: str = "assets/scene/mujoco/dual_piper_with_silk_camera.xml",
        camera_name: str = "photoneo_cam",
        include_point_cloud: bool = True,
        num_points: int = 2048,
        img_height: int = 480,
        img_width: int = 640,
        **kwargs,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self._max_episode_steps = max_episode_steps
        self._step_count = 0
        self._num_points = num_points
        self._camera_name = camera_name

        include_rgb = "pixels" in obs_type
        self._include_pcd = "pcd" in obs_type and include_point_cloud
        self._include_rgb = include_rgb

        self._robot_config = SimDualPiperDeformableConfig(
            id="sim_eval_env",
            xml_path=xml_path,
            render=False,
            camera_name=camera_name,
            img_width=img_width,
            img_height=img_height,
            include_rgb=include_rgb,
            include_point_cloud=self._include_pcd,
            num_points=num_points,
        )

        self.robot = SimDualPiperDeformable(self._robot_config)

        # Action space: 14-D (6 arm joints + 1 gripper) × 2 arms
        # Joint limits are approximate; gripper is [0, 1]
        low = np.full(14, -np.pi, dtype=np.float32)
        high = np.full(14, np.pi, dtype=np.float32)
        low[6] = 0.0   # left gripper
        high[6] = 1.0
        low[13] = 0.0  # right gripper
        high[13] = 1.0
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Observation space
        obs_spaces = {
            "agent_pos": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32),
        }
        if self._include_pcd:
            obs_spaces["point_cloud"] = spaces.Box(
                -np.inf, np.inf, shape=(num_points, 3), dtype=np.float32
            )
        if self._include_rgb:
            obs_spaces[f"pixels/{camera_name}"] = spaces.Box(
                0, 255, shape=(img_height, img_width, 3), dtype=np.uint8
            )
        self.observation_space = spaces.Dict(obs_spaces)

    def _build_obs(self, obs_dict: dict) -> dict:
        """Convert robot observation dict to gym observation dict."""
        state = np.array([obs_dict[name] for name in MOTOR_NAMES], dtype=np.float32)
        obs = {"agent_pos": state}

        if self._include_pcd and "observation.point_cloud" in obs_dict:
            obs["point_cloud"] = obs_dict["observation.point_cloud"].astype(np.float32)

        cam_key = f"observation.images.{self._camera_name}"
        if self._include_rgb and cam_key in obs_dict:
            obs[f"pixels/{self._camera_name}"] = obs_dict[cam_key]

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.robot.is_connected:
            self.robot.connect()

        obs_dict = self.robot.reset()
        self._step_count = 0
        return self._build_obs(obs_dict), {}

    def step(self, action: np.ndarray):
        action_dict = {name: float(action[i]) for i, name in enumerate(MOTOR_NAMES)}
        self.robot.send_action(action_dict)
        obs_dict = self.robot.get_observation()

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self._max_episode_steps

        # Sparse reward (0 by default; task-specific subclasses can override)
        reward = 0.0
        info = {"step_count": self._step_count}

        return self._build_obs(obs_dict), reward, terminated, truncated, info

    def render(self):
        if not self.robot.is_connected:
            return None
        obs_dict = self.robot.get_observation()
        cam_key = f"observation.images.{self._camera_name}"
        if cam_key in obs_dict:
            return obs_dict[cam_key]
        return None

    def close(self):
        if self.robot.is_connected:
            self.robot.disconnect()

    @property
    def task_description(self) -> str:
        return self.task