import logging
import os
import sys
from functools import cached_property
from typing import Any

import numpy as np
import torch

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_sim_dual_piper_deformable import SimDualPiperDeformableConfig

logger = logging.getLogger(__name__)


class SimDualPiperDeformable(Robot):
    """
    Simulated Dual Piper robot wrapping MujocoStyle3DEnv.

    Produces identical observation/action feature names as the real DualPiper,
    enabling seamless sim-real mixed training via LeRobot.

    Motor names (14-D, matching real DualPiper exactly):
        left_joint_1.pos ~ left_joint_6.pos, left_gripper.pos,
        right_joint_1.pos ~ right_joint_6.pos, right_gripper.pos
    """

    config_class = SimDualPiperDeformableConfig
    name = "sim_dual_piper_deformable"

    # Exactly matching real DualPiper motor names
    MOTOR_NAMES = [
        "left_joint_1.pos",
        "left_joint_2.pos",
        "left_joint_3.pos",
        "left_joint_4.pos",
        "left_joint_5.pos",
        "left_joint_6.pos",
        "left_gripper.pos",
        "right_joint_1.pos",
        "right_joint_2.pos",
        "right_joint_3.pos",
        "right_joint_4.pos",
        "right_joint_5.pos",
        "right_joint_6.pos",
        "right_gripper.pos",
    ]

    def __init__(self, config: SimDualPiperDeformableConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self.env = None
        self.steps_per_action = config.physics_freq // config.control_freq  # 500/25 = 20

        # Previous targets for linear interpolation in send_action (simulates servo PID)
        self._prev_target_left = None
        self._prev_target_right = None
        self._prev_gripper_left = None
        self._prev_gripper_right = None

        # GPU for point cloud processing
        self._use_gpu = torch.cuda.is_available()
        self._device = torch.device("cuda" if self._use_gpu else "cpu")

    # ------------------------------------------------------------------
    # Feature properties (callable without connection)
    # ------------------------------------------------------------------
    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        return {name: float for name in self.MOTOR_NAMES}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        ft: dict[str, type | tuple] = {**self._motors_ft}
        if self.config.include_point_cloud:
            ft["observation.point_cloud"] = (self.config.num_points, 3)
        if self.config.include_rgb:
            for cam_name in self.config.effective_camera_names:
                ft[f"observation.images.{cam_name}"] = (
                    self.config.img_height,
                    self.config.img_width,
                    3,
                )
        return ft

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True  # Sim needs no calibration

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Ensure project root is importable
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from envs.mujoco_style3d_py.mujoco_style3d_env_with_vision_kinematic_rand import MujocoStyle3DEnv

        env_config = self._build_env_config()
        self.env = MujocoStyle3DEnv(env_config)
        self._connected = True
        logger.info(f"{self} connected (sim env created).")

    def disconnect(self) -> None:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.env is not None:
            self.env.close()
            self.env = None
        self._connected = False
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------
    # Observation / Action
    # ------------------------------------------------------------------
    def get_observation(self) -> dict[str, Any]:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        include_visuals = self.config.include_rgb or self.config.include_point_cloud
        obs = self.env.get_observation(include_visuals=include_visuals)

        # Motor state (14-D, matching real DualPiper)
        obs_dict: dict[str, Any] = {}
        for i in range(6):
            obs_dict[f"left_joint_{i + 1}.pos"] = float(obs.joint_left[i])
        obs_dict["left_gripper.pos"] = float(obs.gripper_left)

        for i in range(6):
            obs_dict[f"right_joint_{i + 1}.pos"] = float(obs.joint_right[i])
        obs_dict["right_gripper.pos"] = float(obs.gripper_right)

        # Point cloud
        if self.config.include_point_cloud:
            from common.pcd_utils import process_point_cloud

            pcd = process_point_cloud(
                obs.pcd_scene,
                num_points=self.config.num_points,
                use_gpu=self._use_gpu,
                device=self._device,
            )
            obs_dict["observation.point_cloud"] = pcd.astype(np.float32)

        # RGB images (multi-camera support)
        if self.config.include_rgb:
            for cam_name in self.config.effective_camera_names:
                rgb = self.env._get_camera_rgb(cam_name)
                if rgb is not None:
                    obs_dict[f"observation.images.{cam_name}"] = rgb

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Parse 14-D action dict into env targets
        target_q_left = np.array([action[f"left_joint_{i + 1}.pos"] for i in range(6)])
        target_g_left = float(action["left_gripper.pos"])
        target_q_right = np.array([action[f"right_joint_{i + 1}.pos"] for i in range(6)])
        target_g_right = float(action["right_gripper.pos"])

        # Step physics N times (500Hz / 25Hz = 20 steps per action)
        # Linear interpolation from prev → current target simulates servo PID behavior,
        # preventing velocity spikes from sending the same target 20 times.
        if self._prev_target_left is not None:
            for i in range(self.steps_per_action):
                alpha = (i + 1) / self.steps_per_action
                q_l = self._prev_target_left * (1 - alpha) + target_q_left * alpha
                g_l = self._prev_gripper_left * (1 - alpha) + target_g_left * alpha
                q_r = self._prev_target_right * (1 - alpha) + target_q_right * alpha
                g_r = self._prev_gripper_right * (1 - alpha) + target_g_right * alpha
                self.env.step_kinematic(q_l, q_r, g_l, g_r)
        else:
            # First frame: no prev target, step directly
            for _ in range(self.steps_per_action):
                self.env.step_kinematic(target_q_left, target_q_right, target_g_left, target_g_right)

        self._prev_target_left = target_q_left.copy()
        self._prev_target_right = target_q_right.copy()
        self._prev_gripper_left = target_g_left
        self._prev_gripper_right = target_g_right

        return action

    # ------------------------------------------------------------------
    # Sim-specific helpers (not part of Robot ABC)
    # ------------------------------------------------------------------
    def reset(self) -> dict[str, Any]:
        """Reset the environment (cloth randomization) and return initial observation."""
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.env.reset()
        # Clear interpolation state for new episode
        self._prev_target_left = None
        self._prev_target_right = None
        self._prev_gripper_left = None
        self._prev_gripper_right = None
        return self.get_observation()

    def get_cloth_position(self, vertex_indices=None):
        """Expose cloth mesh positions for task success checking."""
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self.env.get_cloth_position(vertex_indices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_env_config(self) -> dict:
        """Convert SimDualPiperDeformableConfig to the env_config dict expected by MujocoStyle3DEnv."""
        return {
            "xml_path": self.config.xml_path,
            "render": self.config.render,
            "camera_name": self.config.camera_name,
            "style3d": self.config.style3d,
            "cloth_config": self.config.cloth_config,
            "randomization": self.config.randomization,
            "robot_joints": self.config.robot_joints,
        }