import json
import os
from pathlib import Path
from collections import deque

import numpy as np
import torch


IMAGE_HEIGHT = 54
IMAGE_WIDTH = 96
ACTION_NAMES = [
    "vx",
    "yaw",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_qx",
    "ee_qy",
    "ee_qz",
    "ee_qw",
    "gripper",
]


class DoorDPJsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")

    def write(self, record):
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_list(x, precision=5):
    arr = _to_numpy(x).astype(np.float64)
    return np.round(arr, precision).tolist()


def make_state_feature_names(num_dofs, num_actions, phase_names):
    names = ["base_roll", "base_pitch", "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"]
    names += [f"dof_pos_{i}" for i in range(num_dofs)]
    names += [f"dof_vel_{i}" for i in range(num_dofs)]
    names += [f"last_low_action_{i}" for i in range(num_actions)]
    names += [f"foot_contact_{i}" for i in range(4)]
    names += ["ee_base_x", "ee_base_y", "ee_base_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]
    names += ["gripper_pos"]
    return names


def get_door_dp_state(env, phase_id, phase_names, env_id=0):
    from isaacgym.torch_utils import euler_from_quat, quat_rotate_inverse

    env_id = int(env_id)
    roll, pitch, _ = euler_from_quat(env.root_states[:, 3:7])
    arm_base_pos = env.base_pos
    if hasattr(env, "arm_base_offset"):
        from isaacgym.torch_utils import quat_apply

        arm_base_pos = env.base_pos + quat_apply(env.base_yaw_quat, env.arm_base_offset)
    ee_pos_base = quat_rotate_inverse(env.root_states[:, 3:7], env.ee_pos - arm_base_pos)
    gripper_pos = env.dof_pos[:, -env.cfg.env.num_gripper_joints :].mean(dim=-1, keepdim=True)
    parts = [
        torch.stack([roll, pitch], dim=-1),
        env.base_ang_vel,
        env.dof_pos,
        env.dof_vel,
        env.last_actions,
        env._reindex_feet(env.foot_contacts_from_sensor).to(torch.float32),
        ee_pos_base,
        env.ee_orn / torch.clamp(torch.norm(env.ee_orn, dim=-1, keepdim=True), min=1e-6),
        gripper_pos,
    ]
    return torch.cat(parts, dim=-1)[env_id].detach().cpu().to(torch.float32).numpy()


def get_door_dp_action(env, env_id=0):
    env_id = int(env_id)
    quat = env.ee_goal_orn_quat / torch.clamp(torch.norm(env.ee_goal_orn_quat, dim=-1, keepdim=True), min=1e-6)
    gripper = env.external_gripper_target[:, :1].mean(dim=-1)
    action = torch.cat(
        [
            env.commands[:, 0:1],
            env.commands[:, 2:3],
            env.curr_ee_goal_cart_world[:, :3],
            quat[:, :4],
            gripper[:, None],
        ],
        dim=-1,
    )
    return action[env_id].detach().cpu().to(torch.float32).numpy()


def make_door_dp_log_record(env, step, dp_action, env_id=0, phase_id=None, phase_names=None, extra=None):
    env_id = int(env_id)
    quat = env.ee_goal_orn_quat / torch.clamp(torch.norm(env.ee_goal_orn_quat, dim=-1, keepdim=True), min=1e-6)
    ee_orn = env.ee_orn / torch.clamp(torch.norm(env.ee_orn, dim=-1, keepdim=True), min=1e-6)
    phase_value = None
    phase_name = None
    if phase_id is not None:
        phase_value = int(_to_numpy(phase_id[env_id]).item())
        if phase_names is not None and 0 <= phase_value < len(phase_names):
            phase_name = phase_names[phase_value]
    record = {
        "step": int(step),
        "controlled_env_id": env_id,
        "num_envs": int(env.num_envs),
        "only_controlled_env_uses_dp": True,
        "phase_id": phase_value,
        "phase_name": phase_name,
        "dp_action_names": ACTION_NAMES,
        "dp_action_raw": _to_list(dp_action),
        "applied_action": _to_list(get_door_dp_action(env, env_id)),
        "robot_command": {
            "vx": float(_to_numpy(env.commands[env_id, 0]).item()),
            "vy": float(_to_numpy(env.commands[env_id, 1]).item()),
            "yaw": float(_to_numpy(env.commands[env_id, 2]).item()),
        },
        "base": {
            "xy": _to_list(env.root_states[env_id, :2]),
            "height": float(_to_numpy(env.root_states[env_id, 2]).item()),
            "lin_vel": _to_list(env.base_lin_vel[env_id]),
            "ang_vel": _to_list(env.base_ang_vel[env_id]),
        },
        "ee": {
            "target_pos_world": _to_list(env.curr_ee_goal_cart_world[env_id, :3]),
            "target_quat": _to_list(quat[env_id, :4]),
            "target_delta_rpy": _to_list(env.ee_goal_orn_delta_rpy[env_id]),
            "actual_pos_world": _to_list(env.ee_pos[env_id, :3]),
            "actual_quat": _to_list(ee_orn[env_id, :4]),
            "pos_error": _to_list(env.curr_ee_goal_cart_world[env_id, :3] - env.ee_pos[env_id, :3]),
        },
        "gripper": {
            "target": _to_list(env.external_gripper_target[env_id]),
            "actual_pos": _to_list(env.dof_pos[env_id, -env.cfg.env.num_gripper_joints :]),
        },
    }
    if hasattr(env, "_door_dof_pos"):
        record["door"] = {
            "dof": _to_list(env._door_dof_pos[env_id]),
        }
    elif hasattr(env, "door_dof_pos"):
        record["door"] = {
            "dof": _to_list(env.door_dof_pos[env_id]),
        }
    if extra:
        record["extra"] = extra
    return record


def print_door_dp_log_record(record):
    action = record["dp_action_raw"]
    cmd = record["robot_command"]
    ee = record["ee"]
    gripper = record["gripper"]
    print(
        "[DoorDP]"
        f" step={record['step']}"
        f" env={record['controlled_env_id']}/{record['num_envs']}"
        f" phase={record.get('phase_name')}"
        f" action(vx,yaw,ee,grip)=({action[0]:.3f}, {action[1]:.3f}, "
        f"[{action[2]:.3f}, {action[3]:.3f}, {action[4]:.3f}], {action[9]:.3f})"
        f" cmd=({cmd['vx']:.3f}, {cmd['yaw']:.3f})"
        f" ee_target={ee['target_pos_world']}"
        f" ee_actual={ee['actual_pos_world']}"
        f" ee_err={ee['pos_error']}"
        f" grip={gripper['target']}",
        flush=True,
    )


def _image_pair_from_camera_tensors(camera_images, mask_key, depth_key, env_id=0):
    env_id = int(env_id)
    if mask_key not in camera_images or depth_key not in camera_images:
        return None, None
    mask = np.squeeze(_to_numpy(camera_images[mask_key][env_id])).astype(np.float32)
    masked_depth = np.squeeze(_to_numpy(camera_images[depth_key][env_id])).astype(np.float32)
    mask_u8 = (255.0 * np.clip(mask, 0.0, 1.0)).astype(np.uint8)
    valid = masked_depth[mask > 0.5]
    valid = valid[np.isfinite(valid) & (valid > 0.0)]
    depth_u8 = np.zeros_like(mask_u8)
    if valid.size > 0:
        d_min = float(valid.min())
        d_max = float(valid.max())
        if d_max - d_min < 1e-4:
            scaled = masked_depth / max(d_max, 1e-4)
        else:
            scaled = (masked_depth - d_min) / (d_max - d_min)
        depth_u8 = (255.0 * np.clip(scaled, 0.0, 1.0) * mask).astype(np.uint8)
    # Store as RGB-compatible images for LeRobot/video tools and the shared 3-channel CNN encoder.
    # The masked depth is still a single grayscale depth visualization; all three channels are identical.
    return np.repeat(mask_u8[..., None], 3, axis=-1), np.repeat(depth_u8[..., None], 3, axis=-1)


def images_from_camera_tensors(camera_images, env_id=0):
    mask_key = "wrist_handle_mask" if "wrist_handle_mask" in camera_images else "handle_mask"
    depth_key = "wrist_handle_masked_depth" if "wrist_handle_masked_depth" in camera_images else "handle_masked_depth"
    return _image_pair_from_camera_tensors(camera_images, mask_key, depth_key, env_id)


def dp_image_inputs_from_camera_tensors(camera_images, env_id=0):
    wrist_mask, wrist_depth = images_from_camera_tensors(camera_images, env_id)
    front_mask, front_depth = _image_pair_from_camera_tensors(
        camera_images,
        "front_handle_mask",
        "front_handle_masked_depth",
        env_id,
    )
    return wrist_mask, wrist_depth, front_mask, front_depth


def _zero_image_like(image):
    if image is None:
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    return np.zeros_like(np.asarray(image, dtype=np.uint8))


class DoorDPLeRobotRecorder:
    def __init__(self, root, repo_id, fps, state_feature_names, task, resume=True):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = int(fps)
        self.task = task
        self.state_feature_names = list(state_feature_names)
        self.action_names = list(ACTION_NAMES)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_root = self.root / repo_id
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.state_feature_names),),
                "names": self.state_feature_names,
            },
            "observation.images.wrist_handle_mask": {
                "dtype": "image",
                "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist_masked_depth": {
                "dtype": "image",
                "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.front_handle_mask": {
                "dtype": "image",
                "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.front_masked_depth": {
                "dtype": "image",
                "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                "names": ["height", "width", "channel"],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(self.action_names),),
                "names": self.action_names,
            },
            "subtask_index": {"dtype": "int64", "shape": (1,), "names": ["subtask_index"]},
        }
        if resume and self.dataset_root.exists():
            try:
                self.dataset = LeRobotDataset(repo_id=repo_id, root=str(self.dataset_root))
            except TypeError:
                self.dataset = LeRobotDataset(repo_id, root=str(self.dataset_root))
        else:
            try:
                self.dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    root=str(self.dataset_root),
                    fps=self.fps,
                    features=features,
                    use_videos=True,
                )
            except TypeError:
                self.dataset = LeRobotDataset.create(repo_id, fps=self.fps, root=str(self.dataset_root), features=features)
        self.frame_count = 0
        self.episode_count = 0
        self._write_feature_sidecar()

    def _write_feature_sidecar(self):
        sidecar = {
            "state": self.state_feature_names,
            "action": self.action_names,
            "image_features": [
                "observation.images.wrist_handle_mask",
                "observation.images.wrist_masked_depth",
                "observation.images.front_handle_mask",
                "observation.images.front_masked_depth",
            ],
        }
        out = self.dataset_root / "door_dp_feature_names.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)

    def add_frame(
        self,
        state,
        mask_rgb,
        masked_depth_rgb,
        action,
        subtask_index,
        front_mask_rgb=None,
        front_masked_depth_rgb=None,
    ):
        front_mask_rgb = _zero_image_like(mask_rgb) if front_mask_rgb is None else front_mask_rgb
        front_masked_depth_rgb = _zero_image_like(masked_depth_rgb) if front_masked_depth_rgb is None else front_masked_depth_rgb
        frame = {
            "observation.state": np.asarray(state, dtype=np.float32),
            "observation.images.wrist_handle_mask": np.asarray(mask_rgb, dtype=np.uint8),
            "observation.images.wrist_masked_depth": np.asarray(masked_depth_rgb, dtype=np.uint8),
            "observation.images.front_handle_mask": np.asarray(front_mask_rgb, dtype=np.uint8),
            "observation.images.front_masked_depth": np.asarray(front_masked_depth_rgb, dtype=np.uint8),
            "action": np.asarray(action, dtype=np.float32),
            "subtask_index": np.asarray([subtask_index], dtype=np.int64),
            "task": self.task,
        }
        try:
            self.dataset.add_frame(frame)
        except TypeError:
            frame_without_task = dict(frame)
            frame_without_task.pop("task", None)
            self.dataset.add_frame(frame_without_task, task=self.task)
        self.frame_count += 1

    def save_episode(self):
        try:
            self.dataset.save_episode(task=self.task)
        except TypeError:
            self.dataset.save_episode()
        self.episode_count += 1

    def finalize(self):
        if hasattr(self.dataset, "finalize"):
            self.dataset.finalize()


class RawDoorDPRecorder:
    def __init__(self, raw_root, fps, state_feature_names, task):
        self.raw_root = Path(raw_root)
        self.fps = int(fps)
        self.task = str(task)
        self.state_feature_names = list(state_feature_names)
        self.action_names = list(ACTION_NAMES)
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.frames = {
            "state": [],
            "action": [],
            "wrist_handle_mask": [],
            "wrist_masked_depth": [],
            "front_handle_mask": [],
            "front_masked_depth": [],
            "subtask_index": [],
        }
        self.frame_count = 0
        self.episode_count = 0
        self._write_feature_sidecar()

    def _write_feature_sidecar(self):
        sidecar = {
            "fps": self.fps,
            "state": self.state_feature_names,
            "action": self.action_names,
            "image_features": ["wrist_handle_mask", "wrist_masked_depth", "front_handle_mask", "front_masked_depth"],
            "format": "door_dp_raw_npz_v1",
        }
        out = self.raw_root / "door_dp_feature_names.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)

    def add_frame(
        self,
        state,
        mask_rgb,
        masked_depth_rgb,
        action,
        subtask_index,
        front_mask_rgb=None,
        front_masked_depth_rgb=None,
    ):
        front_mask_rgb = _zero_image_like(mask_rgb) if front_mask_rgb is None else front_mask_rgb
        front_masked_depth_rgb = _zero_image_like(masked_depth_rgb) if front_masked_depth_rgb is None else front_masked_depth_rgb
        self.frames["state"].append(np.asarray(state, dtype=np.float32))
        self.frames["action"].append(np.asarray(action, dtype=np.float32))
        self.frames["wrist_handle_mask"].append(np.asarray(mask_rgb, dtype=np.uint8))
        self.frames["wrist_masked_depth"].append(np.asarray(masked_depth_rgb, dtype=np.uint8))
        self.frames["front_handle_mask"].append(np.asarray(front_mask_rgb, dtype=np.uint8))
        self.frames["front_masked_depth"].append(np.asarray(front_masked_depth_rgb, dtype=np.uint8))
        self.frames["subtask_index"].append(np.asarray([subtask_index], dtype=np.int64))
        self.frame_count += 1

    def _next_episode_path(self):
        existing = sorted(self.raw_root.glob("episode_*.npz"))
        if not existing:
            return self.raw_root / "episode_000000.npz"
        max_idx = -1
        for path in existing:
            try:
                max_idx = max(max_idx, int(path.stem.split("_")[-1]))
            except ValueError:
                continue
        return self.raw_root / f"episode_{max_idx + 1:06d}.npz"

    def save_episode(self):
        if self.frame_count <= 0:
            print("Warning: RawDoorDPRecorder has no frames; skipped saving episode.")
            return
        out = self._next_episode_path()
        payload = {
            "state": np.stack(self.frames["state"], axis=0).astype(np.float32),
            "action": np.stack(self.frames["action"], axis=0).astype(np.float32),
            "wrist_handle_mask": np.stack(self.frames["wrist_handle_mask"], axis=0).astype(np.uint8),
            "wrist_masked_depth": np.stack(self.frames["wrist_masked_depth"], axis=0).astype(np.uint8),
            "front_handle_mask": np.stack(self.frames["front_handle_mask"], axis=0).astype(np.uint8),
            "front_masked_depth": np.stack(self.frames["front_masked_depth"], axis=0).astype(np.uint8),
            "subtask_index": np.stack(self.frames["subtask_index"], axis=0).astype(np.int64),
            "task": np.asarray(self.task),
            "fps": np.asarray(self.fps, dtype=np.int64),
            "state_feature_names": np.asarray(self.state_feature_names, dtype=object),
            "action_names": np.asarray(self.action_names, dtype=object),
        }
        np.savez_compressed(out, **payload)
        self.episode_count += 1
        print(f"Saved raw Door DP episode: {out}")

    def finalize(self):
        pass


def import_lerobot_or_raise():
    try:
        import lerobot  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "LeRobot is required for this DP dataset pipeline. Install dependencies with "
            "`pip install -r high-level/dp/requirements_dp.txt` inside a Python>=3.10 training environment."
        ) from exc


class DoorDPPolicyController:
    def __init__(self, checkpoint, device=None, num_inference_steps=None, action_horizon=None):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        try:
            from .models.door_diffusion_policy import DoorDiffusionPolicy
        except ImportError:
            from models.door_diffusion_policy import DoorDiffusionPolicy

        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.config = ckpt["config"]
        self.stats = {k: v.to(self.device) for k, v in ckpt["stats"].items()}
        self.obs_horizon = int(self.config.get("obs_horizon", 16))
        self.pred_horizon = int(self.config.get("pred_horizon", 32))
        self.action_horizon = int(action_horizon or self.config.get("action_horizon", 16))
        self.action_dim = int(self.config.get("action_dim", len(ACTION_NAMES)))
        self.model = DoorDiffusionPolicy(
            state_dim=int(self.config["state_dim"]),
            action_dim=self.action_dim,
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(self.config.get("num_diffusion_iters", 100)),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.noise_scheduler.set_timesteps(int(num_inference_steps or self.noise_scheduler.config.num_train_timesteps))
        self.obs_buffer = deque(maxlen=self.obs_horizon)
        self.action_queue = deque()

    def _normalize_state(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return (state - self.stats["state_mean"]) / self.stats["state_std"]

    def _normalize_action(self, action):
        return (action - self.stats["action_mean"]) / self.stats["action_std"]

    def _denormalize_action(self, action):
        return action * self.stats["action_std"] + self.stats["action_mean"]

    def append_observation(self, state, mask_rgb, masked_depth_rgb, front_mask_rgb=None, front_masked_depth_rgb=None):
        state = self._normalize_state(state)
        mask = torch.as_tensor(mask_rgb, dtype=torch.uint8, device=self.device).permute(2, 0, 1)
        depth = torch.as_tensor(masked_depth_rgb, dtype=torch.uint8, device=self.device).permute(2, 0, 1)
        front_mask_rgb = _zero_image_like(mask_rgb) if front_mask_rgb is None else front_mask_rgb
        front_masked_depth_rgb = _zero_image_like(masked_depth_rgb) if front_masked_depth_rgb is None else front_masked_depth_rgb
        front_mask = torch.as_tensor(front_mask_rgb, dtype=torch.uint8, device=self.device).permute(2, 0, 1)
        front_depth = torch.as_tensor(front_masked_depth_rgb, dtype=torch.uint8, device=self.device).permute(2, 0, 1)
        item = (state, mask, depth, front_mask, front_depth)
        if len(self.obs_buffer) == 0:
            for _ in range(self.obs_horizon):
                self.obs_buffer.append(item)
        else:
            self.obs_buffer.append(item)

    @torch.no_grad()
    def sample_action_chunk(self):
        state = torch.stack([x[0] for x in self.obs_buffer], dim=0).unsqueeze(0)
        mask = torch.stack([x[1] for x in self.obs_buffer], dim=0).unsqueeze(0)
        depth = torch.stack([x[2] for x in self.obs_buffer], dim=0).unsqueeze(0)
        front_mask = torch.stack([x[3] for x in self.obs_buffer], dim=0).unsqueeze(0)
        front_depth = torch.stack([x[4] for x in self.obs_buffer], dim=0).unsqueeze(0)
        action = torch.randn(1, self.pred_horizon, self.action_dim, device=self.device)
        for timestep in self.noise_scheduler.timesteps:
            ts = torch.full((1,), int(timestep), device=self.device, dtype=torch.long)
            noise_pred = self.model(action, ts, state, mask, depth, front_mask, front_depth)
            action = self.noise_scheduler.step(noise_pred, timestep, action).prev_sample
        action = self._denormalize_action(action[0]).detach().cpu().numpy()
        self.action_queue.clear()
        for row in action[: self.action_horizon]:
            self.action_queue.append(row.astype(np.float32))

    def act(self, state, mask_rgb, masked_depth_rgb, front_mask_rgb=None, front_masked_depth_rgb=None):
        self.append_observation(state, mask_rgb, masked_depth_rgb, front_mask_rgb, front_masked_depth_rgb)
        if not self.action_queue:
            self.sample_action_chunk()
        return self.action_queue.popleft()


def apply_door_dp_action(env, action, env_id=0, delta_rpy_fn=None):
    env_id = int(env_id)
    action = np.asarray(action, dtype=np.float32)
    quat = torch.as_tensor(action[5:9], dtype=torch.float32, device=env.device)
    quat = quat / torch.clamp(torch.norm(quat), min=1e-6)
    env.commands[env_id, 0] = float(action[0])
    env.commands[env_id, 1] = 0.0
    env.commands[env_id, 2] = float(action[1])
    env.curr_ee_goal_cart_world[env_id, :3] = torch.as_tensor(action[2:5], dtype=torch.float32, device=env.device)
    env.ee_goal_orn_quat[env_id, :4] = quat
    if delta_rpy_fn is not None:
        target_pos = env.curr_ee_goal_cart_world[env_id : env_id + 1]
        target_quat = env.ee_goal_orn_quat[env_id : env_id + 1]
        env_ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
        try:
            delta_rpy = delta_rpy_fn(target_pos, target_quat, env_ids=env_ids)
        except TypeError:
            delta_rpy = delta_rpy_fn(target_pos, target_quat)
        env.ee_goal_orn_delta_rpy[env_id : env_id + 1] = delta_rpy
    env.external_gripper_target[env_id, :] = float(action[9])
    if hasattr(env, "freeze_arm_default"):
        env.freeze_arm_default[env_id] = False
    if hasattr(env, "freeze_arm_zero"):
        env.freeze_arm_zero[env_id] = False
