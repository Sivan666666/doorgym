from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


# TODO : 未来可能需要把这点云传感器单独拉一个配置，暂时先这样兼容一下
@dataclass
class PointCloudSensorConfig:
    """
    3D 点云传感器配置
    
    Note:
        配置层独立管理点云参数，运行时会转换为对应的 CameraConfig
    """
    enabled: bool = False
    camera_type: str = "photoneo"  # "photoneo" or "zed"
    device_id: str = "PAG-076"
    num_points: int = 2048  # FPS 采样后的点数
    
    # 外参（默认值 - 你测试成功的值）
    translation: list[float] = field(
        default_factory=lambda: [1.54116268, 0.13879753, 0.75927529]
    )
    quaternion: list[float] = field(
        default_factory=lambda: [0.58455770, 0.60577063, -0.40007590, -0.36231688]
    )
    
    calibration_path: str | None = None
    fps: int = 25
    
    # 占位参数（保持接口一致）
    width: int = 640
    height: int = 480

@RobotConfig.register_subclass("dual_piper")
@dataclass
class DualPiperConfig(RobotConfig):
    # Port to connect to the arm
    # port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    # 最大相对目标位置的限制
    max_relative_target: int | None = None

    # cameras
    # 定义机器人使用的相机配置字典，键是相机的名称（字符串），值是 CameraConfig 类型的配置对象
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    # 是否使用角度单位（degree）而不是弧度（radian）
    use_degrees: bool = False

    point_cloud: PointCloudSensorConfig = field(default_factory=PointCloudSensorConfig)


