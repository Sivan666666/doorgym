from dataclasses import dataclass, field

from ..configs import CameraConfig

__all__ = ["PhotoneoCameraConfig"]


@CameraConfig.register_subclass("photoneo")
@dataclass
class PhotoneoCameraConfig(CameraConfig):
    """
    Photoneo PhoXi 3D 扫描仪配置
    
    Example:
        ```python
        config = PhotoneoCameraConfig(
            device_id="PAG-076",
            fps=20,
            translation=[1.54116268, 0.13879753, 0.75927529],
            quaternion=[0.706, -0.695, 0.094, -0.096]
        )
        camera = PhotoneoCamera(config)
        camera.connect()
        point_cloud = camera.async_read()
        ```
    
    Attributes:
        device_id: Photoneo 设备 ID (不含 'PhotoneoTL_DEV_' 前缀)
        fps: 采集帧率（仅用于标识，实际采集速度由硬件决定）
        width, height: 占位参数（保持接口一致，Photoneo 不使用）
        translation: 相机在世界坐标系的平移 [x, y, z] (米)
        quaternion: 相机在世界坐标系的旋转四元数 [x, y, z, w]
        calibration_path: 外参标定文件路径（可选）
    """
    
    device_id: str = "PAG-076"

    # 点云采样参数（处理后的点数）
    num_points: int = 2048
    
    # 外参（优先级：translation+quaternion > calibration_path）

    translation: list[float] | None = field(
        default_factory=lambda: [-0.03846401,-0.11231157,1.13300097]
    )
    quaternion: list[float] | None = field(
        default_factory=lambda: [0.70629295,-0.69512124,0.09361616,-0.09587879]
    )
    # translation: list[float] | None = None
    # quaternion: list[float] | None = None
    calibration_path: str | None = None
    
    def __post_init__(self):
        # Photoneo 点云分辨率固定，这里的 width/height 只是占位
        if self.width is None:
            self.width = 640
        if self.height is None:
            self.height = 480
        if self.fps is None:
            self.fps = 25  # 典型值