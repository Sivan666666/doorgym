from dataclasses import dataclass, field

from ..config import RobotConfig


@RobotConfig.register_subclass("sim_dual_piper_deformable")
@dataclass(kw_only=True)
class SimDualPiperDeformableConfig(RobotConfig):
    """Configuration for the simulated Dual Piper robot with Style3D deformable physics."""

    # MuJoCo + Style3D environment
    xml_path: str = "assets/scene/mujoco/dual_piper_with_green_tshirt_rgb.xml"
    urdf_path: str = "assets/robot/piper/urdf/piper_with_gripper.urdf"

    # Physics
    physics_freq: int = 500
    control_freq: int = 25
    render: bool = False

    # Camera / RGB
    camera_name: str = "photoneo_cam"
    camera_names: list[str] = field(default_factory=lambda: [])
    img_width: int = 640
    img_height: int = 480
    include_rgb: bool = True

    @property
    def effective_camera_names(self) -> list[str]:
        """Return camera_names if set, otherwise fallback to [camera_name]."""
        return self.camera_names if self.camera_names else [self.camera_name]

    # Point Cloud
    include_point_cloud: bool = True
    num_points: int = 2048

    # Style3D credentials
    style3d: dict = field(default_factory=lambda: {
        "username": "SHJD_test01_en",
        "password": "YpCVTFAK",
    })

    # Cloth physics (Style3D parameters)
    cloth_config: dict = field(default_factory=lambda: {
        "stretch_stiff": [40000e-3, 25000e-3, 3000e-3],
        "bend_stiff": [100e-9, 50e-9, 70e-9],
        "density": 43e-3,
    })

    # Randomization
    randomization: dict = field(default_factory=lambda: {
        "enabled": True,
        "target_body_name": "cloth",
        "pos_rand": {
            "enabled": False,
            "range_x": [-0.1, 0.1],
            "range_y": [-0.1, 0.1],
        },
        "rot_rand": {
            "enabled": False,
        },
        "dynamic_settle": {
            "enabled": True,
            "pin_duration": 0.1,
            "settle_duration": 0.1,
        },
    })

    # Robot joint names in MuJoCo XML
    robot_joints: dict = field(default_factory=lambda: {
        "left_arm": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        "left_gripper": ["joint7", "joint8"],
        "right_arm": ["joint1_arm2", "joint2_arm2", "joint3_arm2", "joint4_arm2", "joint5_arm2", "joint6_arm2"],
        "right_gripper": ["joint7_arm2", "joint8_arm2"],
        "gripper_limit": 0.035,
    })