import pinocchio as pin
import numpy as np
import os
from typing import List, Optional, Union, Dict
from scipy.spatial.transform import Rotation as R
from transforms3d.quaternions import mat2quat, quat2mat
import sys
from loguru import logger

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from robot.kinematics.base_kinematics import BaseKinematics

class PinocchioKinematics(BaseKinematics):
    def __init__(self, urdf_path: str, base_link: str = "base_link", ee_link: str = "tcp_link",
                 base_height: float = 0.0):
            super().__init__(urdf_path, base_link, ee_link)

            if not os.path.exists(urdf_path):
                raise FileNotFoundError(f"URDF file not found: {urdf_path}")

            # 加载模型
            self.model = pin.buildModelFromUrdf(urdf_path)
            self.data = self.model.createData()

            # 获取 Frame ID
            if not self.model.existFrame(ee_link):
                raise ValueError(f"Frame '{ee_link}' does not exist in URDF.")
            self.ee_frame_id = self.model.getFrameId(ee_link)
            self.base_frame_id = self.model.getFrameId(base_link) # Pinocchio默认World是0，如果base_link不是root需要注意

            self.num_joints = self.model.nq

            # TODO: 这里仅对夹爪，就直接减2了，后续考虑更多自由度这里最好传一下参
            self.arm_joints = self.model.nq -2

            # Ground penetration check: threshold = -base_height
            # IK works in URDF local frame; ground plane in local frame is at -base_height.
            # e.g. base at world Z=+0.015 → ground in local frame is Z=-0.015
            self.ground_z_threshold = -base_height
            # 默认 False (新): grasp z~0 时 True 容易把 IK seed 全拒. 真机/特殊场景显式开.
            self.ground_check_enabled = False

            # Cache body frame IDs for ground checking (exclude base_link/dummy)
            self._ground_check_frame_ids = []
            for i in range(self.model.nframes):
                frame = self.model.frames[i]
                if frame.type == pin.FrameType.BODY and frame.name not in ('dummy_link', 'base_link'):
                    self._ground_check_frame_ids.append(i)

    def _ensure_numpy(self, data) -> np.ndarray:
        "ensure data is numpy array"
        if isinstance(data, list):
            return np.array(data)
        elif hasattr(data, 'cpu'): # torch tensor
            return data.cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
    def solve_fk(self, joint_angles: Union[List, np.ndarray], ee_link: Optional[str] = None) -> np.ndarray:
        """
        Forward Kinematics
        Returns: [x, y, z, qw, qx, qy, qz]
        """
        q = self._ensure_numpy(joint_angles)

        if len(q) < self.model.nq:
            q_padded = np.zeros(self.model.nq)
            q_padded[:len(q)] = q
            q = q_padded
            # logger.info(f"Padded joint angles from {len(joint_angles)} to {self.model.nq}")
        elif len(q) > self.model.nq:
            # 截断多余的关节
            q = q[:self.model.nq]
            logger.warning(f"Truncated joint angles from {len(joint_angles)} to {self.model.nq}")
        
        
        # 指定计算的 Frame
        fid = self.ee_frame_id
        if ee_link is not None and ee_link != self.ee_link:
            fid = self.model.getFrameId(ee_link)

        # 更新运动学
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # 获取位姿
        oMf = self.data.oMf[fid]
        pos = oMf.translation
        # Pinocchio quaternion is [x,y,z,w] in internal eigen, but let's use scipy for safety conversion
        # oMf.rotation 是 3x3 旋转矩阵
        quat_wxyz = mat2quat(oMf.rotation)
        
        return np.concatenate([pos, quat_wxyz])
    
    def _check_ground_penetration(self, q: np.ndarray, threshold: float = None) -> tuple:
        """
        Check if any robot link penetrates ground plane.

        Args:
            q: Joint configuration (can be arm_joints only, will be padded)
            threshold: Ground z threshold. Links with z < threshold are considered penetrating.

        Returns: (penetrates: bool, frame_name: str, min_z: float)
        """
        if threshold is None:
            threshold = self.ground_z_threshold

        if len(q) < self.model.nq:
            q_full = np.zeros(self.model.nq)
            q_full[:len(q)] = q
            q = q_full

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        min_z = float('inf')
        min_frame = ""
        for fid in self._ground_check_frame_ids:
            z = self.data.oMf[fid].translation[2]
            if z < min_z:
                min_z = z
                min_frame = self.model.frames[fid].name

        return (min_z < threshold, min_frame, min_z)

    def solve_ik(
        self, 
        target_pose: Union[np.ndarray, List[float]],
        seed_joints: Optional[np.ndarray] = None,
        threshold: float = 1e-4,
        pose_constraint: Optional[Union[np.ndarray, List[float], Dict]] = None,
        constraint_frame: str = "world",
        ee_link: Optional[str] = None,
        max_restarts: int = 30, 
        strict_seed: bool = False,  # 新增参数：是否严格使用用户提供的seed
        **kwargs,
    ) -> Optional[np.ndarray]:
        """
        IK Solver using Damped Least Squares with Local and World Frame constraints.
        
        Args:
            target_pose: [x, y, z, qw, qx, qy, qz] 目标的完整位姿
            pose_constraint: 
                - Mode A (Simple): [x,y,z, rx,ry,rz] weights (e.g. [1,1,1,0,0,0]).
                - Mode B (Mixed): {"local": [...], "world": [...]} for advanced solvers.
                -  something important: not rpy, but rx,ry,rz (rotation vector)
            strict_seed: 
                - True: 严格使用用户提供的seed_joints，失败则直接返回None
                - False: 用户seed失败后尝试其他种子（默认）     
            kwargs:          
                - max_iter (int): 最大迭代次数 (默认 1000)
                - dt (float): 步长 (默认 0.1)
                - damp (float): 阻尼系数 (默认 1e-6)
        """

        # --- 1. 参数解析与路由 (Scientific Routing) ---
        mask_local = None
        mask_world = None

        # 情况 A: 混合模式 (传入了字典)
        if isinstance(pose_constraint, dict):
            mask_local = pose_constraint.get("local")
            mask_world = pose_constraint.get("world")
            # 安全转换为 numpy
            if mask_local is not None: mask_local = np.array(mask_local)
            if mask_world is not None: mask_world = np.array(mask_world)

        # 情况 B: 简单模式 (传入了列表/数组 或 None)
        else:
            # 默认全约束
            if pose_constraint is None:
                constraint_arr = np.ones(6)
            else:
                constraint_arr = np.array(pose_constraint)
            
            # 自动补全：如果只给了位置 [1,1,1]，补全为 [1,1,1, 0,0,0]
            if constraint_arr.shape == (3,):
                constraint_arr = np.concatenate([constraint_arr, [0, 0, 0]])
            
            # 根据 frame 分配
            if constraint_frame == "local":
                mask_local = constraint_arr
            else:
                mask_world = constraint_arr
        
         # --- 2. 提取求解参数 ---
        max_iter = kwargs.get("max_iter", 1000)
        dt = kwargs.get("dt", 0.05)
        damp = kwargs.get("damp", 1e-6)
        threshold = kwargs.get("threshold", 1e-4) 
        
        pose_arr = np.array(target_pose)
        fid = self.model.getFrameId(ee_link) if ee_link else self.ee_frame_id



        # --- 3. 种子策略 ---
        seed_strategies = []
        
        # 策略A: 用户提供的seed（优先级最高）
        if seed_joints is not None:
            seed_joints_array = self._ensure_numpy(seed_joints)
            if len(seed_joints_array) < self.model.nq:
                seed_joints_array = np.concatenate([seed_joints_array, np.zeros(self.model.nq - len(seed_joints_array))])
            else:
                seed_joints_array = seed_joints_array[:self.model.nq]
            seed_strategies.append(("user_seed", seed_joints_array.copy()))
            if strict_seed:
                max_restarts = 1  # 严格模式：只尝试一次
        
        # 策略B: 如果允许尝试其他种子
        if not strict_seed:
            # 零位
            seed_strategies.append(("zero", np.zeros(self.model.nq)))
            # 中间位置
            seed_strategies.append(("middle", (self.model.lowerPositionLimit + self.model.upperPositionLimit) / 2.0))
            # 随机种子（多样性探索）
            for i in range(max_restarts - len(seed_strategies)):
                # random_seed = np.random.uniform(self.model.lowerPositionLimit, self.model.upperPositionLimit)
                random_seed = pin.randomConfiguration(self.model) # 自带随机配置，固定随机序列
                seed_strategies.append((f"random_{i}", random_seed))

        seed_strategies = seed_strategies[:max_restarts]

        # --- 4. 迭代尝试 ---
        ground_check = kwargs.get("ground_check", self.ground_check_enabled)
        ground_z = kwargs.get("ground_z_threshold", self.ground_z_threshold)

        # Track best penetrating solution as fallback
        best_penetrating_sol = None  # (q_sol, min_z, frame_name)
        rejected_count = 0

        for restart_idx, (strategy_name, q_init) in enumerate(seed_strategies):

            q = q_init.copy()
            q_sol = self._solve_ik_core(
                q_init=q,
                target_pose=pose_arr,
                mask_local=mask_local, mask_world=mask_world,
                max_iter=max_iter, dt=dt, damp=damp, threshold=threshold,fid=fid
            )

            if q_sol is not None:
                # Ground penetration filter
                if ground_check:
                    penetrates, pen_frame, pen_z = self._check_ground_penetration(q_sol, ground_z)
                    if penetrates:
                        rejected_count += 1
                        # Keep the least-penetrating solution as fallback
                        if best_penetrating_sol is None or pen_z > best_penetrating_sol[1]:
                            best_penetrating_sol = (q_sol.copy(), pen_z, pen_frame)
                        if kwargs.get("verbose", False):
                            logger.debug(
                                f"IK '{strategy_name}' rejected: {pen_frame} z={pen_z:.4f} < {ground_z}"
                            )
                        continue  # try next seed

                if kwargs.get("verbose", False):
                    logger.debug(
                        f"IK solved using strategy '{strategy_name}' "
                        f"(attempt {restart_idx}, {rejected_count} ground-rejected), "
                        f"seed joints {q_init}"
                    )
                return q_sol[:self.arm_joints]

        # Fallback: all converged solutions penetrate ground
        if best_penetrating_sol is not None:
            sol, pen_z, pen_frame = best_penetrating_sol
            logger.warning(
                f"IK: all {rejected_count} converged solutions penetrate ground. "
                f"Returning least-penetrating: {pen_frame} z={pen_z:.4f}"
            )
            return sol[:self.arm_joints]

        return None
    
    
    def _solve_ik_core(
        self, 
        q_init: np.ndarray,
        target_pose: np.ndarray,
        mask_world: Optional[np.ndarray],
        mask_local: Optional[np.ndarray],
        max_iter: int,
        dt: float,
        damp: float,
        threshold: float,
        fid: int
    ) -> Optional[np.ndarray]:
        """
        CLIK迭代求解器 (单个seed)
        
        Args:
            q_init: 初始关节角度
            oMdes: 目标位姿（SE3对象）
            fid: 末端frame ID
            W_local: 局部坐标系约束权重矩阵 (6x6 或 None)
            W_world: 世界坐标系约束权重矩阵 (6x6 或 None)
            max_iter: 最大迭代次数
            dt: 步长
            damp: 阻尼系数
            threshold: 收敛阈值
            
        Returns:
            成功: 收敛的关节角度
            失败: None
        """

        q = q_init.copy()
        q_min = self.model.lowerPositionLimit
        q_max = self.model.upperPositionLimit

        pos_des = target_pose[:3]
        quat_wxyz = target_pose[3:7]
        r_des = quat2mat(quat_wxyz)
        oMdes = pin.SE3(r_des, pos_des)

        # 预处理 Mask 矩阵
        W_world = np.diag(mask_world) if mask_world is not None else None
        W_local = np.diag(mask_local) if mask_local is not None else None

        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
        
            oMtool = self.data.oMf[fid]
            
            J_list = []
            err_list = []

            # --- A: Local Frame 约束 ---
            if W_local is not None:
                # 误差: oMtool.inv * oMdes (在工具系下看目标的偏差)
                dM_local = oMtool.actInv(oMdes)
                err_local_full = pin.log6(dM_local).vector
                
                # 应用 Mask
                err_local = W_local @ err_local_full
                
                # 雅可比: LOCAL
                J_local_full = pin.computeFrameJacobian(self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL)
                J_local = W_local @ J_local_full
                
                err_list.append(err_local)
                J_list.append(J_local)

            # --- B: World Frame 约束 ---
            if W_world is not None:
                # # 误差: 需要将 Local 误差转换到 World 对齐系
                err_pos_world = pos_des - oMtool.translation
                
                # 2. 旋转误差 (将旋转差转到 World 系)
                # log3 计算的是 R_current^T * R_des (Local frame error)
                # 我们需要将其左乘 R_current 变换到 World frame
                rot_err_local = pin.log3(oMtool.rotation.T @ r_des)
                err_rot_world = oMtool.rotation @ rot_err_local
                
                # 合并为 6维误差向量 [dx, dy, dz, drx, dry, drz] (World Frame)
                err_world_full = np.concatenate([err_pos_world, err_rot_world])


                # 应用 Mask
                err_world = W_world @ err_world_full
                
                # 雅可比: LOCAL_WORLD_ALIGNED (即世界系下的雅可比)
                J_world_full = pin.computeFrameJacobian(self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                J_world = W_world @ J_world_full
                
                err_list.append(err_world)
                J_list.append(J_world)

            # --- 4. 堆叠与求解 ---
            # 将所有任务垂直堆叠
            if not err_list:
                break # 无约束

            err_stack = np.concatenate(err_list)
            J_stack = np.vstack(J_list)

            # 检查误差 (注意：只检查被 Mask 选中的维度的误差)
            if np.linalg.norm(err_stack) < threshold:
                return q

            # DLS 求解: (J^T * J + damp * I) * dq = J^T * err
            # 注意: Pinocchio 的 log6(dM) 是 "从当前指向目标" 的速度向量，即 Error = Target - Current
            # 所以我们用 +err_stack
            H = J_stack.T @ J_stack + damp * np.eye(self.model.nv)
            g = J_stack.T @ err_stack
            
            dq = np.linalg.solve(H, g)
            q_new = pin.integrate(self.model, q, dq * dt)
            q_new = np.clip(q_new, q_min, q_max)

            if np.linalg.norm(q_new - q) < 1e-6 and i < max_iter - 1:
                dt *= 0.5  # 动态减小步长
            q = q_new

        return None


# # ================= 使用示例 =================
if __name__ == "__main__":
    
    project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Project Path:", project_path)
    urdf_path = os.path.join(project_path, "assets", "Urdf", "piper_description","piper_with_gripper.urdf")
    kinematics = PinocchioKinematics(urdf_path, base_link="base_link", ee_link="tcp_link")
    
    # 获取初始位姿信息
    init_fk = kinematics.solve_fk([0.0]*8)
    init_pos = init_fk[0:3]
    init_quat = init_fk[3:7]
    print("=" * 60)
    print("Initial FK pos:", init_pos)
    print("Initial FK quat (wxyz):", init_quat)
    print("=" * 60)

    # ============================= 
    # 场景 1: 简单模式 - 世界坐标系下约束
    # =============================
    print("\n[Scenario 1] World Frame - ")
    target_pose = [0.5, 0, 0.5, 1, 0, 0, 0]  # Z轴朝上的姿态
    result = kinematics.solve_ik(
        target_pose, 
        pose_constraint=[1, 1, 0, 0, 0, 0],  # 位置全约束
        constraint_frame="world"
    )
    if result is not None:
        print("🎯 target_pose:", target_pose)
        print("✓ IK Result:", result)
        fk_result = kinematics.solve_fk(result)
        print("  Achieved pose:", fk_result)
    else:
        print("✗ IK failed to converge")

    
    # =============================
    #           初始状态测试
    # =============================
    print("\n[Scenario 2] Init Frame  Test- ")
    # 姿态1;rpy姿态， 但无法很好控制工具坐标系的朝向
    target_rpy = np.array([00.0, 90.0, 00.0])* np.pi/180.0
    from transforms3d.euler import euler2mat, mat2euler, euler2quat, quat2euler
    quat_wxyz = euler2quat(target_rpy[0], target_rpy[1], target_rpy[2])  # returns w, x, y, z
    # 姿态2，旋转矩阵计算的姿态，向前加上工作坐标系的旋转
    R_init = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    quat_wxyz = mat2quat(R_init)  

    init_fk = kinematics.solve_fk([0.0]*8)
    ik = kinematics.solve_ik(init_fk)
    init_pos = init_fk[0:3]
    init_quat = init_fk[3:7]
    target_xyz = init_pos.tolist()
    init_rpy = R.from_quat([init_quat[1], init_quat[2], init_quat[3], init_quat[0]]).as_euler('xyz') *180.0/np.pi
    
    # 逆解
    target_pos = target_xyz + quat_wxyz.tolist()
    print("Target pos + quat:", target_pos)
    result = kinematics.solve_ik(target_pos, 
                                pose_constraint=[1,1,1,1,1,1],
                                # seed_joints=[0.0]*8 # if not give this seed joints, will loss
                                ) 
    if result is not None:
        print("🎯 target_pose:", target_pose)
        print("✓ IK Result:", result)
        fk_result = kinematics.solve_fk(result)
        print("  Achieved pose:", fk_result)
    else:
        print("✗ IK failed to converge")

    # =============================
    # 场景 3: 工具坐标系下约束
    # =============================
    print("\n[Scenario 3] Local Frame ")
    target_position = [0.31056932, -0.1639544 + 0.25,   0.00049935]
    
    target_position = [0.04664177, -0.14751578,  0.00753939]

    heading_yaw_degrees = 0.0
    r = R.from_euler('z', heading_yaw_degrees, degrees=True)
    quat = r.as_quat()  # x, y, z, w
    quat_wxyz = [quat[3], quat[0], quat[1], quat[2]]  # w, x, y, z


    def calculate_base_to_target_horizontal_quat(base_pos: np.ndarray, target_pos:np.ndarray) -> np.ndarray:
        """计算从 base_pos 指向 target_pos 的四元数表示的朝向"""
        dx = target_pos[0] - base_pos[0]
        dy = target_pos[1] - base_pos[1]
        yaw = np.arctan2(dy, dx)

        half_yaw = yaw / 2.0
        qw = np.cos(half_yaw)
        qx = 0.0
        qy = 0.0
        qz = np.sin(half_yaw)    
        
        return np.array([qw, qx, qy, qz])
    quat_wxyz = calculate_base_to_target_horizontal_quat(np.array([0.0,0.3,0.0]),np.array(target_position)).tolist()

    target_pose = target_position + quat_wxyz  # 这里的姿态只需保证 Z 轴朝向大致正确
    target_pose = [0.04664177,  -0.14751578,  0.00753939,  0.80668188,  0.0, 0.0, -0.59098591]
    seed_joints = [-5.63146385e-06,  8.24992976e-06,  6.34652006e-23, -1.08965383e-05, 8.55372578e-06,  1.48596908e-05]
    result = kinematics.solve_ik(
        target_pose,
        pose_constraint=[1, 1, 1, 1, 0, 1],  # 忽略世界系 pitch, 合页约束
        constraint_frame="local",
        ee_link="tcp_link_bottom",
        seed_joints = seed_joints,
        verbose=True,
    )
    if result is not None:
        print("🎯 target_pose:", target_pose)
        print("✓ IK Result:", result)
        fk_result = kinematics.solve_fk(result)
        print("  Achieved pose:", fk_result)
    else:
        print("✗ IK failed to converge")

    # =============================
    # 场景 3: 混合坐标系约束 - 桌面贴合任务
    # =============================
    print("\n[Scenario 3] Mixed Frame - 桌面贴合 + 工具朝向约束")
    target_position = [0.31056932, -0.1639544 + 0.25,   0.00049935]
    result = kinematics.solve_ik(
        target_pose,
        pose_constraint={
            # 世界系：必须贴合桌面 (Z=1)，不能倾斜 (Rx=1, Ry=1)，绕Z转动随意
            "world": [0, 0, 1, 1, 1, 0],
            # 局部系：工具 Z 轴朝向对齐 (Rx=1, Ry=1)，绕工具转动随意 (Rz=0)
            "local": [0, 0, 0, 1, 1, 0]
        }
    )
    if result is not None:
        print("✓ IK Result:", result)
        fk_result = kinematics.solve_fk(result)
        print("  Achieved pose:", fk_result)
    else:
        print("✗ IK failed to converge")

    # =============================
    # 场景 5: 地面穿透检测测试
    # =============================
    print("\n[Scenario 5] Ground Penetration Check - 低位目标 (z=0.005)")
    target_pose = [0.25, -0.1, 0.005, 1, 0, 0, 0]
    result = kinematics.solve_ik(
        target_pose,
        pose_constraint=[1, 1, 1, 0, 0, 0],
        verbose=True,
    )
    if result is not None:
        penetrates, frame, min_z = kinematics._check_ground_penetration(result)
        print(f"✓ IK Result: {result}")
        print(f"  Ground check: penetrates={penetrates}, lowest_frame={frame}, min_z={min_z:.6f}")
    else:
        print("✗ IK failed (all solutions penetrate ground)")

    # 对比：关闭地面检测
    print("\n[Scenario 5b] Same target, ground_check=False")
    result_no_check = kinematics.solve_ik(
        target_pose,
        pose_constraint=[1, 1, 1, 0, 0, 0],
        ground_check=False,
    )
    if result_no_check is not None:
        penetrates, frame, min_z = kinematics._check_ground_penetration(result_no_check)
        print(f"✓ IK Result: {result_no_check}")
        print(f"  Ground check: penetrates={penetrates}, lowest_frame={frame}, min_z={min_z:.6f}")
    else:
        print("✗ IK failed to converge")
