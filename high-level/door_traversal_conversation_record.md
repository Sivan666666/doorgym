# 对话记录整理：Image2DoorTraverse / DoorTraverse-ACT 论文方案

> 本文档整理了本次对话中关于“四足机器狗 + 机械臂门穿越任务”的主要讨论、方案演化、论文故事、创新点、相关工作对比和实验设计建议。  
> 该记录为结构化整理版，不是逐字逐句转录。

---

## 0. 最终确定的论文主线

最终认可的整体方案为：

```text
ArtiCraft 生成门
→ 结构事件关键帧
→ residual Keyframe-ACT
→ 四足 base-arm 协同
→ 真实 door traversal 实验
```

更准确的定位是：

```text
Image → Articulated Door Digital Twin
→ Handle/Hinge-aware Trajectory Synthesis
→ Structure-Event Keyframes
→ Residual Keyframe-ACT
→ Quadruped Door Traversal
```

核心观点：

> 对于开门 / 门穿越这类具有明确 articulated structure 的任务，机器人不一定需要从人类视频中模仿手部轨迹，而应该首先恢复门的可交互结构，再基于结构生成轨迹，并通过学习策略进行闭环修正。

---

## 1. 从“视频提取手部轨迹”到“结构先验生成轨迹”

### 1.1 最初设想

最初考虑从第三视角固定 RGB-D 视频中提取人手轨迹：

1. RGB 图像用 SAM 分割手部 mask；
2. mask 投影到 depth 上得到手部点云；
3. 提取手部中心点 pose；
4. 用 GPT / VLM 判断关键帧，例如是否接触物体、物体状态是否改变、手是否闭合；
5. 只保存关键帧手部 pose；
6. 中间插值生成机械臂 EE 轨迹；
7. 将夹爪开合作为 gripper command。

### 1.2 对该方案的评价

该方案可作为 prototype，但存在几个核心问题：

- SAM mask + depth 主要能得到 3D position，不能稳定得到完整 EE 6D pose。
- 手部中心点不一定等于机器人 TCP 应该到达的位置。
- VLM 适合判断动作阶段，不适合精确定位帧级接触时刻。
- 直接重放 camera/base 坐标下的手轨迹，对物体位置变化不鲁棒。
- 关键帧之间不能简单线性插值，需要 IK、碰撞检测、速度 / 加速度 / jerk 限制。

因此建议从：

```text
hand center trajectory imitation
```

转为：

```text
object/contact/structure-centric EE keypose generation
```

---

## 2. 为什么不一定需要从 video 中提取轨迹

后来明确意识到：如果系统已经能够从单张图像生成一模一样的铰链门，并且知道：

- 把手中心点；
- 把手轴向；
- 把手旋转方向；
- 门铰链轴；
- 门开合方向；
- 门板几何与 joint limit；

那么不必从视频中提取人手轨迹。

可以直接使用结构参数生成操作轨迹：

| 门结构信息 | 对应机器人动作 |
|---|---|
| 把手中心点 | 定义 gripper 抓取点 |
| 把手轴向 | 定义 gripper 姿态 |
| 把手旋转方向 | 定义旋转 / 解锁动作 |
| 门铰链轴 | 定义门板运动圆弧 |
| 开门方向 | 定义推 / 拉方向 |
| 门打开角度 | 定义任务完成条件 |

因此论文问题从：

> 从人类视频中学习怎么开门

转化为：

> 从单张图像中恢复可交互 articulated door，并基于门结构自动生成机器人门穿越策略。

这被认为是更清晰、更可解释、更适合论文投稿的方向。

---

## 3. 与 real-to-sim-to-real / video-to-robot 方向的区别

已有 real-to-sim-to-real / video-to-robot 方法典型流程为：

```text
Human video
→ hand/object motion extraction
→ robot retargeting
→ simulation refinement
→ real execution
```

代表性方向包括：

- YOTO / YOTO++：从人类视频中提取手部轨迹和关键帧，再注入双臂机器人；
- Video2Sim2Real：从单个 RGB-D 人类视频中提取人手和物体 motion priors，再做 object-centric keyframe refinement；
- Human2Sim2Robot：从人类 RGB-D 视频中提取物体轨迹和 pre-manipulation hand pose，用于仿真 RL；
- X-SIM：从 RGB-D 人类视频中提取 object motion，用 object-centric reward 训练 policy。

你的区别：

```text
不是从视频学习“人怎么动”，
而是从单张图像恢复“门怎么被打开”。
```

更正式的表述：

> 现有 real-to-sim-to-real 方法通常将人类演示视频作为技能来源，而本文将 articulated door structure 作为技能来源。本文不依赖人手轨迹重定向，而是通过门的把手、铰链和开合约束直接合成可解释的门穿越轨迹。

---

## 4. 与 legged mobile manipulation / 开门方向的区别

已有四足 / 移动机器人开门相关工作大致包括：

| 方向 | 特点 | 局限 |
|---|---|---|
| 特制末端执行器开门 | 例如 hook / 专用 gripper | 对普通夹爪和泛化不友好 |
| RGB / RL 训练开门策略 | 不需要显式轨迹先验 | 收敛慢、GPU 消耗大、奖励设计复杂 |
| teacher-student RL / whole-body policy | 能处理 push/pull 或复杂接触 | 训练成本高，末端精度不一定高 |
| haptic primitive / 状态机 | 工程鲁棒，反馈强 | 依赖手工设计流程，学习能力有限 |

你的区别：

```text
先用单图生成门结构先验，
再用结构轨迹作为 nominal behavior，
最后用 residual Keyframe-ACT 做闭环修正。
```

重点优势：

- 比纯 RL 更数据高效；
- 比端到端 whole-body policy 更容易保证机械臂末端精度；
- 比手工 primitive 更具学习和闭环修正能力；
- 比普通 door opening 更完整，因为任务目标是 door traversal。

---

## 5. Door Opening 改为 Door Traversal

最终建议将任务从 **door opening** 改成 **door traversal**。

因为真实目标不是“把门打开就结束”，而是：

```text
机器狗走近门
→ 停下
→ 机械臂抓住把手
→ 扭开把手
→ 机械臂和机器狗一起往前运动
→ 推开门
→ 机械臂持续抵住门
→ 机器狗整体穿过门洞
```

Door traversal 比 door opening 更完整，包含以下阶段：

1. approach door；
2. pre-grasp handle；
3. grasp handle；
4. rotate / unlock handle；
5. push / swing door；
6. hold door open；
7. body traversal；
8. release / retract。

核心表述：

> 本文将机器人开门任务从单纯的 door opening 扩展为 door traversal，即机器人不仅需要打开门，还需要在保持门可通过的同时完成自身穿越。

---

## 6. Whole-body control 的表述

因为当前方案不打算训练端到端 RL whole-body controller，而是准备分开控制四足和机械臂，所以不建议直接称为 **whole-body control**。

推荐表述：

```text
hierarchical base-arm coordination
```

或中文：

```text
分层式全身协同移动操作
```

更准确的说法：

> 本文不追求端到端全身力矩控制，而是采用分层式协同控制：底层分别使用成熟的四足运动控制器和机械臂任务空间控制器，高层策略统一生成机器狗底盘速度和机械臂末端残差动作，从而实现任务层面的 base-arm coordination。

动作空间建议定义为：

```text
a_t = [v_x, ω_yaw, ΔT_EE, g]
```

其中：

- `v_x`：机器狗前向速度；
- `ω_yaw`：机器狗偏航速度；
- `ΔT_EE`：机械臂末端残差动作；
- `g`：夹爪开合指令。

---

## 7. 结构事件关键帧

建议不使用通用运动关键帧，而是使用 door traversal 任务中的结构事件关键帧。

### 7.1 关键帧来源

关键帧由以下信号定义：

- 把手角度；
- 门铰链角度；
- EE-handle 接触状态；
- 夹爪状态；
- 机器人相对门洞的位置；
- 机械臂关节状态；
- base 与门 / 门框的相对位姿。

### 7.2 推荐关键帧

| Keyframe | 含义 |
|---|---|
| K0 | 初始 / ready |
| K1 | base 到达可操作位置 |
| K2 | EE 到达把手前方 pre-grasp |
| K3 | 接触把手 |
| K4 | 夹爪闭合 / 抓住把手 |
| K5 | 把手开始旋转 |
| K6 | 把手达到解锁角 |
| K7 | 门开始运动 |
| K8 | 门达到可通过角度 |
| K9 | 机械臂保持门打开 |
| K10 | 机器狗身体开始通过 |
| K11 | 机器狗完全通过门洞 |
| K12 | 释放 / 撤离 |

### 7.3 与已有关键帧方法的区别

已有关键帧方法包括：

- PerACT：关节速度接近零 + gripper 状态；
- FrameSkip：动作变化、视觉变化、任务进度；
- KEMO：事件显著性 + 视觉去重；
- Keyframe-Chaining VLA：语义记忆关键帧。

你的区别：

> 本文关键帧不是从通用运动显著性或人类手部轨迹中提取，而是由门的 articulated structure 和机器人门穿越阶段共同定义，直接对应物理因果事件。

---

## 8. residual Keyframe-ACT

### 8.1 原始 ACT 的不足

原版 ACT 直接从观测预测 action chunk：

```text
Depth / State → Action Chunk
```

但在门穿越任务中存在问题：

- 长时序；
- 接触阶段关键；
- 普通 imitation learning 容易平均化；
- 接触失败会传导到后续阶段；
- 纯连续动作学习难以关注结构事件。

### 8.2 建议的 residual Keyframe-ACT

不是从零预测动作，而是在结构轨迹基础上预测 residual：

```text
a_t = a_t^planner + Δa_t^ACT
```

其中：

- `a_t^planner`：由 handle/hinge-aware trajectory generator 生成的 nominal action；
- `Δa_t^ACT`：由 ACT 根据 depth/state 预测的 residual correction。

### 8.3 模型输入输出

输入：

```text
Depth + robot state
```

可选状态包括：

- base pose / velocity；
- arm joint state；
- EE pose；
- gripper state；
- door / handle state estimate；
- 当前 phase / keyframe index。

输出：

```text
[v_x, ω_yaw, EE residual, gripper]
```

### 8.4 训练重点

在结构事件关键帧附近提高监督权重：

```text
L = Σ_t w_t · ||a_t - â_t||
```

并可加入：

- phase prediction head；
- next keyframe prediction head；
- residual action penalty；
- contact consistency loss；
- door angle progress loss。

---

## 9. 最终推荐标题

最推荐标题：

```text
DoorTraverse-ACT: Structure-Guided Residual Keyframe Imitation for Quadruped Mobile Manipulation
```

中文：

```text
DoorTraverse-ACT：面向四足移动操作的结构引导残差关键帧模仿学习
```

其他可选标题：

1. 从单张图像到四足机器人门穿越：基于可交互门数字孪生的结构引导模仿学习
2. Image-to-Door-Traversal: Structure-Guided Residual Imitation Learning for Legged Mobile Manipulators
3. Articulation-Aware Door Traversal with Quadruped Mobile Manipulators via Residual Keyframe-ACT

---

## 10. 摘要核心内容

摘要建议包含以下逻辑：

1. 四足移动操作机器人门穿越是进入真实环境的重要能力；
2. 该任务不同于桌面机械臂，因为需要 base-arm coordination；
3. 现有方法依赖人类视频重定向、大规模 RL 或手工状态机；
4. 本文提出 DoorTraverse-ACT；
5. 用改进 ArtiCraft + GPT-5.5 agent 从单张图生成可交互门 digital twin；
6. 用门结构生成 nominal trajectory 和 structure-event keyframes；
7. 用 residual Keyframe-ACT 根据 depth/state 学习闭环修正；
8. 输出机器狗速度、机械臂 EE residual 和夹爪指令；
9. 仿真和真实实验验证。

---

## 11. 最终浓缩版 Contributions

本文的主要贡献如下：

### Contribution 1

**提出一种面向四足机器人门穿越的结构感知 real-to-sim-to-real 框架。**  
本文基于改进的 ArtiCraft 和 GPT-5.5 agent，从单张门图像生成可交互门数字孪生，并将门板、把手、铰链轴、把手旋转方向和门开合约束转化为机器人操作先验。在控制层面，本文采用分层式四足移动操作框架：底层分别使用稳定的四足运动控制器和机械臂任务空间控制器，高层统一协调移动基座与机械臂，从而兼顾运动稳定性、末端精度和长时序门穿越能力。

### Contribution 2

**提出结构事件关键帧驱动的 residual Keyframe-ACT。**  
不同于通用运动启发式关键帧或从人类视频中提取的手部关键帧，本文根据把手角度、门铰链角度、接触状态、夹爪状态以及机器人相对门洞位置定义门穿越任务中的结构事件关键帧。基于这些关键帧，本文进一步提出 residual Keyframe-ACT，在结构轨迹生成器提供的 nominal action 基础上预测残差动作，并同时输出机器狗前向速度、偏航速度、机械臂末端残差动作和夹爪指令，实现深度观测下的闭环 base-arm coordination。

### Contribution 3

**系统验证所提方法在仿真和真实四足移动操作平台上的门穿越能力。**  
本文在不同门几何、把手位置、开合方向、初始机器人位姿和感知扰动条件下评估方法性能，并与纯结构规划、原始 ACT、无关键帧残差 ACT、仅机械臂控制以及学习式门操作方法进行对比，验证结构先验、结构事件关键帧和分层式 base-arm 协同对门穿越任务的有效性。

---

## 12. RAL 创新性评估

### 12.1 当前方案是否有 RAL 潜力

评价：

```text
有 RAL 潜力，但需要故事收紧、实验扎实。
```

原因：

- 不只是改 ACT；
- 不只是生成门资产；
- 不只是让机器狗开门；
- 而是将结构先验、结构关键帧、残差模仿学习和四足移动操作结合成完整系统。

### 12.2 风险点

| 风险 | 应对方式 |
|---|---|
| ArtiCraft 不是原创 | 表述为“基于并改进 ArtiCraft”，核心贡献放在机器人任务化解释和轨迹 / policy |
| Keyframe-ACT 容易被认为只是加权 | 强调结构事件关键帧，而不是通用 keyframe |
| 分开控制狗和臂不应叫 WBC | 使用 hierarchical base-arm coordination |
| 真实实验太少会弱 | 至少主门多次 trial + 多门 pipeline 复用 |

---

## 13. 实验设计建议

### 13.1 任务范围

可以只做 **推门穿越**，但论文要明确范围：

```text
push-side hinged door traversal
```

不建议声称：

```text
universal push-pull door traversal
```

### 13.2 是否必须做 door0 训练、door1 测试？

不一定。

因为本文是 real-to-sim-to-real，更合理的定位是：

```text
instance-adaptive real-to-sim-to-real
```

也就是：

```text
给定目标门的一张图像
→ 为该目标门生成 digital twin
→ 在该门仿真中生成数据 / 训练策略
→ 部署到对应真实门
```

因此不需要强制做：

```text
train door0 → test door1
```

但推荐做：

```text
door0: image → sim → train → real test
door1: image → sim → train/adapt → real test
door2: image → sim → train/adapt → real test
```

这证明的是：

```text
pipeline generalization
```

而不是：

```text
single-policy generalization
```

### 13.3 推荐实验规模

#### 真实实验

最低建议：

```text
1 个主门配置 × 20–30 次 trial
+ 2–3 个额外门配置 × 每个若干 trial
```

更理想：

```text
3–5 个真实门配置 × 每个 10 次 trial
```

配置变化包括：

- 左 / 右铰链；
- 不同把手高度；
- 不同门宽；
- 不同门阻尼；
- 不同初始机器狗位置；
- 不同初始 yaw；
- depth 噪声 / 遮挡。

#### 仿真实验

建议：

```text
20–50 个 generated door variants
200–500 次测试 trial
```

仿真变化：

- 门宽；
- 门高；
- 把手高度；
- 把手长度；
- 铰链方向；
- 门质量；
- 阻尼；
- friction；
- latch resistance；
- 机器人初始 pose；
- depth noise；
- digital twin 几何误差。

---

## 14. 推荐 Baselines / Ablations

### 14.1 必做 baseline

| Baseline | 目的 |
|---|---|
| 纯结构轨迹 planner | 证明 open-loop 结构轨迹不够鲁棒 |
| 原版 ACT | 证明普通 imitation learning 不如 residual Keyframe-ACT |
| Residual ACT without keyframes | 证明 residual 有用，但关键帧进一步提升 |
| Keyframe-ACT with generic keyframes | 证明通用运动关键帧不如结构事件关键帧 |
| Arm-only control | 证明 base-arm coordination 对 traversal 必要 |

### 14.2 可选 baseline

| Baseline | 目的 |
|---|---|
| base scripted + arm learned | 看 base 是否需要学习 |
| base learned + arm scripted | 看 arm residual 是否需要学习 |
| no door prior policy | 证明结构先验重要 |
| imitation without residual | 证明 residual formulation 更稳 |

---

## 15. 推荐指标

不要只报告 success rate。建议分阶段报告。

### 总体指标

- Door Traversal Success Rate；
- Task Completion Time；
- Collision Rate；
- Sim-to-real Success Rate。

### 阶段指标

- 抓把手成功率；
- 把手旋转成功率；
- 门达到可通过角度成功率；
- 保持门打开成功率；
- 机器狗穿门成功率；
- 机械臂接触丢失次数；
- 夹爪滑脱次数。

### 轨迹质量指标

- 最大门打开角度；
- EE tracking error；
- base trajectory deviation；
- 机械臂关节是否接近极限；
- 通过门洞时最小安全间隙；
- residual action magnitude。

---

## 16. 论文中推荐使用的关键表述

### 16.1 与 video-to-robot 区别

> 与从人类视频中提取手部轨迹的 real-to-sim-to-real 方法不同，本文不依赖人手运动重定向，而是从单张图像生成带有把手、铰链和运动约束的可交互门数字孪生，并直接从门的 articulated structure 合成门穿越轨迹。

### 16.2 与 legged door RL 区别

> 与现有 legged mobile manipulation 开门方法不同，本文不依赖大规模端到端 RL 或 monolithic whole-body controller，而是采用结构先验驱动的 residual imitation learning：底层使用稳定的四足运动和机械臂控制器，高层策略统一预测机器狗速度与机械臂末端残差，实现任务级 base-arm coordination。

### 16.3 对实验范围的限定

> 本文关注 push-side hinged door traversal。该设置覆盖了四足移动操作门穿越中的关键挑战，包括把手解锁、门板推开、移动基座与机械臂协同、保持门开启以及机器人整体通过门洞。尽管本文不直接处理 pull-side traversal，但所提出的结构事件关键帧和 residual imitation learning 框架可自然扩展到拉门场景。

### 16.4 对 real-to-sim-to-real 设定的限定

> 本文关注 instance-adaptive real-to-sim-to-real door traversal。给定目标门的一张图像，系统自动生成该门的可交互数字孪生，并在对应仿真中生成结构轨迹和训练数据，从而获得适配该门的四足移动操作策略。本文进一步验证该流程可复用于多种门配置，并在目标门上对初始位姿、感知噪声和物理参数扰动具有鲁棒性。

---

## 17. 最终论文故事总结

一句话版本：

```text
本文提出一种结构先验驱动的四足机器人门穿越 real-to-sim-to-real 框架：从单张图像生成可交互门数字孪生，利用门的把手和铰链结构定义结构事件关键帧并合成 nominal traversal trajectory，再通过 residual Keyframe-ACT 学习 depth-based closed-loop correction，实现四足 base 和机械臂 EE 的任务级协同。
```

更短版本：

```text
不是从视频模仿人怎么开门，
也不是用大规模 RL 盲目探索怎么穿门，
而是从单张图像恢复门的结构，
用结构生成可解释轨迹，
再学习闭环残差完成四足门穿越。
```
