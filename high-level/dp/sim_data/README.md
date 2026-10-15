# PhysX → MeisterRender → LeRobot 批量数据管线

本管线为 WC 拉门任务的第一版工程实现，借鉴 SIM1 的物理/渲染解耦方式；不是完整复现 SIM1，也没有实现其扩散轨迹生成器。运行机器和资产依赖由配置管理，旧数据、旧训练不覆盖，不上传 Hugging Face。

## 分层

1. **录制**：`tests/run_wc_contact_physics.py --record-actions`，在 PhysX 新运行中保存关节实际值、实际下发关节目标、规定底盘位移指令、门状态、各刚体与相机位姿。共享机器人 URDF 不再原地修改；改写的碰撞 URDF 单独存入每条原始数据目录。
2. **场景**：`build_scene.py` 调用 WC 资产适配器生成打包的 `.blend`，验证所有采样帧的刚体位姿。配置控制白墙、白顶、杂物随机种子、收纳箱/纸箱颜色、两侧摆放和灯光能量。对全 2601 个记录帧的机器人及门运动范围做保守 AABB 间距检查。
3. **渲染**：`render_meister.py` 使用官方 MeisterRender `Pipeline` 的 BlendLoader、processor、BlenderPathtracingRenderer 与 compositor。不是把原 bpy 循环重命名为 MeisterRender。上游固定在 `c031e1f8594dab3a91832e817d884f55189a4e8a`，未修改上游源代码。同步调用使错误可以传播，避免上游队列接口在渲染子进程退出后无限等待。覆盖其初始化设置以明确控制采样数、设备和色彩管理；不沿用其硬编码 64 samples 或隐式预热帧。
4. **批处理/导出**：`pipeline.py` 管理 episode、GPU 分配、锁、失败记录、重试、状态和超时；`export_lerobot.py` 调用正式 `LeRobotDataset` API 生成 **LeRobot v3.0**，重新加载核对数值和图像，再原子发布数据目录。

## 运行

配置中的路径为当前 ps1 环境。默认只使用 GPU 3，不动其他训练进程。改变 GPU 列表可让不同 worker 处理不同 episode，每张 GPU 同时一个任务；无需让所有 GPU 参与同一帧渲染。

```bash
python pipeline.py --config configs/smoke.json
python pipeline.py --config configs/smoke.json --stage status
python pipeline.py --config configs/smoke.json --retry-failed
```

`--stage episodes` 只做录制、场景、渲染；`--stage export` 仅导出已有完整 episodes。默认 `all` 做全部步骤。进程异常退出后保留日志与状态；显式使用 `--retry-failed` 再尝试已失败任务。每次调用每阶段最多尝试 `max_attempts` 次，不会无限重试。

`configs/production_500.json` 是 **500 条、640×480、25 fps、32 samples** 的生产模板；本次不会自动启动它。首次建议复制模板，将 episodes 改为 2–5，验证不同起始偏移与随机场景，再扩规模。运行配置/代码指纹改变会拒绝复用旧 run root；改配置时使用新目录，以免混用帧缓存或动作定义。

独立 MeisterRender 环境使用 Python 3.11、官方 requirements 的 `bpy==4.5.0`；不修改原 Isaac Gym 或 LeRobot 环境。依赖与上游库放在结果工作区的 `runtime/` 和 `SIM1MeisterRender/`。源文件来自仓库 `high-level/dp/sim_data/`，WC 适配器来自 `high-level/dp/tests/`。

## 动作与同步约定（必须阅读）

`action` 共 10 维：

`[base_vx, base_vy, base_vyaw, joint1, joint2, joint3, joint4, joint5, joint6, jointGripper]`

- 关节值是 **实际下发的位置目标，单位 rad**，不是下一帧测量关节位置，也不是关节增量。
- 底盘三维是规定底盘目标位姿增量除以 20 ms 得到的前一底盘指令坐标系速度，单位 m/s、m/s、rad/s；额外保留原始底盘位姿目标。不是轮速或力矩控制。
- `observation.state` 是当前快照记录的上一步底盘命令 3 维和实际测量关节 7 维。
- 原始快照为 **POST-step**。导出第 i 帧观察时，动作取第 i+1 个物理步实际执行的指令，避免将已经执行完的动作错配到观察之后。末尾无后继动作的快照不作为训练帧。
- 原始控制 50 Hz。smoke 为 10 fps，生产模板为 25 fps；动作标签是采样时刻接下来的 20 ms 指令，不声称它在整个 RGB 间隔保持不变。全部 50 Hz 指令保存在 raw JSON。需要严格等控制频率的行为克隆时，将 fps 配为 50，或另外定义带子步序列的动作接口，不要直接把 10 Hz 标签视为 100 ms 的保持动作。
- 这是 **Joint10** 数据，旧 Joint9 模型不能直接加载使用。训练和部署必须明确选择相同维度、顺序、绝对关节目标及底盘语义。

`observation.images.front/wrist/overview` 为分开存储的 RGB 视频，不把拼接图喂给策略。另有 `observation.camera_c2w.*`（行主序 4×4，Blender +X右/+Y上/−Z朝前）、`observation.camera_K.*`、`observation.door_state`、`source_step` 和 `next.done`。

## 输出与可靠性

每个 episode 保留 `status.json`、分阶段日志、原始仿真、场景和 `render/rgb/`。每组同步 RGB 完整写入后才原子发布帧标记；重启时核对场景/配置签名和 PNG 哈希，跳过已完成帧，重算损坏或不完整的帧。

完整运行包含 `manifest.json` 与 `dataset/`。LeRobot 导出先写入唯一 `.partial.*` 目录；只有 API 读回、全部 state/action 对比、视频全帧解码通过后才发布 `dataset/`。失败的临时导出保留供排查，重新导出使用新的临时目录，不把残缺目录当作成品。`pipeline_provenance.json` 和 `file_sha256.json` 记录来源和验收结果。

物理质量门：完整轨迹、最终收臂阶段、开门至少 25°、越过门平面至少 0.5 m、动作与实际发送目标一致。该 WC 门平面判定目前专用于 world −X 穿门方向，换门方向必须增加新的资产/任务适配器。

## 边界

- 手臂/门有 PhysX 动力学，底盘仍为规定运动，不是全身轮地动力学。
- 墙、天花板、杂物仅参与渲染；距离检查针对已记录轨迹，不是新物理接触仿真或连续碰撞检测。
- 颜色/位置/灯光随机化范围有限，白墙白顶保持不变；没有自动构建真实材质库。
- 批量行为多样性目前来自有限的起始横向偏移和仿真 seed，不等价于 500 种独立技能或 SIM1 的生成模型。
- 低分辨率 smoke 只验证管线，不能据此报告策略成功率或训练效果。未启动策略训练。

本地轻量测试：`python -m unittest discover -s high-level/dp/sim_data -p test_pipeline.py -v`。

## 已完成验收（2026-09-20）

- `configs/smoke.json` 全流程通过，远端输出 `wc_meister_pipeline_20260920/smoke_v3/dataset/`；同名结果目录已下载到本地 `high-level/dp/result/` 下。
- 新物理记录 2601 帧，开门 80.476°，最终越过门平面 1.308 m。带动作标签，不复用旧的无动作视频数据冒充训练样本。
- LeRobot v3 数据集 1 个 episode、520 帧、52 秒；三路 160×120、10 fps RGB，4 samples，是低分辨率管线验收样例。
- 远端正式 LeRobot API 重新读取，520 行 state/action 全部与原始命令对比一致；三路视频全部 1560 帧解码通过。
- 本地下载后 9 个数据文件 SHA256 一致；本地 LeRobot v3 API 成功读出第 0、260、519 帧，三路图像与 10 维状态/动作形状正确。
- 受控 SIGTERM 中断首轮渲染后，自动重试从第 110 帧继续，前 109 帧复用；第一张 PNG 的修改时间未改变。`render_001.log` 的被终止状态是这次测试造成的，不是最终未处理错误。
- 全流程再次运行直接命中成品，输出 `DATASET_ALREADY_VERIFIED`，没有重新仿真或渲染。
- 6 项轻量单元测试通过。500 条生产配置仅提供模板，未启动 500 条生成或策略训练。

## 5 条 RGB 增强试采（2026-09-21）

`configs/rgb_aug5_no_material.json` 是独立试采配置，不覆盖 smoke，也不启动训练：

- 每条重新运行 2601 步 PhysX，固定起点和控制器、不同种子；目的是测试视觉多样性，不声称有 5 种不同动作技能。
- 3 视角 640×480、10 fps、16 samples/denoise。每条 520 组 RGB，共 2600 组 / 7800 张图。GPU 0、1 各一个 worker；目录 `wc_meister_rgb_aug5_20260921/run/`。
- `scene.randomize=false` 禁用原来的箱子颜色/摆放随机化。独立 `scene.lighting` 仅改变灯光能量（0.8–1.2 倍）、环境光（0.85–1.15 倍）及轻微暖冷光色，不改变物体材质或几何。
- 直接复用 `float_ik/rgb_doorman.py` 的光度函数，参考 stable 配置。亮度/对比度每条必采样 0.9–1.1 倍（原 stable 的概率为 0.25，此处为 1，以便小样本能覆盖）；饱和度、色相、噪声和模糊沿用温和范围。不是完整复现 DoorMan 的光照/HDR 系统，也不把 Gym 的强度数值直接当成 Blender 瓦数。
- 光照和光度参数整条固定、三个视角共享；噪声按帧/视角确定性采样，断点重算可复现。没有图像裁剪、翻转或几何变换，相机标定不变。
- 每条 `scene_*/lighting_augmentation.json` 审计物体材质的节点、参数、纹理身份；灯光修改前后摘要必须相同，导出还检查五条之间材质摘要一致。
- 每条 `render/augmentation.json` 和 LeRobot 的 `pipeline_provenance.json` 保留实际采样参数/实现哈希。`render/clean_preview/` 保存第 0、240、490 帧的光度增强前图（已包含该条随机光照），可与同编号 `render/rgb/` 比较；不是完全无增强的光照基准图。
- 验证命令：`python -m unittest discover -s high-level/dp/sim_data -p 'test_*.py'`。当前 9 项通过。完整数据是否完成以各条状态和 `dataset/pipeline_provenance.json` 为准。

本次 5 条已完成采集/渲染/LeRobot 导出和本地读回，共 2600 行标签，远端全部 7800 帧视频解码通过。本地核对 9 个数据文件 SHA256、全部 2600 行状态/动作，以及每条起/中/末三组图像。结果在 `result/wc_meister_rgb_aug5_20260921/run/`。

试采发现两点限制：五次确定性采集的原始轨迹 SHA256 完全相同，因此是 **1 种动作的 5 组视觉变体**；episode 001/002 的部分视角出现高亮裁剪，`review/quality_report.json` 已记录警告。数据格式/标签检查通过，不等于图像质量完全通过。白色场景上直接增加 sRGB 亮度会损失高光细节，正式扩规模前应收紧增强范围或引入曝光/高光保护，并增加真实轨迹多样性；本次未启动训练。

## 低亮度 / 室内顶光（2026-09-22）

用户后续指定先只交付一条、用天花板向下照的室内顶光。`rgb_aug5_dim.json` 的五条重渲染已被停止，不自动恢复；当前配置为 `configs/rgb_ceiling1.json`：

- 只保留 `Room / entrance ceiling light` 和 `Room / back room ceiling light`，z=3.24 m，朝本地 −Z（世界向下），直径 1.2 m 的面光，Blender 能量分别为 250、150。此数值是渲染器功率，不对应家用灯泡标称电功率。
- 摄影棚主光、正面/侧面补光能量设为 0，World Background 强度为 0；室内填充光由 Cycles 的光线反弹产生。未添加发光物体或改材质。外部视角墙边露出的场景外区域因此可能为黑色。
- 固定 RGB 亮度乘以 0.97，contrast=1，关闭色相/饱和度/噪声/模糊变化，先确认纯顶光外观。不是新的一批增强随机样本。
- `physics.reuse_run_root` 显式复用原有动作/状态，核对源状态/轨迹哈希和种子后复制原始文件到新目录，记录 `reuse_provenance.json`；不会把新 `.blend` 写入旧数据目录，也不声称是新物理采集。
- `pipeline.py --stage prepare` 只准备原始数据和烘焙场景；`render_meister.py --preview-frames 0,240,490` 只输出预览完成标志，不能伪装成全片完成。全片渲染可验证并复用这些帧。
- `preview_wc_dim_lighting.py` 对三个时刻的三视角进行前后亮度/过曝检查；`audit_wc_dim_lighting.py` 比较全片原始数据、材质、相机、每帧亮度及高光裁剪，`review_wc_rgb_aug5.py` 支持按配置导出一条或五条拼接视频。
- 12 项单元测试通过，新增顶光模式会关闭其他灯/环境光的检查。

### 用户更正：远距离法线方向点光

最终灯位按门板平面法线设置（不是左右）：`(0,-4,1.6)`、`(0,3.5,1.6)`，门前/后分别约 4 m 和 3.5 m；各 12 单位能量、0.08 m 半径、其他灯和环境光全关。`rgb_point_dark1.json` 已更新为此配置，`run_normal` 完整数据管线尚未启动。

为了快速确认效果，`tests/preview_wc_normal_points.py` 直接读取已烘焙场景，只改光照，输出 `result/wc_meister_point_dark1_20260922/normal_preview/` 的 480×360、4 samples 快速预览；`--video` 以 2 FPS 覆盖全部 52 秒动作，不是精渲染或新的 LeRobot 数据。此快速预览直接调用 Blender，不冒称走了 MeisterRender 导出。此前 `run`/`run_v2` 的左右灯位测试不再继续。13 项单元测试包括灯光类型、能量、法线两侧位置、关闭旧光源和材质不变。

### 完整视频：小幅提亮（2026-09-24）

`rgb_point_final1.json` 保持上述法线方向灯位、光色和半径，只把两灯能量从 12 提到 15。`tests/render_wc_single_parallel.py` 在 GPU 0/3 分别渲染偶数/奇数帧，随后标准 pipeline 全量核对缓存并导出。已完成 520 组、52 秒、10 FPS、640×480、16 samples 的三视角视频及 LeRobot 数据，结果 `result/wc_meister_point_final1_20260924/run/`。

`compare_wc_point_brightness.py` 比对上一版 104 组采样图，确认灯位/材质不变、能量提高 25%、画面均值小幅上升。本地数值/视频读回通过，14 项单元测试通过。完整光度审计仍记录 17 张图的深阴影/遮挡超过 5% 纯黑阈值；报告 `passed=false` 并保留明细，不掩盖警告或擅自移动相机/提高更多光照。

### 点光源能量 30（2026-10-07 请求）

`configs/rgb_point30_1.json` 将两盏固定灯能量由 15 改为 30，其余视觉配置保持不变，复用前版轨迹。GPU 0/1/3 分帧渲染完整 520 组、52 秒、10 FPS、640×480 三视角，并导出 LeRobot v3。成品在 `result/wc_meister_point30_20261007/run/`。

`tests/audit_wc_point30.py` 对比前版全部 520 组帧，验证灯位、光色、半径、材质、相机和原始记录不变，只增加功率；全片通过高光/纯黑比例阈值检查。15 项单元测试通过，新增测试确保视觉配置只改两盏灯的能量。完整视频和参数对比图已下载本地。
