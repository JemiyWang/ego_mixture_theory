# EgoBridge 双域 Push-T 可复现性尽职调查报告

- 调查日期：2026-10-22
- 调查对象：EgoBridge（arXiv:2509.19626，NeurIPS 2025 录用）仿真 Push-T 双域实验；底座 Diffusion Policy Push-T（real-stanford/diffusion_policy）
- 标记约定：**[已核实]** = 有链接或命令输出佐证；**[推断]** = 基于证据的估计/判断；**[未知]** = 公开材料未给出

---

## 0. TL;DR

- **EgoBridge 未开源**（截至 2026-10-22）：项目页无代码链接，GitHub 无官方仓库，NeurIPS 检查单明确写 "No. But we plan to release after acceptance"，至今未兑现。
- **但仿真部分论文披露足够自复刻**：NeurIPS 附录 G.2/H 给出了双域 Push-T 的全部关键构造参数（4 场景 350 条演示、摩擦系数 ×0.7、评估协议 100 种子、表 5 训练超参）。
- **底座 Diffusion Policy Push-T 完全开源（MIT）且数据可下载**，官方 206 条鼠标遥操作演示恰好就是"圆形推动器/白底/标准 T"域，可直接复用。
- **结论：Go**。推荐路线 (a)：fork `real-stanford/diffusion_policy`，自行做双域环境改造 + 自写协同训练 workspace。预计搭建周期 **约 1 周**，网格实验（50–100 配置 × 3 seeds）在 8×RTX4090 上 **约 2–4 天满负荷机时**。

---

## 1. EgoBridge 开源情况

### 1.1 论文与版本

- arXiv:2509.19626，仅 v1（2025-09-23 提交，无更新）[已核实，[arxiv.org/abs/2509.19626](https://arxiv.org/abs/2509.19626)]
- NeurIPS 2025 正式录用，camera-ready 版附录比 arXiv v1 **多出关键仿真细节**（附录 G.2 数据收集、附录 H 完整结果表 7）[已核实，[NeurIPS 论文页](https://papers.neurips.cc/paper_files/paper/2025/hash/434756a76c44e220d5f32d6e0b488c0c-Abstract-Conference.html)，[PDF](https://papers.neurips.cc/paper_files/paper/2025/file/434756a76c44e220d5f32d6e0b488c0c-Paper-Conference.pdf)]
- 另获 CoRL 2025 H2R Workshop Oral [已核实，[项目页](https://ego-bridge.github.io/)]

### 1.2 代码是否开源：**否** [已核实]

证据链：

1. **项目页 HTML 源码无代码链接**：`curl -sL https://ego-bridge.github.io/ | grep href` 全部外链只有 arXiv、作者主页、模板致谢等，无任何 GitHub 仓库链接。
2. **GitHub 搜索无官方仓库**（命令输出，2026-10-22）：
   - `api.github.com/search/repositories?q=egobridge` → 仅 2 个无关仓库（NMT 域适配置致敬项目）。
   - `q=ego-bridge` → 29 个全部无关。
   - 第一作者账号 [rpunamiya](https://github.com/rpunamiya) 仓库列表无 EgoBridge；合作者 [simarkareer](https://github.com/simarkareer)、[lawrencez22](https://github.com/lawrencez22) 同样没有。
   - HuggingFace models/datasets 搜索 "egobridge" 均返回空 `[]`。
3. **NeurIPS 检查单原文**（camera-ready 第 28–29 页）：
   - "Does the paper provide open access to the data and code...?" → **Answer: [No]. "No. But we plan to release after acceptance."**
   - "Are new assets introduced in the paper well documented...?" → "[NA] This work does not release new assets."
   - 即：训练代码、双域 Push-T 环境、真机数据**均未发布**；license 无从谈起。NeurIPS 2025 开会（2025-12）至今已约 10 个月，承诺的 release 未兑现。

### 1.3 论文对双域 Push-T 的披露程度（够不够自复刻）

**结论：基本够。** 以下内容全部来自论文原文 [已核实]：

仿真设置（§5.1 + 附录 B.2/G.2/H）：

| 项目 | 论文披露内容 |
|---|---|
| 底座 | "the PyMunk simulator environment published with Diffusion Policy [6]"（附录 G.2 原话），数据处理"follow the PushT environment setup introduced in the Diffusion Policy benchmark suite"（附录 B.2） |
| 机器人域 | salmon 色**三角形推动器**、白底、标准 T 构型、**摩擦系数 ×0.7**（"friction coefficient multiplier of 0.7 on the triangle pusher to embody a kinematic gap"） |
| 人类域 | 蓝色**圆形推动器**（= 底座默认 agent）、紫底、T 构型**镜像**（"mirrored, requiring a new motion to slot into place"） |
| 演示数据 | 共 **350 条**，4 场景：圆形/白底/标准T 100 条；三角形/白底/标准T 100 条；圆形/紫底/标准T 50 条；圆形/紫底/镜像T 50 条；圆形/白底/镜像T 50 条（附录 G.2） |
| 观测/动作 | RGB 96×96×3（ImageNet 归一化）+ 2D 末端位置 proprioception（min-max 归一化）；动作 = 2D 末端目标位置；**动作块 k=16；观测上下文窗口=1** |
| 策略 | 标准 ResNet-18（截断）+ UNet 的图像版 Diffusion Policy；OT-joint loss 加在 ResNet 编码器特征输出上 |
| 训练 | 单卡 A40，**130,000 迭代 ≈ 2 小时**；AdamW lr=1e-4, wd=1e-6，cosine + 500 warmup，**batch 32**，EMA(power=0.75)，α_OT=0.2，GeomLoss Sinkhorn(blur=0.01)（附录 B.2 表 5） |
| 评估 | 3 设置（标准 / 紫底 / 紫底+镜像T）；**100 个固定种子**，从 [101, 9999] 区间以 seed=42 确定性随机采样；指标 = 平均 max-IoU（mean reward）与 reward≥0.9 的成功率（附录 H） |
| 结果锚点 | 表 7：EgoBridge SR 53/48/39%；Target-only 39/0/0%；Co-train 48/42/31%；Standard OT 38/15/8%；MMD 45/22/14%（三设置顺序：域内 / 紫底 / 紫底+镜像T）——复刻时可作趋势对照 |

**未披露、需自行决定的细节 [未知]**：

- 紫色的精确 RGB 值、salmon 精确色值
- "镜像 T 构型"的精确几何变换（见 §3.3 分析：底座 T 形本身轴对称，故"镜像"应指初始/目标位姿的镜像，论文未给公式）
- 演示收集方式细节（推测为鼠标遥操作，底座自带工具，见 §2.3）
- 扩散推理步数（底座默认 DDPM 100 训练步 / 推理 100 步，EgoBridge 未说明是否减少）
- 是否使用底座的随机裁剪增广（表 5 的 "Data Augmentations" 只列了 ImageNet 归一化，**[推断]** 未用 crop）
- 用户任务书中提到的 "2–3× 时序播放"：经全文检索，**论文仿真部分未做播放速度扰动**；2–3× 出现在 §3.2，是 DTW 的动机（"humans might be 2-3 times faster than teleoperated robots"），描述的是真机人机速度差。仿真域差异 = 外观（推动器形状/颜色、背景色）+ 动力学（摩擦 0.7）+ 运动分布（镜像 T）三项 [已核实]

---

## 2. Diffusion Policy Push-T 底座

### 2.1 仓库状态 [已核实]

- [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)，**MIT license**，4676 stars，默认分支 main，**最后提交 2024-12-24**（维护已停滞，但代码稳定、被广泛复现）
- Push-T 环境为仓库内置实现（`diffusion_policy/env/pusht/`），改编自 [google-research/ibc](https://github.com/google-research/ibc)（README 致谢节）

### 2.2 安装依赖与版本坑 [已核实 + 推断]

官方 `conda_environment.yaml`（[源文件](https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/conda_environment.yaml)）关键钉版：

```
python=3.9, cudatoolkit=11.6, pytorch=1.12.1, torchvision=0.13.1
gym=0.21.0, pymunk=6.2.1, pygame==2.1.2, zarr=2.12.0, numcodecs=0.10.2
numpy=1.23.3, numba==0.56.4, hydra-core=1.2.0, diffusers=0.11.1, ray[default,tune]==2.2.0
```

版本坑逐项：

1. **RTX 4090（sm_89）与 torch 1.12.1+cu116 不兼容** [已核实为通用问题]。sm_89 需要 CUDA ≥11.8 的 PyTorch 构建；cu116 轮子不含 sm_89 内核，会报 "NVIDIA GeForce RTX 4090 with CUDA capability sm_89 is not compatible with the current PyTorch installation"（同类报错见 [Track-Anything#101](https://github.com/gaomingqi/Track-Anything/issues/101) 等）。**对策 [推断/社区通行做法]**：保持 gym 0.21 / pymunk 6.2.1 不动，仅升级 torch 到 1.13.1+cu117 或 2.x+cu118/121（DP 代码为纯 PyTorch，无定制 CUDA 算子，升级通常无代码改动；社区已有大量在新 torch 上跑通的实践，如 [Robium 博客](https://robium.ai/blog/diffusion-policy-pusht/) 在 Mac/现代环境运行）。
2. **pymunk 必须保持 6.2.1**：环境源码使用 `space.add_collision_handler(0, 0)`（pymunk 6 API；pymunk 7 已改名 `on_collision`），升级 pymunk 会报错 [已核实，源码 pusht_env.py:310 附近]。
3. **gym==0.21.0**：旧版 gym API（step 返回 4 元组）。用 conda 安装可绕开 pip 下 gym 0.21 的构建问题 [推断]。
4. Python 3.9 已 EOL，但 conda-forge 仍可用；mujoco 相关 apt 依赖（`libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`）仅 robosuite/kitchen 需要，**纯 Push-T 可以不用** [推断]。
5. numba 0.56.4 与 numpy≥1.24 不兼容，官方钉版组合（numpy 1.23.3）是自洽的，不要乱升级 [已核实钉版]。

### 2.3 Push-T 环境形式 [已核实，源码 [pusht_env.py](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/pusht/pusht_env.py)]

| 项 | 值 | 源码位置 |
|---|---|---|
| 物理 | pymunk 2D，无重力，sim 100 Hz，控制 10 Hz（每动作 10 物理步），PD 控制 k_p=100, k_v=20 | L43–46, L120 |
| 观测（state） | 5 维：agent (x,y) + block (x,y,θ)，范围 0–512 / 0–2π | L51–57 |
| 观测（image） | 512×512 画布 resize 到 **96×96×3** RGB | L36, `pusht_image_env.py` |
| 动作 | 2 维 agent **绝对目标位置**，Box([0,0],[512,512]) | L59–65 |
| 奖励 | coverage = block 与 goal 位姿的 IoU；reward = clip(coverage/0.95, 0, 1)；coverage>0.95 判 done | L128–133, L316 |
| agent | **圆形，半径 15，RoyalBlue 色，kinematic body，friction=1** | L305, L324–328 |
| block | T 形（`add_tee`，两个 Poly），LightSlateGray，friction=1 | L305–306, L340–366 |
| goal | 固定位姿 (256, 256, π/4)，LightGreen 半透明绘制 | L308 |
| 背景 | 纯白 `canvas.fill((255,255,255))` | L192 |
| 初始状态 | seed 决定：agent ∈ [50,450]²，block ∈ [100,400]²，角度均匀 | L96–105 |
| episode 长度 | runner 默认 `max_steps=300`（image 任务配置），即 ≤30s @10Hz | [pusht_image.yaml](https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/diffusion_policy/config/task/pusht_image.yaml) |
| 域随机化接口 | 构造函数已暴露 `block_cog`、`damping` 参数（非默认随机化，但方便改造） | L34, L92–93 |

### 2.4 专家演示数据 [已核实]

- **现成数据集**：官方托管 `https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip`（HTTP 200 验证，**30,988,725 字节 ≈ 31MB**）。解压后为 `pusht/pusht_cchi_v7_replay.zarr`，含 **206 episodes / 25,650 步**，字段 `img(96×96×3), state(5), action(2), keypoint(9,2), n_contacts`（README 内 ReplayBuffer 示例 + 本地下载 `unzip -l` 验证）。LeRobot 镜像 [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht)（MIT）确认 206 eps / 10Hz。
- **数据来源**：**人类鼠标遥操作**采集。仓库自带采集脚本 [`demo_pusht.py`](https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/demo_pusht.py)（docstring："Hover mouse close to the blue circle to start. Push the T block into the green area."）；DP 论文 Table III 将 Push-T 标注为 PH（proficient human teleoperated）数据，200 demos、300 步上限。**没有官方 scripted expert**——要自己生成专家数据就得用该脚本遥操作，或自写启发式规划器 [推断]。
- **训练集惯例**：官方 image 配置 `max_train_episodes: 90`（只用前 90 条训练）[已核实，image_pusht_diffusion_policy_cnn.yaml]。
- 注意：官方 206 条数据 = **圆形推动器/白底/标准 T**，恰好对应 EgoBridge 的"人类域（域内场景）"，可直接复用为双域数据的一部分 [推断]。

### 2.5 训练时长与显存

| 数据点 | 数值 | 性质 |
|---|---|---|
| EgoBridge 仿真训练 | 130k 迭代，bs=32，ResNet18-UNet，**单卡 A40 约 2 小时** | [已核实，论文附录 B.2] |
| DP 官方 image Push-T 配置 | 3050 epochs，bs=64（≈54 万步量级） | [已核实，config] |
| DP 真机 Push-T 基准 | 固定训练 12 小时/方法（硬件未注明） | [已核实，DP 论文附录 C.1.2] |
| LeRobot 复现 | 单 GPU 训练 12 小时未达到原作者成功率；官方 lerobot/diffusion_pusht 为 200k 步、bs=64，175k 步 checkpoint 达 65.4% SR（500  episodes） | [已核实，[lerobot#43](https://github.com/huggingface/lerobot/issues/43)，[模型卡](https://huggingface.co/lerobot/diffusion_pusht)] |
| 社区经验 | "CNN 版 DP 在单张 RTX 4090 上 4–6 小时（100 演示）"（[roboticscenter.ai](https://www.roboticscenter.ai/blog/diffusion-policy-guide)）；"500 演示单任务 2–6 小时 @A100/4090"（[claru.ai](https://claru.ai/guides/how-to-train-a-diffusion-policy)） | [已核实为社区说法] |

**对本项目的推算 [推断]**：

- 采用 EgoBridge 预算（130k iters, bs32）：A40 2h → RTX 4090 约 **1–1.5h/次**（A40 FP32 ≈37 TFLOPS，4090 ≈83 TFLOPS；小模型吞吐受 kernel launch/数据加载限制，保守按 1.5–2× 加速）。
- 单次评估（100 种子 × 300 步，DDPM 100 步推理，AsyncVectorEnv 多进程）：约 **0.3–0.6h**。
- **单次完整实验 ≈ 1.5–2 GPU·h（4090）**。
- **显存**：bs=64、96×96 图像、ResNet18+UNet(512/1024/2048) 的官方配置估计 8–14GB；官方提供在免费 T4(16GB) 上运行的 [Colab (vision)](https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing) 佐证。**RTX 4090 24GB 足够**。

---

## 3. 双域改造技术清单

底座代码层：`diffusion_policy/env/pusht/pusht_env.py`（核心物理+渲染，367 行单文件）、`pusht_image_env.py`（图像观测封装）、`pusht_keypoints_env.py`（demo 采集用）、`diffusion_policy/env_runner/pusht_image_runner.py`（评估）。建议做法：给 `PushTEnv` 增加构造参数（pusher 形状/颜色、背景色、摩擦倍率、镜像开关），按域实例化，**不改训练侧接口**。

| # | 改造项 | 论文要求 | 改动位置 | 工作量 | 难度评估 |
|---|---|---|---|---|---|
| 1 | 三角形推动器（机器人域） | salmon 色三角形替换默认圆形 agent | `pusht_env.py` L305 `self.agent = self.add_circle((256,400),15)` → 新增 `add_triangle()`（pymunk.Poly 三顶点 + KINEMATIC body，参照 L324–338 `add_circle`/`add_box` 模式），颜色 salmon | ~30 行 | **低**。注意 `pusht_keypoints_env.py`/`demo_pusht.py` 共用 `_setup`，改一处即全链路生效；几何变化→接触力学变化即论文要的"动力学 gap" |
| 2 | 紫色背景（人类域） | 白底 → 紫底 | `_render_frame` L192 `canvas.fill((255,255,255))` → 构造参数 `bg_color` | ~3 行 | **极低**。紫色 RGB 论文未给 [未知]，自选（如 (128,0,128)），属复刻自由度 |
| 3 | 镜像 T 构型（人类域） | "T configuration mirrored, requiring a new motion" | **分析 [已核实]**：`add_tee` 顶点（L340–360）显示 T 形关于纵轴**几何对称**，单纯镜像形状无效果；故"镜像"应指**初始/目标构型镜像**（论文亦称 "change in starting configuration"）。实现：`goal_pose` 与 reset 采样做镜像变换（如 x→512−x、θ→π−θ 或 θ→−θ），或镜像 block 初态分布 | ~20 行 | **低–中**。精确变换论文未给 [未知]，需自行定义并保持训练/评估一致 |
| 4 | 摩擦 ×0.7（仅机器人域推动器） | "friction coefficient multiplier of 0.7 on the triangle pusher" | agent body 的 `friction = 1`（L326）→ 参数化 | ~2 行 | **极低** |
| 5 | 2–3× 时序播放 | **论文仿真未使用** [已核实全文检索]；仅 §3.2 作为 DTW 动机（人比遥操作机器人快 2–3×） | 如需模拟：数据层对人域动作序列做 2–3× 降采样+插值，或改环境 `control_hz` | 数据层 ~20 行 | **低**。建议作为可选消融旋钮，而非默认构造 |
| 6 | 评估协议对齐 | 100 种子 ∈[101,9999]、seed=42 确定性采样；3 评估设置；mean max-IoU + SR@0.9 | `pusht_image_runner.py`（`n_test=50`→100，种子列表注入）；奖励阈值 0.9 与底座 done 阈值 0.95 不同——按 max-IoU≥0.9 自算即可 | ~20 行 | **低** |
| 7 | 训练配置对齐 | obs 窗口=1（底座默认 2）、动作块 k=16（底座 horizon=16 一致）、bs=32（底座 64）、130k iters（底座 3050 epochs） | hydra config 覆盖即可，零代码 | 0 | **极低** |

配套工程量（理论实验本身需要）：

- **双域数据集封装**：两个 zarr ReplayBuffer + 域标签 + 等比例采样（EgoBridge "evenly-sampled" co-train）；DP 的 `SequenceSampler` 可复用，需自写拼接 dataset（~100 行）[推断]。
- **协同训练 workspace**：参照 `train_diffusion_unet_hybrid_workspace.py` 加 OT/MMD 辅助损失（EgoBridge 用 GeomLoss Sinkhorn，pip 包 `geomloss` 现成）[推断]。
- **演示采集**：`demo_pusht.py` 鼠标遥操作。350 条 × 平均 10–20s/条 + 重试 ≈ **3–5 人时**；三角形/紫底/镜像变体需先完成改造项 1–4 再采集。可复用官方 206 条顶替"圆形/白底/标准T"域的 100 条 [推断]。

---

## 4. 替代底座评估

| 路线 | 内容 | 优势 | 劣势 |  verdict |
|---|---|---|---|---|
| **(a) fork real-stanford/diffusion_policy** | 在其内置 PushT 环境上做双域改造，自写协同训练 | 与论文底座逐字一致（附录 G.2 明示用该环境）；206 条官方数据即人类域；环境单文件 367 行，改造点全部定位明确（§3）；评估 runner、训练管线、Ray 多 seed 编排现成 | 依赖老（py3.9/torch1.12，4090 需升 torch）；仓库停更 | **推荐** |
| (b) [gym-pusht](https://github.com/huggingface/gym-pusht) + [LeRobot](https://github.com/huggingface/lerobot) | HF 维护的 gymnasium 版 PushT（`obs_type` 支持 state/keypoints/pixels/pixels_agent_pos，96×96，≥95% coverage 判成功）+ LeRobot 的 DP 实现（复现 65.4% vs 64.2% SR） | 现代环境（py3.10+、新 torch）、维护活跃、pip 安装 | 双域改造仍需 fork gym-pusht（改动量与 (a) 相当）；协同训练/双域采样/OT loss 同样要自写；与论文底座存在实现细节差异（渲染、pymunk 版本、种子协议），复刻对标表 7 时多一层不确定 | 备选（若团队已熟悉 LeRobot 栈） |
| (c) 其他 2D 操作环境 | DP 仓库内的 Block Pushing（pymunk，BET/IBC 改编）、robomimic（MuJoCo 3D）等 | — | 没有现成"双域 Push-T"等价物；要么推倒重来，要么偏离论文设定，削弱与 EgoBridge 结果的可比性 | 不推荐 |

**推荐 (a)**。核心理由：实验目的是对 EgoBridge 论文做理论检验，复刻结论需要与论文底座严格对齐；(a) 的环境改造总代码量估计 <150 行，全部改动点已核实到行号。

---

## 5. 工作量与机时估计（8×RTX 4090）

日历时间 [推断]：

| 阶段 | 内容 | 估计 |
|---|---|---|
| 环境搭建 | conda 建环境、torch 升级适配 4090、官方 pusht.zip 跑通 smoke test（数 epoch + eval） | 0.5–1 天 |
| 双域改造 | §3 表 1–6 项 + 渲染截图/奖励正确性单测 | 2–3 天 |
| 专家数据 | 遥操作采集 250–350 条（可复用官方 206 条减负）+ zarr 打包 | 0.5–1 天（3–5 人时采集） |
| 管线对齐 | 双域 dataset/co-train workspace/OT loss；先跑 Target-only 与 Co-train 对照表 7 趋势（Target-only 域内 SR≈39%） | 2–3 天 |
| **搭建小计** | | **约 1 周** |

机时 [推断，基于 §2.5 单次 ≈1.5–2 GPU·h@4090]：

| 项 | 计算 |
|---|---|
| 网格规模 | 50–100 配置 × 3 seeds = **150–300 runs** |
| 总 GPU 时 | 150–300 × (1.5–2h) ≈ **225–600 GPU·h** |
| 8×4090 墙钟 | 满负荷 **约 28–75 小时 ≈ 1.2–3.1 天**；计入排队/失败重跑/中途检查，按 **3–5 天**规划 |
| 对照实验（搭建期） | 5 基线 × 3 seeds ≈ 15 runs ≈ 30 GPU·h ≈ 8×4090 半天 |

风险提示：若误用 DP 官方默认 3050 epochs 预算，单跑将升至 6–12h@4090 [推断]，网格膨胀 4–6 倍——**应按 EgoBridge 表 5 的 130k iters 预算执行**。

---

## 6. Go / No-Go 结论

**Go。** 理由：

1. EgoBridge 虽未开源，但双域 Push-T 的构造参数在 NeurIPS camera-ready 附录中披露完整（350 条数据构成、摩擦 0.7、100 种子协议、表 5 超参、表 7 结果锚点），**自复刻缺口只剩紫色色值、镜像变换定义等次要自由度**。
2. 底座完全开源（MIT）、数据可下载（已验证 31MB pusht.zip 在线）、环境改造点少且已逐行定位。
3. 算力需求温和：单次实验 ~2 GPU·h，全网格 8×4090 约 3–5 天，4090 24GB 显存充足。
4. 唯一需要自研的部分（双域协同训练 + OT/MMD 对齐损失）本就是理论检验实验的核心交付物，不算额外负担。

**推荐路线**：路线 (a)——fork `real-stanford/diffusion_policy` → 双域环境参数化改造（<150 行）→ 复用官方 206 条演示 + 遥操作补齐域变体 → 自写 co-train workspace → 以论文表 7 的 Target-only/Co-train 趋势做 sanity check → 上网格。

---

## 附录 A：关键核实命令与输出摘要

```bash
# EgoBridge 项目页无代码链接
curl -sL https://ego-bridge.github.io/ | grep -oiE 'href="[^"]*"' | sort -u
# → 仅 arXiv/作者主页/字体等，无 github 仓库链接

# GitHub 仓库搜索（2026-10-22）
curl -s "https://api.github.com/search/repositories?q=egobridge"
# → total_count: 2（均为无关 NMT 项目）
# rpunamiya / simarkareer / lawrencez22 仓库列表均无 EgoBridge

# NeurIPS camera-ready 检查单（本地解析 PDF）
# "open access to the data and code" → Answer: [No] "No. But we plan to release after acceptance."

# 官方 Push-T 数据可下载
curl -sI https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
# → HTTP 200, content-length: 30988725
unzip -l pusht.zip  # → pusht/pusht_cchi_v7_replay.zarr（829 文件）

# 底座仓库元数据
curl -s https://api.github.com/repos/real-stanford/diffusion_policy
# → license: MIT, pushed_at: 2024-12-24, stars: 4676
```

## 附录 B：主要参考链接

- EgoBridge：[项目页](https://ego-bridge.github.io/) / [arXiv:2509.19626](https://arxiv.org/abs/2509.19626) / [NeurIPS 2025 页面](https://papers.neurips.cc/paper_files/paper/2025/hash/434756a76c44e220d5f32d6e0b488c0c-Abstract-Conference.html) / [camera-ready PDF](https://papers.neurips.cc/paper_files/paper/2025/file/434756a76c44e220d5f32d6e0b488c0c-Paper-Conference.pdf)
- Diffusion Policy：[GitHub](https://github.com/real-stanford/diffusion_policy) / [项目页与数据](https://diffusion-policy.cs.columbia.edu/) / [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) / [conda_environment.yaml](https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/conda_environment.yaml) / [pusht_env.py](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/pusht/pusht_env.py) / [demo_pusht.py](https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/demo_pusht.py) / [pusht_image.yaml](https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/diffusion_policy/config/task/pusht_image.yaml)
- 替代底座：[gym-pusht](https://github.com/huggingface/gym-pusht) / [lerobot/pusht 数据集](https://huggingface.co/datasets/lerobot/pusht) / [lerobot/diffusion_pusht 模型卡](https://huggingface.co/lerobot/diffusion_pusht)
- 训练开销佐证：[lerobot#43](https://github.com/huggingface/lerobot/issues/43) / [roboticscenter.ai 指南](https://www.roboticscenter.ai/blog/diffusion-policy-guide) / [claru.ai 指南](https://claru.ai/guides/how-to-train-a-diffusion-policy)
- 4090 兼容性：[sm_89 报错示例](https://github.com/gaomingqi/Track-Anything/issues/101)
