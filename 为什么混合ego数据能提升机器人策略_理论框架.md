# 为什么混合 Ego 数据能提升机器人策略？——一个理论化研究框架

**日期**：2026-10-22（2026-10-22 一致性修订：统一记号 λ/Λ/Δ_φ/κ_ex）
**状态**：研究框架草案（基于三路联网深度调研 + 本地文献库 `wjm/ego/relative/`）

---

## 0. 一句话版本

> 把人手 ego 视频与机器人遥操作数据视为**同一任务流形经两个 embodiment 通道（观测映射、动作映射）产生的两个分布**，则"混合 ego 数据为何提升机器人策略"可以抽象为**混合分布族上的高维统计学习问题**；借用随机矩阵理论（混合协方差的确定性等价）、域适应的混合比定理（含相变）、Gromov–Wasserstein 对齐成本与模仿学习的 compounding error 链，可以产出可证明、可证伪、且目前是空白的理论。

**全文记号约定**：$\lambda$ 专指混合比例（人域占比）；$\Lambda$ 专指 Ben-David 界的理想联合假设误差；$\Delta_\varphi$ 专指重定向失配；$\kappa_{\mathrm{ex}}$ 专指汇率（ego↔遥操作有效样本折算率）；$\kappa(\hat G)$ 专指梯度矩阵条件数。Ben-David 2010 原文的目标域权重 $\alpha$ 与本文 $\lambda$ 满足 $\alpha = 1-\lambda$。

---

## 1. 研究问题

**经验事实**：训练机器人策略时，将部分（甚至全部）遥操作数据替换/补充为第一视角（egocentric）人类操作视频——其中执行任务的是**人手**而非机械臂——策略性能反而提升。

**要回答的问题**：为什么？何时成立、何时失效（负迁移）？收益随数据量、混合比例、人-机差距如何变化？

**研究范式**（参考 `hessian_geometry_of_latent_space.pdf`，ICML 2025, arXiv:2506.10632）：

```text
具体异常现象
    ↓
抽象成数学对象（参数 → 分布的映射）
    ↓
发现与成熟数学领域同构
    ↓
建立可证明的桥接定理
    ↓
在精确可解的模型上验证
    ↓
回到复杂系统，解释新现象 / 给出可证伪预测
```

---

## 2. 总体判断

1. **该问题可以作为研究问题，且时机恰好。** 2024–2026 年该领域积累了大量互相咬合的定量实证规律（scaling 律、剂量响应曲线、相变式的成败分界），但**没有任何论文给出理论解释**——所有解释停留在"coverage / 多样性 / 动作先验"等口头机制。
2. **可以抽象成数学模型，且存在几乎现成的可解模板**：Wu 2026 的混合数据 scaling law 理论（高维混合回归 + 随机矩阵理论）之于此问题，相当于 Ising 模型之于 Hessian 论文。
3. **Hessian 论文的方法论映射异常整齐**（见第 4 节）。
4. **必须正视最近的邻居工作**：Lei et al. 2026 已对 sim+real 混合训练做过扩散策略的解析机制分析（见第 9 节），差异化定位是必须的战略问题。

---

## 3. 现象基础：待解释的实证规律

任何候选理论必须同时解释以下已发表的规律：

| # | 规律 | 来源 |
|---|---|---|
| R1 | **相变式成败**：朴素混合人/机数据，行为泛化成功率 0%；动作感知联合对齐后 33%。朴素共训时人机样本在隐空间分成两个不相交区域（t-SNE 可见） | [EgoBridge](https://arxiv.org/abs/2509.19626)（NeurIPS 2025） |
| R2 | **ego 数据单位时间效率更高**：15 分钟人手视频（75%）> 30 分钟遥操作训练的 ACT（51.2%）；8 分钟人类数据 > 30 分钟机器人数据（采集效率 3.75×） | [HumanEgo](https://arxiv.org/abs/2605.24934) |
| R3 | **log-linear scaling 律**：ego 预训练验证损失 L = 0.024 − 0.003·ln(D)，R²=0.998（1k–20k 小时）；HumanScale 复现 L = a − b·ln(D)，R²=0.86–0.94 | [EgoScale](https://arxiv.org/abs/2602.16710)、[HumanScale](https://arxiv.org/abs/2606.20521) |
| R4 | **机器人数据 OOD 不 scale**：真机预训练在 unseen 验证集上 loss 不随数据量下降（停在 ~0.025），ego 预训练持续下降；100 小时 ego ≈ 45,000 条轨迹 vs 遥操作 ~8,000 条（信息密度 5–6×） | [HumanScale](https://arxiv.org/abs/2606.20521) |
| R5 | **规模与对齐解耦互补**：20k 小时噪声人类预训练 + 50h 人/4h 机的对齐 mid-training > 任一单独使用（56% vs 38%/28%） | [EgoScale](https://arxiv.org/abs/2602.16710) |
| R6 | **边际收益递减 / 负迁移存在**：迁移增益随机器人数据增多递减（低数据 regime +29.7%）；强制边缘动作对齐"注定失败"（guaranteed to fail）；EgoBridge 的 OT 对齐在动作差距大时不如纯 BC | [What Matters When Cotraining](https://arxiv.org/abs/2606.06627)（MIT/UIUC 2026） |
| R7 | **不变状态抽象是瓶颈，视觉外观不是**：逐步消除视觉具身差异只到 32.5%（7.5%→20%→32.5%），加入交互中心表征（ICT，手-物 6DoF 相对关系）跳到 85% | [HumanEgo](https://arxiv.org/abs/2605.24934) |
| R8 | **经验汇率**：~10K 小时 ego ≈ 2K 小时机器人数据（75.3% vs 74.7%），汇率约 4–5:1 | [Zeva-Ego](https://arxiv.org/abs/2609.24411) |
| R9 | **剂量-响应**：手部姿态真值加噪（0.5×/1.0×），迁移效果单调下降；单目估计（W-MPJPE≈162mm）仍有部分迁移 | [What Matters](https://arxiv.org/abs/2606.06627) |
| R10 | **OOD 收益远大于 ID 收益**：ego 预训练 OOD 成功率 +90% vs ID +52.5%；无预训练基线 OOD 直接崩溃（0%） | [HumanScale](https://arxiv.org/abs/2606.20521)；EgoBridge 新物体+新场景 27% vs 基线 0% |

**机制归因的可识别性问题**（构建理论前必须正视）：当前实验中多样性、信息密度、动作先验彼此共线（更多小时 = 更多场景 = 更多轨迹），需要能单独操纵单变量的受控环境。

---

## 4. 方法论映射：Hessian 论文范式 → 本问题

| Hessian 论文的做法 | 本研究的对应 |
|---|---|
| 观察异常现象：隐空间微小变化导致生成图突变 | 观察异常规律：R1（0% → 33% 的对齐相变）、R4（机器人数据 OOD 完全不 scale）、R2（人数据单位时间效率反超） |
| 抽象：隐变量不是向量，而是参数化了一个分布 p(x\|t) | **embodiment 不是数据标签，而是同一任务流形上两套观测/动作映射的"参数"**：人手视频与遥操作视频是同一底层任务过程经两个"渲染通道"产生的分布 |
| 发现与统计物理/信息几何同构 | 与三个成熟领域同构：① 混合分布族 = 信息几何的 **mixture family**，混合比 λ 是 m-flat 仿射坐标（Amari 对偶结构）；② 两域混合学习的泛化曲线 = 高维统计中**混合协方差的 spiked 模型**，可用随机矩阵理论求确定性等价；③ 人/机动作空间差异可用 **Gromov-Wasserstein 距离**度量（无需共同坐标系） |
| 桥接定理（其 Thm 3.1/3.2） | 本文第 6 节的 T1–T4 |
| 在 Ising / TASEP 上精确验证 | 在线性-高斯因子模型（解析可解）与 EgoBridge 的 Push-T 双域仿真（受控"半精确"）上验证 |
| 回到扩散模型解释新现象（分形相界） | 回到真实策略给出可证伪预测（第 7 节），用本地 stack_blocks 配对数据检验 |
| 假设边界：失效处解释成相边界 | 失效测试现成：理论必须同时预言"何时有害"（R6 的 guaranteed-to-fail 与负迁移），见第 8 节 |

---

## 5. 数学抽象

**共享任务流形 + embodiment 索引的观测/动作映射。**

设存在低维任务潜变量 $z \in \mathcal{Z}$（手—物相对位姿、交互阶段）与 embodiment 无关的运动程序 $a^*$。embodiment $e \in \{\text{人}, \text{机}\}$ 选择两个通道：

- **观测通道** $g_e: z \mapsto o_e$（人：ego 相机 + 人手外观；机：固定相机 + 机械臂外观）
- **动作通道** $h_e: a^* \mapsto a_e$（手腕运动 → 末端位姿/关节角，含重定向算子 $\varphi$；其失配记为 $\Delta_\varphi$）

策略在混合分布 $P_\lambda = (1-\lambda) P_{\text{机}} + \lambda P_{\text{人}}$ 上做行为克隆。研究对象是**机器人域闭环风险** $\mathcal{R}_{\text{机}}(\hat\pi_\lambda)$ 作为 $(\lambda, n_{\text{人}}, n_{\text{机}}, \text{gap})$ 的函数。经 Ross & Bagnell 的 compounding error 链（BC 单步误差 ε → 闭环 $O(T^2\varepsilon)$，[arXiv:1011.0686](https://arxiv.org/abs/1011.0686)），理论主战场放在分布层面。

**几何图景（纤维丛）**：任务流形是底空间，embodiment 外观是纤维，策略是丛的截面。ego 数据负责**加密底空间覆盖**；机器人数据负责**把截面锚定在机器人纤维上**；EgoBridge 式对齐损失本质是**规范固定（gauge fixing）**。理论需要的三个量由此自然出现：底空间覆盖半径、锚定样本数、规范缺陷（对齐误差）。

---

## 6. 定理体系

> **范围说明（2026-10-22 分层修订）**：T1–T4 是**研究纲领级**的完整体系；落到单篇论文时按分层收敛——T1 为唯一主定理、T2 为其推论、T4 降级为一节"边界"（至多一个短命题 + 引用 + remark）、T3 降级为闭环桥（标度关系 + 单实验 P4）。详见《论文研究方案.md》§2.2 与 §3.3；若 T3 的闭环实验发现新规律，可拆分为后续论文（对应 §10 缺口 #4）。

### T1（主线）：混合收益相图定理

**工具来源**：[Wu 2026, "When do data mixtures improve scaling laws? Insights from high-dimensional mixed-data regression"](https://arxiv.org/abs/2609.38011)（证明模板：minimax + 椭圆约束下 ridge 回归的确定性等价）+ Ben-David 2010 §6 的相变结构。

**建模**：ego 域 = 与机器人域**共享回归结构**、但协方差谱更宽、标签噪声更高、带 embodiment 失配项的辅助域。

**定理目标**：

> 存在 $\lambda^* > 0$ 使混合严格优于任一单域，**当且仅当**两域协方差谱衰减率与相对样本量满足某显式不等式。输出 $(\lambda, \text{谱}, n_{\text{人}}/n_{\text{机}})$ 空间中的"有益/有害"相区划分。

**解释的现象**：R3（log-linear 律）、R4（机器人数据谱太窄 → 标度指数为零的相区）、R5（规模轴与对齐轴正交）。呈现形式可直接借用统计物理的"掺杂相图"语言（λ = 掺杂浓度，标度指数 = 序参量）。

**已知锚点**：Ben-David et al. 2010, Theorem 3 + §6（[论文](https://www.alexkulesza.com/pubs/adapt_mlj10.pdf)）对凸组合目标 $\hat\varepsilon_\alpha = \alpha\hat\varepsilon_T + (1-\alpha)\hat\varepsilon_S$ 给出闭式 $\alpha^*(m_T, m_S; A)$，自带两个相变（注意其 $\alpha$ 为目标域损失权重，与本文混合比 $\lambda$ 满足 $\alpha = 1-\lambda$，故 $\alpha^*=1 \iff \lambda^*=0$）：

- $m_T \ge d/A^2$ 时 $\alpha^*=1$（目标数据足够 → 源数据只有害）→ 解释 R6 的 diminishing returns；
- 源数据太少时 $\alpha^*$ 也回到 1（少量源数据不值得混入）；
- $A \to 0$ 时 $\alpha^* = \beta$（按样本量等权）。

**创新点**：该定理基于 VC 维 + 0-1 损失，从未实例化到跨具身场景。把 $d$ 换成表征空间复杂度（Maurer 2016 的数据依赖协方差范数 $\|\hat C\|_1$），把 $A = \frac12 d_{\mathcal{H}\Delta\mathcal{H}} + \Lambda$ 中的理想联合误差 $\Lambda$ **结构化为重定向失配** $\Delta_\varphi = \|\delta\|^2 + \|\Pi_\perp\beta^*\|^2$（投影损失 + 系统偏移）+ 具身不变子空间残差。

### T2：汇率定理（ego 小时 ↔ 遥操作小时）

**工具来源**：[Ben-David, Shalev-Shwartz & Urner 2014, "Domain adaptation – can quantity compensate for quality?"](https://cs.uwaterloo.ca/~shai/ISAIM12.pdf)：数量可以补偿质量，源数据放大因子 $\approx C^{-(d+1)}$（$C$ = 两域密度比下界，$d$ = 有效维数）；[Hanneke & Kpotufe 2019](https://openreview.net/forum?id=SJG4yBHx8S) 的 transfer exponent $\rho(\varepsilon)$（源数据只能把目标误差压到由 ρ 决定的地板）；GW 距离度量跨空间对齐成本（[Mémoli 2011](https://link.springer.com/article/10.1007/s10208-011-9093-5)；熵正则 GW 参数速率估计 [Zhang & Goldfeld 2023](https://arxiv.org/html/2212.12848v3)）；有效样本量 [Cai & Wei 2021](https://arxiv.org/abs/1906.02909)。

**定理目标**：

> $n_{\text{eff}} = n_{\text{机}} + \kappa_{\mathrm{ex}}(d_{\text{gap}}, \Delta_\varphi) \cdot n_{\text{人}}$，其中汇率 $\kappa_{\mathrm{ex}} \in [0,1]$ 单调递减于 gap 度量。$d_{\text{gap}}$ 为候选域差度量（表征空间 W2 / $\mathcal{H}\Delta\mathcal{H}$ / 动作空间 GW），由实验确定最具预测力者。

**解释的现象**：R8（Zeva-Ego 的 4–5:1 汇率 $\approx \kappa_{\mathrm{ex}} \in [0.2, 0.25]$ 的实测值）、R9（手部姿态噪声 → $\kappa_{\mathrm{ex}}$ 单调下降的剂量响应）。

### T3：开环-闭环转移定理

**工具来源**：Ross & Bagnell compounding error（$O(T^2\varepsilon)$）。

**定理目标**：

> 闭环成功率损失 = f(开环标度指数, 任务 horizon T, 协变量漂移放大因子)。

**意义**：所有现有 scaling 律的 y 轴都是开环验证 loss，闭环成功率是否同律**完全未知**（自动驾驶已有开环律闭环失效先例）。注意 $T^2$ 放大意味着**单步误差的小幅下降会被平方放大为闭环收益**——这本身就是"少量 ego 数据产生大成功率效果"的天然杠杆。

### T4：负迁移边界（失效刻画）

三条可引用的锚：

1. **[Zhao et al. 2019](https://arxiv.org/abs/1901.09453)（标签偏移下界）**：即使表征完美域不变且源误差为零，$\varepsilon_T \ge \frac12 D_{JS}(D_S^Y, D_T^Y)$。人/机动作边缘分布天然不同 → **盲目表征对齐必然失败**。这是 What Matters "guaranteed to fail" 观察的严格对应。
2. **[Rosenfeld et al. 2021](https://arxiv.org/abs/2010.05761)（IRM 失败条件）**：非线性 IRM 可灾难性失败；且本问题"环境数 E=2 ≪ 伪特征维数"，不满足可识别条件 → IRM 路线只能做部分不变性（外观子空间不变 + 动作头域条件化）。
3. **[Ben-David, Lu, Luu, Pál 2010](https://proceedings.mlr.press/v9/david10a/david10a.pdf)（不可能性）**：域相近 + 少量目标数据不足以保证适应成功；[Ben-David-Shalev-Shwartz-Urner 的负结果](https://cs.uwaterloo.ca/~shai/ISAIM12.pdf)：proper learning 下无穷多源数据也不能替代目标数据。→ 为"混合而非替代"定理论基调。

**理论推论**：表征对齐必须是**结构化的**（对手臂外观子空间不变、对动作头保留域条件），不能是坍缩式的。与 R7（HumanEgo 的 ICT 消融）互为印证——Zhao 下界管"为什么不能只做对齐"，ICT 结果管"为什么不变交互表征是上限的解锁项"。

---

## 7. 可证伪预测清单（理论区别于叙事的分界线）

| # | 预测 | 检验方式 | 已有证据 |
|---|---|---|---|
| P1 | **序参量相变**：隐空间域间 Wasserstein 距离 / 域分类器 AUC 随对齐强度或 λ 跨越临界值时锐变；临界附近两域梯度余弦相似度取极小 | 受控仿真中扫对齐强度/混合比 | EgoBridge t-SNE（朴素共训分成两簇）为定性证据 |
| P2 | **收益曲线非单调**：Δ(λ) 存在内点最优 λ\*，λ\* 随 n_机 增大而减小、随对齐质量增大而增大 | 混合比扫描 + Ben-David α\* 公式拟合（α = 1−λ） | R6 diminishing returns；DexWild 的 2:1/1:1/1:2 消融 |
| P3 | **汇率可预测**：$\kappa_{\mathrm{ex}}$ 对 gap 度量（重定向失配 $\Delta_\varphi$、域差 $d_{\text{gap}}$）单调 | 手部姿态加噪的剂量-响应曲线拟合 T2 函数形 | R9（What Matters 已给一条边） |
| P4 | **开环-闭环偏差**：闭环成功率与开环 loss 的偏离由 compounding error 项预测（horizon 越长偏差越大） | 同一 checkpoint 的开环/闭环双评测 + 变 horizon 任务 | 无（空白） |
| P5 | **覆盖 vs 不变性分离**：固定数据总量，单独操纵底空间覆盖度（场景/物体多样性），OOD 收益应随覆盖度单调而 ID 收益快速饱和 | 受控 ego 数据采集设计（Level 1 仿真中设覆盖度轴） | R10、Lin et al. 2025（多样性是唯一不饱和的轴） |
| P6 | **梯度条件数影子**：Aligned-MTL 的 $\kappa(\hat G)$（堆叠任务梯度矩阵条件数；勿与 T2 汇率 $\kappa_{\mathrm{ex}}$ 混淆）作为具身差距在参数空间的度量，预测负迁移 onset | 训练中记录双域梯度矩阵谱 | [Senushkin et al. 2023](https://arxiv.org/abs/2305.19000) 在线性 MTL 已证稳定性由 κ 控制 |

---

## 8. 验证序列（对应"先精确可解、再复杂模型"）

- **Level 0（Ising 位置）**：线性-高斯因子模型 + ridge 回归，全部解析。直接站在 Wu 2026 框架内，改动是加入 embodiment 失配项 $\Delta_\varphi$（重定向算子 φ 的投影损失与系统偏移）。
- **Level 1（TASEP 位置）**：EgoBridge 论文的 Push-T 双域仿真（圆形 vs 三角形推动器、镜像 T 形、异色背景）是现成的受控环境，可单独操纵 gap 的每个分量与覆盖度轴——同时解决机制共线的可识别性问题。
- **Level 2（回到扩散模型的位置）**：本地 `dataset/ego` 的 stack_blocks ego+遥操作配对数据（`task.json` 中 `teleop_dataset` 指向 LeRobot 格式遥操作集）。注意先修帧级时间同步（`timing_quality: host_events_not_frame_sync`）。
- **Level 3（规模验证）**：Ego4D / DROID 等公开数据子集上的 scaling 律复现与外推检验。

---

## 9. 最近邻工作与差异化定位

**[Lei et al. 2026, "A Mechanistic Analysis of Sim-and-Real Co-Training in Generative Robot Policies"](https://science-of-co-training.github.io/)**（[arXiv:2604.13645](https://arxiv.org/abs/2604.13645)）：对扩散策略的 sim+real 混合训练给出解析分析——三种区域（disjoint 无迁移 / structured aligned 有效迁移 / overlapping 表征坍缩→双峰动作→负迁移），约解释 50% 损失方差；混合比实用准则 $w_n = N/(N+M) \le w \le \sqrt{N/M}$。**这是最近的邻居：把 sim 换成 ego 即本课题。**

差异化在于三个它不覆盖的轴：

1. **gap 的结构不同**：sim-real 主要是视觉差；ego-robot 多了运动学/动作空间差（重定向算子 φ 及其失配 $\Delta_\varphi$、时序速度差 2–3×）——其模型没有该项；
2. **覆盖/scaling 轴**：ego 数据的独特价值是可大规模获取的多样性（R3/R4/R5），sim 数据没有这个轴；统一"机制相图"与"混合比/scaling 相图"是空白；
3. **闭环闭合**：其分析停在单步损失；T3 的 compounding error 闭合无人做过。

其他必须引用的相邻工作：[Cheng et al. NeurIPS 2025](https://arxiv.org/abs/2509.18631)（sim-real 域适应）、[Kareer et al. 2025](https://arxiv.org/abs/2512.22414)（VLA 规模人→机迁移涌现）、[Lin et al. ICLR 2025](https://arxiv.org/abs/2410.18647)（模仿学习数据 scaling 律：泛化对环境/物体多样性呈幂律、对演示数 ~50 条即饱和）、[Towards Embodiment Scaling Laws in Locomotion](https://arxiv.org/abs/2505.05753)（具身数量作为独立 scaling 轴）、[Shidani et al. 2025](https://arxiv.org/abs/2510.08095)（合成数据作为正则化的闭式最优混合比，U 形曲线）、LLM 侧 [Data Mixing Laws (Ye et al. 2024)](https://arxiv.org/abs/2403.16952)（L_i(r) = c_i + k_i·exp(Σ t_ij r_j)）、[Dai et al. 2026](https://arxiv.org/abs/2606.08167)（capacity competition + noise reduction 解释混合律）、[DoReMi](https://arxiv.org/abs/2305.10429) / [DoGE](https://arxiv.org/abs/2310.15393)（域权重优化）。

---

## 10. 文献确认的理论缺口（本研究的立足点）

1. **完全空白**：没有任何工作给出 L(D_机, D_人, gap) 的联合函数形式——所有 scaling 律都是单变量的，混合比例只出现在消融表里。
2. **完全空白**：embodiment gap 没有进入任何泛化界的可测量项。[The Embodiment Gap 综述](https://arxiv.org/abs/2608.18433) 只给了操作性定义，无数学 bound。
3. **完全空白**：IB/IRM 框架未被真正借用于人-机数据混合。
4. **部分空白**：所有拟合律的 y 轴是开环验证 loss，闭环成功率是否同律未验证（T3 的位置）。
5. **可识别性缺口**：多样性/信息密度/动作先验在当前实验中彼此共线，缺能单独操纵单变量的因果设计（Level 1 仿真的位置）。

---

## 11. 风险与诚实评估

- **假设缝隙**：RMT 要求协方差结构假设（高斯/椭圆），与真实扩散策略的缝隙同 Hessian 论文"指数族↔扩散模型"的缝隙同性质——处理范式也一样：写明边界，把失效处本身变成研究对象（如协方差不可交换时相图如何变形）。
- **特征学习路线最忠实但最难**：ego 收益的本质可能是特征学习（多任务表征分摊：[Maurer 2016](https://www.jmlr.org/papers/v17/maurer16a.html) 的 T 任务分摊界、[Tripuraneni 2020](https://arxiv.org/abs/2006.11650) 的任务多样性参数 ν），但 NTK/lazy 假设与此矛盾，需用 2024 年后才成熟的 feature-learning 理论。建议作为第二期工作。
- **IRM 路线脆弱**（E=2 ≪ d_e 不可识别 + 非线性失败定理），仅作叙事层。
- **竞争态势**：GEN-0 的 ossification 讨论、Dai 2026、GW-VLA（ICML 2026）、Lei 2026 都在邻域活动。窗口存在但不会一直在。

---

## 12. 建议的推进路线

**第一步（精读三篇）**：Ben-David 2010（§5–6 混合比定理）、Lei et al. 2026（最近邻居）、Wu 2026（证明模板）。

**里程碑**：

1. M1：完成 Level 0 解析模型（线性-高斯 + 重定向失配项 $\Delta_\varphi$），复现并扩展 Wu 2026 的相图，产出 T1 初稿；
2. M2：Level 1 Push-T 双域仿真实验，检验 P1/P2/P5/P6 与 T4a 边界，标定理论常数；
3. M3：T2 汇率定理 + 本地 stack_blocks 配对数据实测（P3），T4 边界实验（故意制造动作边缘分布偏移验证 Zhao 下界预测）；
4. M4：T3 闭环转移（开环/闭环双评测），成文。

**论文叙事骨架**：现象（R1–R10）→ 抽象（任务流形 × embodiment 通道，λ 为 mixture-family 坐标）→ 主定理（T1 相图 + T2 汇率）→ 失效刻画（T4，呼应 Hessian 论文的"假设告诉我们哪里失效"）→ 闭环闭合（T3）→ 可证伪预测逐条检验。

（与《论文研究方案.md》的对应：方案 §6 的 M0–M5 是本节 M1–M4 的日期化展开，增加了 M0 精读与 M5 写作投稿两个阶段，内容一一对应。）

---

## 13. 完整参考文献

### 现象侧（ego/人数据 → 机器人策略）

- HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos, 2026, [arXiv:2605.24934](https://arxiv.org/abs/2605.24934)
- HumanScale: Egocentric Human Video Can Outperform Real-Robot Data for Embodied Pretraining, 2026, [arXiv:2606.20521](https://arxiv.org/abs/2606.20521)
- EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data, 2026, [arXiv:2602.16710](https://arxiv.org/abs/2602.16710)
- EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data, NeurIPS 2025, [arXiv:2509.19626](https://arxiv.org/abs/2509.19626)
- What Matters When Cotraining Robot Manipulation Policies on Everyday Human Videos?, 2026, [arXiv:2606.06627](https://arxiv.org/abs/2606.06627)
- Zeva-Ego, 2026, [arXiv:2609.24411](https://arxiv.org/abs/2609.24411)
- EgoMimic: Scaling Imitation Learning via Egocentric Video, ICRA 2025, [arXiv:2410.24221](https://arxiv.org/abs/2410.24221)
- EgoZero: Robot Learning from Smart Glasses, 2025, [arXiv:2505.20290](https://arxiv.org/abs/2505.20290)
- EgoVLA, 2025, [arXiv:2507.12440](https://arxiv.org/abs/2507.12440)
- DexWild, 2025, [arXiv:2505.07813](https://arxiv.org/abs/2505.07813)
- Emergence of Human to Robot Transfer in VLA Models, 2025, [arXiv:2512.22414](https://arxiv.org/abs/2512.22414)
- The Embodiment Gap in Robot Foundation Models（综述）, 2026, [arXiv:2608.18433](https://arxiv.org/abs/2608.18433)
- Robot Learning from Human Videos: A Survey, 2026, [arXiv:2604.27621](https://arxiv.org/html/2604.27621v1)

### 机器人侧 scaling / 混合先例

- Lin et al., Data Scaling Laws in Imitation Learning for Robotic Manipulation, ICLR 2025, [arXiv:2410.18647](https://arxiv.org/abs/2410.18647)
- Lei et al., A Mechanistic Analysis of Sim-and-Real Co-Training in Generative Robot Policies, 2026, [arXiv:2604.13645](https://arxiv.org/abs/2604.13645)
- Cheng et al., Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training, NeurIPS 2025, [arXiv:2509.18631](https://arxiv.org/abs/2509.18631)
- Ai et al., Towards Embodiment Scaling Laws in Robot Locomotion, CoRL 2025, [arXiv:2505.05753](https://arxiv.org/abs/2505.05753)
- Open X-Embodiment, [arXiv:2310.08864](https://arxiv.org/abs/2310.08864)

### 域适应与迁移学习理论

- Ben-David et al., A theory of learning from different domains, Machine Learning 79:151–175, 2010, [PDF](https://www.alexkulesza.com/pubs/adapt_mlj10.pdf)（Theorem 3 + §6 最优混合比）
- Mansour, Mohri, Rostamizadeh, Domain adaptation: learning bounds and algorithms, COLT 2009, [arXiv:0902.3430](https://arxiv.org/abs/0902.3430)
- Zhang et al., Bridging theory and algorithm for domain adaptation (MDD), ICML 2019, [arXiv:1904.05801](https://arxiv.org/abs/1904.05801)
- Ben-David, Lu, Luu, Pál, Impossibility theorems for domain adaptation, AISTATS 2010, [PMLR v9:129](https://proceedings.mlr.press/v9/david10a/david10a.pdf)
- Ben-David, Shalev-Shwartz, Urner, Domain adaptation – can quantity compensate for quality?, AMAI 70:185–202, 2014, [PDF](https://cs.uwaterloo.ca/~shai/ISAIM12.pdf)
- Zhao et al., On learning invariant representations for domain adaptation, ICML 2019, [arXiv:1901.09453](https://arxiv.org/abs/1901.09453)
- Hanneke & Kpotufe, On the value of target data in transfer learning, NeurIPS 2019, [OpenReview](https://openreview.net/forum?id=SJG4yBHx8S)
- Wu, Zhang, Ré, Understanding and improving information transfer in multi-task learning, ICLR 2020, [arXiv:2005.00955](https://arxiv.org/abs/2005.00955)
- Redko et al., A survey on domain adaptation theory, [arXiv:2004.11829](https://arxiv.org/abs/2004.11829)

### 多任务/表征学习理论

- Baxter, A model of inductive bias learning, JAIR 12:149–198, 2000
- Maurer, Pontil, Romera-Paredes, The benefit of multitask representation learning, JMLR 17(81), 2016, [arXiv:1505.06279](https://arxiv.org/abs/1505.06279)
- Tripuraneni, Jordan, Jin, On the theory of transfer learning: the importance of task diversity, NeurIPS 2020, [arXiv:2006.11650](https://arxiv.org/abs/2006.11650)
- Du, Hu, Kakade, Lee, Lei, Few-shot learning via learning the representation, provably, ICLR 2021, [arXiv:2006.13431](https://arxiv.org/abs/2006.13431)

### 数据混合 scaling 理论（LLM 侧）

- Wu, When do data mixtures improve scaling laws? Insights from high-dimensional mixed-data regression, 2026, [arXiv:2609.38011](https://arxiv.org/abs/2609.38011)（**主线证明模板**）
- Dai et al., Explaining Data Mixing Scaling Laws, 2026, [arXiv:2606.08167](https://arxiv.org/abs/2606.08167)
- Ye et al., Data Mixing Laws, ICLR 2025, [arXiv:2403.16952](https://arxiv.org/abs/2403.16952)
- Shukor et al., Scaling Laws for Optimal Data Mixtures, 2025, [arXiv:2507.09404](https://arxiv.org/abs/2507.09404)
- Shidani et al., Beyond Real Data: Synthetic Data through the Lens of Regularization, 2025, [arXiv:2510.08095](https://arxiv.org/abs/2510.08095)
- Xie et al., DoReMi, NeurIPS 2023, [arXiv:2305.10429](https://arxiv.org/abs/2305.10429)
- Fan et al., DoGE, ICML 2024, [arXiv:2310.15393](https://arxiv.org/abs/2310.15393)
- Chen et al., Skill-it!, NeurIPS 2023, [arXiv:2307.14430](https://arxiv.org/abs/2307.14430)

### 信息几何 / 最优传输 / 统计物理 / 优化

- Amari & Nagaoka, Methods of Information Geometry, 2000（mixture family, m-flat 坐标）
- Chentsov 定理现代严格化：Ay, Jost, Lê, Schwachhöfer, [arXiv:1207.6736](https://arxiv.org/abs/1207.6736)
- Mémoli, Gromov–Wasserstein Distances and the Metric Approach to Object Matching, FoCM 2011, [link](https://link.springer.com/article/10.1007/s10208-011-9093-5)
- Zhang, Goldfeld et al., Gromov-Wasserstein: Entropic Regularization, Duality, Sample Complexity, 2023, [arXiv:2212.12848](https://arxiv.org/abs/2212.12848)
- Fickinger et al., Cross-Domain Imitation Learning via Optimal Transport (GWIL), ICLR 2022, [arXiv:2110.03684](https://arxiv.org/abs/2110.03684)
- Choromanska et al., The Loss Surfaces of Multilayer Networks, AISTATS 2015（spin glass 同构）
- Pennington & Bahri, Geometry of Neural Network Loss Surfaces via RMT, ICML 2017
- Sener & Koltun, Multi-Task Learning as Multi-Objective Optimization (MGDA), NeurIPS 2018, [arXiv:1810.04650](https://arxiv.org/abs/1810.04650)
- Yu et al., Gradient Surgery for Multi-Task Learning (PCGrad), NeurIPS 2020, [arXiv:2001.06782](https://arxiv.org/abs/2001.06782)
- Liu et al., Conflict-Averse Gradient Descent (CAGrad), NeurIPS 2021, [arXiv:2110.14048](https://arxiv.org/abs/2110.14048)
- Senushkin et al., Aligned-MTL, CVPR 2023, [arXiv:2305.19000](https://arxiv.org/abs/2305.19000)（κ(Ĝ) 条件数稳定性定理）

### 模仿学习闭环理论

- Ross & Bagnell, A Reduction of Imitation Learning to No-Regret Online Learning (DAgger), [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
- Ke et al., Imitation Learning as f-Divergence Minimization, 2019, [arXiv:1905.12888](https://arxiv.org/abs/1905.12888)

### 方法论参考

- Lobashev et al., Hessian Geometry of Latent Space in Generative Models, ICML 2025, [arXiv:2506.10632](https://arxiv.org/abs/2506.10632)（本地：`wjm/ego/relative/hessian_geometry_of_latent_space.pdf`）
