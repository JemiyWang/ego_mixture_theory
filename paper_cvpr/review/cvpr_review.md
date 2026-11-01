# CVPR 审稿意见

**稿件**: *When Does Egocentric Human Data Help Robot Policies? A Spectral Theory of Cross-Embodiment Data Mixing*
**审稿日期**: 2026-10-24
**评审范围**: main.pdf(9 页,正文 8 节)+ supplement(引理 L1–L4、Thm T1-b、数值验证协议 v5–v16、实验细节),含编译产物与全部图表检查

---

## 论文摘要(Summary)

本文研究"何时、以何种比例将人类第一视角(ego)数据混入机器人策略训练"这一问题。作者建立了一个双域谱模型:两个域的协变量为高斯、特征谱为幂律、具身差距被分解为任务失配向量 δ,策略拟合建模为混合加权 ridge 回归。主要理论结果包括:

1. 混合风险的逐坐标闭式刻画(Thm 1 / T1-a),含一个 "label feedthrough" 有效噪声项 $\tilde\sigma_e^2 = \sigma_e^2 + F_e$;
2. 边际收益的三项分解(对齐 A / 方差 B / 覆盖 C)与精确的混合有害判据 $H$ 及临界样本量 $n_r^{\mathrm{crit}} = K/H$(Thm 2 / C2);
3. "交换率" $k_{\mathrm{ex}}$(Cor / T2),声称与经验报告的 4–5:1 替代比定量吻合。

实验在一个可控双域 Push-T 平台(镜像几何 + 紫色背景 + 圆形 pusher 的 surrogate ego 域)上用扩散策略做了第一批混合实验,声称确认了结构性预测(非单调收益、内点最优 $\lambda^* \approx 0.5$、$\lambda \to 1$ 有害边界)。

---

## 优点(Strengths)

### S1. 问题重要且时机恰当
EgoMimic / UMI / DexCap / EgoBridge 一类工作已在实践上大量混合 ego 数据,但混合比例确实是经验法则。把"一条 ego 数据值多少条机器人数据"变成可计算量的目标有价值。

### S2. 理论成体系、证明较扎实
Supplement 中 L1–L4 的推导链条完整(feedthrough 的识别、$\chi^2$ 集中、三阶混合矩 $8\lambda^3 b^3\delta^2/n_h^2$ 的闭式闭合、boundary-band 装配);T1-b 在 fixed-$p$ 情形下把对角近似的结论升级到了完整 ridge,并诚实指出 proportional 情形尚未证明。数值验证协议(v5–v16)比一般理论论文详尽;v9 的"朴素装配被证伪 → 导出 $H$ 泛函"的 falsification loop 是很好的科学实践。

### S3. "失配位置比失配幅度更重要"是新颖、可操作、可证伪的预测
(Table 1 / Fig 1c):同样的 $\|\delta\|$ 放在低频高权重坐标上几乎处处有害,放在高频坐标上无害。这对 ego 数据过滤 / retargeting 策略有直接的实践含义。

### S4. 诚实性值得肯定
论文明确报告了最优混合比的定量预测失败(预测 $\lambda^* \approx 0.12$ vs 实测 $\approx 0.5$),并列出失败来源;Limitations 一节写得充分,没有掩盖。

### S5. 写作与实验细节
写作密度高但清晰;dual-domain 平台的镜像约定细节($\theta \mapsto -\theta$ vs $\pi - \theta$,0/206 vs 206/206 回放成功率)体现了实验的严谨。

---

## 缺点(Weaknesses,按严重程度排序)

### W1. 完整性硬伤:核心证据链依赖占位引用,当前形态不可投稿

参考文献中有 **4 条 TODO-VERIFY 占位条目**(`main.bib:164-193`),且它们不是边缘引用:

| 条目 | 支撑的主张 |
|---|---|
| `zevaego2025` | 摘要和 Cor 2 声称 "$k_{\mathrm{ex}} \approx 0.2$–$0.25$ **定量匹配**经验 4–5:1 替代比"的**唯一实证锚点** |
| `wu2026mixture` | 作者自己声明的**最接近的相关工作**("Closest to us is..."),差异化论证建立在其上 |
| `egobridge2025` | 实验平台直接追随的对象(Sec 5 的三处 deviation 相对它定义) |
| `lei2026simreal` | Discussion 中负结果(over-alignment collapse)的解读依据 |

这些条目以 "TODO-VERIFY authors / TODO-VERIFY venue" 的字样**直接出现在编译后的 PDF 参考文献中**(见 `main.bbl`)。此外:

- `sec/4_theory.tex:122` 写 "Numerically (Fig.~in suppl.)",但**补充材料中不存在任何图**;
- Paper ID 未填(`main.tex:17`)。

作为正式投稿,这接近 desk-reject 级别的问题;即使忽略格式,"定量匹配经验替代比"这一卖点在引用落实前无法核实。

### W2. 实验强度不足以支撑结论

- **全部实验结论建立在每个格子 1 个 seed、20 个评估 episode 上**(`sec/6_experiments.tex:5`)。headline 数字(0.524→0.670,+28%;150 epochs 下 0.763,+32%;F4 中 MMD 非单调效应 0.475/0.464 vs 0.629/0.587)均无误差棒、无显著性检验。模仿学习闭环评估方差众所周知地大,20 episodes 单 seed 的覆盖率差异完全可能在噪声范围内。作者自己承认 "a multi-seed batch follows"——但审稿只能基于已提交的内容。
- **"human domain" 不是 ego 数据**,而是镜像几何 + 紫色背景 + 圆形 pusher 的另一个 Push-T 仿真。作者对此是诚实的("surrogate"),但标题与摘要的承诺("Egocentric Human Data")与实验对象之间的差距很大:没有真实 ego 视频、没有真实机器人、单一任务、单一策略类。
- 结果内部存在未解释的反常:F3 声称 $\lambda = 0.9$ 时(90% 训练 batch 来自镜像域)开环机器人域验证损失仍单调改善至 0.036,优于 $\lambda = 0$ 的 0.072。在动作方向被镜像翻转的情况下,这是一个强得反常的声明,单 seed 下难以采信,至少需要讨论机制(是表征获益,还是 mixture 采样带来的有效步数/正则效应?)。

### W3. 理论的关键定量预测失败,且"结构性验证"的证伪性弱

- 理论在最核心的可操作量上错了 4 倍:预测 $\lambda^* \approx 0.12$(平台参数代入 Eq (2),`sec/5_platform.tex:41`),实测 $\lambda^* \approx 0.5$。预测方向("只能少量混入")与实测方向("混入一半最优")对实践者的指导是**相反的**。论文将其归为"结构性预测之外",但这恰恰削弱了 "operational theory" 的立论。
- 已确认的三个结构性预测(非单调、存在内点最优、$\lambda \to 1$ 有害)**过于宽泛**:几乎任何 bias–variance 权衡故事都预测同样的曲线形状。理论真正独特、可证伪的预测——$H$ 判据的数值边界、mismatch-location 效应、交换率数值、$n_r^{\mathrm{crit}}$ 标度——**全部没有在学习的系统上检验**:Table 1 和 Fig 1(c) 只是理论公式自身的数值模拟。换句话说,本文实验只验证了理论最不惊人的部分。
- F3 暴露的是更深的概念问题:理论的泛函是**开环预测风险**,而论文自己的实验显示开环与闭环**反向发散**(开环改善到 $\lambda = 0.9$,闭环在 $\lambda^* = 0.5$ 后下降)。理论所优化的对象与论文标题问题(策略执行表现)目前并不对应,作者也承认该桥接 "our current model does not yet cover"。这应当被更显著地承认为当前框架的边界,而不是留作 outlook。

### W4. 交换率论述内部矛盾且量纲不一致

- 摘要/引言称 $k_{\mathrm{ex}}$ "is **capped at 1**",而 Cor 2(iv)(`sec/4_theory.tex:119`)写的是 "capped: there is a large-gap regime in which $k_{\mathrm{ex}} \ge 1$"。按 supplement Prop 1(ii)(`sec/X_suppl.tex:222`),cap 情形的真实含义是"不存在有限的机器人数据量能复现混合最优"——即 $k_{\mathrm{ex}}$ 发散,而非被截断在 1。两种表述直接矛盾,且 "capped at 1" 在字面上是错误的。
- "定量匹配 4–5:1" 的声明存在量纲换算问题:理论按**样本**计价,经验按**小时**计价。作者加注了 caveat,但换算依赖采集吞吐,未给出任何使比较成立的换算因子。在 W1 的占位引用落实前,该匹配主张没有可核查的基础。

### W5. 命名与表述不一致(影响可读性)

- 定理环境编号与内部命名脱节:`thm:h` 编译为 "Theorem 2" 但自称 "C2";`cor:kex` 编译为 "Corollary 2" 但自称 "T2";C1 是 Corollary 1。读者看到 "Theorem 2 (Harmful-mixing criterion, C2)" 会困惑。
- 二元成功率的表述三处互相矛盾:
  - 主文 protocol(`sec/6_experiments.tex:6`):"binary success at the 0.855 threshold **where attained**";
  - F2(`sec/6_experiments.tex:46`):报告 150-epoch 臂成功率 0.40 vs 0.25;
  - suppl(`sec/X_suppl.tex:515`):"the binary success threshold 0.855 is **never reached** at this training budget"。
- 结论中 "a closed-form, **numerically exact** account"(`sec/7_discussion.tex:25`)与 T1-a 对完整 ridge 尚有 8–28% 差距、对扩散策略仅有结构对应的事实不符,属于过度声明。

### W6. 图表质量低于投稿标准

- **Table 2**(`sec/6_experiments.tex:16`)LaTeX 结构错乱:列声明 `{lc|cc|cc}` 与表头、数据行的列数不匹配,存在多余空列,部分行为空;
- **Fig 2**(`fig/first_batch.png`)是内部调试产物:图内标题 "Rebatch s0 (corrected final checkpoints, single seed) + theory overlay"、面板标注 "dR0=-1.5e+04" 均未改写为论文语言;左图 theory overlay 只有 "normalized shape comparison",读者无法判断理论到底预测了什么量;
- teaser 的两张渲染图(`fig/render_robot.png` / `render_human.png`)是 96×96 的占位级图片;
- Fig 1(c) 把 9 宫格相图压进 `figure*` 的 0.47 栏宽,印刷尺寸下不可读。

### W7. 相关工作覆盖面不足

缺少与机器人数据混合的**实证文献**的对话:OXE/RT-X 的混合消融、大规模模仿学习中数据配比研究、sim-real 共训练的系统工作等;对 LLM 数据混合 scaling law 文献也只引用了一篇(且是占位条目)。对一个"数据混合理论"论文,当前的相关工作地图偏薄。

### W8. Venue 匹配度存疑

论文内容本质是 ridge 回归谱分析 + 仿真机器人学习;视觉(图像观测)只是实验的附属设置,全文没有任何计算机视觉方法或视觉基准。以 CVPR 审稿标准看,更自然的归宿是 NeurIPS/ICML(理论)或 CoRL/RSS(机器人学习)。这一点单独不构成拒稿理由(CVPR 有 embodied AI 方向),但作者应说明本文对 CVPR 读者群的核心价值。

---

## 给作者的问题(Questions)

1. 实验中的 $\mu$ 对应扩散策略训练中的什么对象?具体数值(weight decay?早停 epoch?)是多少?不建立这个对应,"$\mu$ 是经典相变的载体"这一论断与实验之间没有关系可言。
2. $\lambda^*$ 预测 0.12 vs 实测 0.5:能否用模型内部量定位主因?是 $\hat\delta / \hat c$ 的线性代理失真,还是 $\tilde\sigma^2$ 的 feedthrough 标定,还是扩散策略根本不在 ridge 渐近区?一个不依赖"下一步实验"的诊断会大幅增强可信度。
3. $H$ 判据中 $A_\infty$ 项对 $\delta$ 取逐坐标反号($|\delta_j|$)的假设,对应到平台镜像构造时逐坐标成立吗?还是只有聚合 $c < 0$?
4. Cor 2(iv) 与摘要关于 $k_{\mathrm{ex}}$ 上界的表述哪个是正确的?若 cap 情形意味着 $k_{\mathrm{ex}}$ 发散,请统一改写。
5. Table 1 的 harmful-cell 计数对 $\mu$、截断 $\kappa_0$ 的取值敏感吗?42 格网格的边界分辨率是否足以支撑 "location beats magnitude" 的强声明?
6. 能否给出多 seed 下 F1/F4 的显著性?特别是 $\mathrm{mmd}_{1.0}$ 在 $\lambda \in \{0.35, 0.7\}$ 的有害效应(0.475/0.464 vs 0.629/0.587)。

---

## 评分

| 项目 | 评分 |
|---|---|
| **Overall** | **Weak Reject**(倾向 borderline) |
| **Confidence** | 4 / 5 |

**评分理由**:理论框架认真、问题真实、写作诚实,具备成为一篇好论文的骨架;但以当前形态——4 条占位引用支撑核心主张、全部实验单 seed、最关键的定量预测偏差 4 倍且只验证了最弱的结构性预测、图表与编号处于草稿状态——距离 CVPR 接收标准有明显差距。熟悉 ridge/谱分析与模仿学习实验协议;未逐行复算 supplement 全部引理,故置信度非满分。

---

## 给作者的建设性路线(修稿建议)

1. **落实 4 条占位引用**,补上 suppl 中缺失的交换率图;修正 "capped at 1" 的矛盾表述。
2. **补多 seed**(至少 3 seeds × 现有网格的关键格子),给出误差棒;这是把 F1–F4 从"观察"升级为"证据"的最低成本路径。
3. 把至少一个**独特的**理论预测拿到学习系统上检验:最可行的是 mismatch-location 效应(在平台上构造同幅度、不同频率位置的 $\delta$);这比再确认一条 U 形曲线有说服力得多。
4. 正视开环/闭环发散:要么在理论中显式建模 compounding(哪怕一阶),要么把声明范围收窄到开环风险并在标题/摘要中体现。
5. 修正 Table 2 排版、替换调试图与低分辨率渲染图、统一定理编号(建议去掉 T1-a/C2/T2 内部命名,或使其与 LaTeX 编号一致)。
