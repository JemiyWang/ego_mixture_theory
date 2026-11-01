# 固定 \(\lambda\) full-ridge 二阶风险专项执行方案

## 1. 目标与边界

在固定 \(p\)、固定 \(\mu>0\)、高斯双域设计、共同谱基和固定比例 \(\rho=n_h/n_r\) 下，证明实际 full-ridge 目标风险满足

\[
R_{\mathrm{full}}(\lambda;n_r)
=R_0(\lambda)+\frac{R_1(\lambda)}{n_r}+o(n_r^{-1}).
\]

先处理固定 \(\lambda\in(0,1)\)，再分别检查端点 \(\lambda=0,1\)。随后判断展开是否在紧子区间上一致成立。只有一致风险和风险导数控制闭合后，才评估优化后风险。

本阶段不直接宣称：

- 旧数值 surrogate 等于已证明的二阶风险；
- 最优混合比例、剂量律、汇率或全局单峰性已经迁移；
- 比例区间 \(p\asymp n\) 或非线性策略结论。

## 2. 精确分解

令

\[
G_e=X_e^\top X_e/n_e,\qquad Z_e=G_e-\Sigma_e,
\]
\[
A=(1-\lambda)G_r+\lambda G_h+\mu I,\qquad B=A^{-1},
\]
\[
D=(1-\lambda)\Sigma_r+\lambda\Sigma_h+\mu I,\qquad R=D^{-1},
\]
\[
Z=(1-\lambda)Z_r+\lambda Z_h,\qquad
v=-\mu\beta^*+\lambda\Sigma_h\delta.
\]

条件化标签噪声后，实际风险严格拆成

\[
R_{\mathrm{full}}
=\mathbb E[(v+\lambda Z_h\delta)^\top B\Sigma_rB(v+\lambda Z_h\delta)]
+\mathbb E\operatorname{tr}\{\Sigma_rBNB\},
\]

其中

\[
N=\frac{(1-\lambda)^2\sigma_r^2}{n_r}G_r
+\frac{\lambda^2\sigma_h^2}{n_h}G_h.
\]

分别计算：

1. 确定向量 \(v\) 的 resolvent 二阶修正；
2. 失配波动平方 \(\lambda^2\delta^\top Z_hB\Sigma_rBZ_h\delta\)；
3. 失配与 resolvent 的交叉项 \(2\lambda v^\top B\Sigma_rBZ_h\delta\)；
4. 标签噪声方差；
5. 好事件 Taylor 余项和坏事件两项。

不能因为 \(\mathbb E Z_h=0\) 就删掉第 3 项：\(B\) 与 \(Z_h\) 相关，其乘积是 \(n_h^{-1}\) 阶。

## 3. 已推导并通过核验的定理系数

对实高斯 Wishart 矩定义

\[
\mathcal T_e(M)=\Sigma_eM\Sigma_e+\operatorname{tr}(\Sigma_eM)\Sigma_e,
\]
\[
\mathcal T=(1-\lambda)^2\mathcal T_r+\frac{\lambda^2}{\rho}\mathcal T_h,
\qquad U=\mathcal T(R),\qquad S=R\Sigma_rR.
\]

resolvent 修正矩阵为

\[
C=RUS+SUR+R\mathcal T(S)R.
\]

相应的 \(n_r^{-1}\) 系数为

\[
\begin{aligned}
R_1={}&v^\top Cv
+\frac{\lambda^2}{\rho}\operatorname{tr}\{S\mathcal T_h(\delta\delta^\top)\}\\
&-\frac{2\lambda^2}{\rho}v^\top[R\mathcal T_h(S)+S\mathcal T_h(R)]\delta\\
&+(1-\lambda)^2\sigma_r^2\operatorname{tr}(S\Sigma_r)
+\frac{\lambda^2\sigma_h^2}{\rho}\operatorname{tr}(S\Sigma_h).
\end{aligned}
\]

人口主项必须为

\[
R_0=v^\top Sv.
\]

推导时重点检查：

- 两个 \(B\) 的二阶项与两个一阶项的交叉配对；
- 人域和机器人域 \(Z_r,Z_h\) 独立，交叉协方差为零；
- 失配交叉项的因子 \(2\lambda\) 和负号；
- 噪声项只需保留 \(B=R\) 的主项，其他噪声修正是 \(n_r^{-2}\)；
- \(\rho=n_h/n_r\)，不能把 \(1/n_h\) 错写成 \(1/n_r\)。

## 4. 证明工作单

### L0：矩与事件

- 写出 \(\mathbb E[Z_eMZ_e]=\mathcal T_e(M)/n_e\)；
- 在 \(\|Z\|\le\mu/2\) 上使用
  \(B=R-RZR+RZRZR+O_p(\|Z\|^3)\)；
- 坏事件上分别控制实际风险和扣除的近似项；
- 明确常数对 \(p,\mu^{-1},\beta^*,\delta,\Sigma_r,\Sigma_h,\rho\) 的依赖。

### L1：确定偏差 sandwich

计算 \(\mathbb E[B\Sigma_rB]\) 到 \(n_r^{-1}\)，得到 \(S+C/n_r\)。不得假定样本 Gram 对角，也不得沿用对角化 feedthrough 独立性。

### L2：失配平方与交叉项

失配平方只需人口 resolvent 的首项；交叉项只需每个 \(B\) 的一阶波动。检查人口主项为零时，失配平方仍能产生正风险。

### L3：标签噪声与合并

条件化噪声，保留 \(G_e=\Sigma_e+Z_e\) 的均值主项。合并 L1–L3，验证回到 T1-b 的 \(R_0\) 和人口噪声项。

### L4：一致性和求导边界

- 固定 \(\lambda\) 的证明完成后，检查紧子区间上的一致性；
- 端点单独验证；
- 若要对 \(\lambda\) 求导，另写导数误差界，不从固定点定理直接推出；
- 优化后风险另需唯一性、非退化曲率和全局条件。

## 5. 数值验收

使用现有 `gen_two_domain` 和 `e2_full_ridge` 直接计算真实风险，不只验证公式代数。

必须覆盖：

- \(p=1\) 的人口主项精确抵消例：\(R_0=0\)，实际风险为正；
- \(p=2\) 非对称谱和正、反失配；
- 谱系 A 的 \(p=20\) 配置；
- \(\lambda=0,0.1,0.3,0.5,0.8,1\)；
- 多组 \(n_r\)，固定 \(n_h/n_r\)，检查 \(n_r[R_{\rm MC}-R_0]\to R_1\)；
- 比较旧 `MSE_full,num`，但不得以旧 surrogate 为通过标准。

通过标准：

1. 紧凑矩阵公式与独立逐项配对一致；
2. 抵消例的首个非零项正确；
3. 随 \(n_r\) 增大，\(R_{\rm MC}-R_0-R_1/n_r\) 相对一阶修正下降；
4. 失败单元保留并区分公式、实现、Monte Carlo 和高阶余项问题。

## 6. 阶段出口与后续顺序

只有风险公式、证明和数值核验全部通过，才更新论文为固定 \(\lambda\) 二阶定理。之后依次评估：

1. 紧区间一致风险导数；
2. 唯一、非退化最优解的扰动；
3. matched-spectrum 剂量律；
4. 固定正则化汇率；
5. 全局单峰和弃用源数据结论。

任一环节未闭合，就只保留已证明的固定 \(\lambda\) 或零点导数结果，不进入 P2 大规模实验扩张。

**执行状态**：固定 \(\lambda\) 风险公式、好/坏事件证明、22 个条件风险单元和 3 个真实估计器单元均已通过，论文已加入 T1-c。完整推导见 `P1-a_full_ridge_固定lambda二阶风险推导.md`。下一阶段尚未开始：一致风险导数与优化后风险控制。
