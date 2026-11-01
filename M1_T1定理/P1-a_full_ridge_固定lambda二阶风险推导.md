# 固定 \(\lambda\) full-ridge 二阶风险推导与核验

## 1. 定理结论

固定特征维度 \(p\)、\(\mu>0\) 和 \(\rho=n_h/n_r\)。两域协方差在共同基下为 \(\Sigma_r,\Sigma_h\)，设计为高斯，标签噪声零均值、方差有限、与设计及其他样本独立，并具有有限四阶矩。则实际 full-ridge 目标风险满足

\[
\boxed{R_{\mathrm{full}}(\lambda;n_r)
=R_0(\lambda)+\frac{R_1(\lambda)}{n_r}+o(n_r^{-1}).}
\]

下述证明对固定 \(\lambda\in[0,1]\) 成立；由于所有系数连续且集中事件可以统一选取，风险展开在 \(\lambda\in[0,1]\) 上一致成立。这里的“一致”只针对风险，不自动包括对 \(\lambda\) 求导或优化后风险。

## 2. 记号与精确风险分解

定义

\[
G_e=\frac{X_e^\top X_e}{n_e},\qquad Z_e=G_e-\Sigma_e,
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

估计误差为

\[
\hat\beta-\beta^*=B(v+\lambda Z_h\delta+\eta),
\]

其中

\[
\eta=\frac{1-\lambda}{n_r}X_r^\top\varepsilon_r
+\frac{\lambda}{n_h}X_h^\top\varepsilon_h.
\]

先对标签噪声取条件期望，交叉项消失，并且

\[
\mathbb E[\eta\eta^\top\mid X_r,X_h]
=\frac{(1-\lambda)^2\sigma_r^2}{n_r}G_r
+\frac{\lambda^2\sigma_h^2}{n_h}G_h=:N.
\]

所以真实风险严格等于

\[
R_{\mathrm{full}}
=\mathbb E[(v+\lambda Z_h\delta)^\top B\Sigma_rB(v+\lambda Z_h\delta)]
+\mathbb E\operatorname{tr}\{\Sigma_rBNB\}.
\]

后文分别展开确定偏差、失配平方、失配交叉和标签噪声四项。

## 3. Wishart 矩与 resolvent 展开

对任意确定矩阵 \(M\)，实高斯设计满足

\[
\mathbb E[Z_eMZ_e]=\frac1{n_e}\mathcal T_e(M),
\qquad
\mathcal T_e(M)=\Sigma_eM\Sigma_e+\operatorname{tr}(\Sigma_eM)\Sigma_e.
\]

因为 \(n_h=\rho n_r\)，

\[
\mathbb E[ZMZ]=\frac1{n_r}\mathcal T(M),
\qquad
\mathcal T=(1-\lambda)^2\mathcal T_r+\frac{\lambda^2}{\rho}\mathcal T_h.
\]

在好事件 \(\|Z\|\le\mu/2\) 上，

\[
B=R-RZR+RZRZR+O_p(\|Z\|^3).
\]

令

\[
S=R\Sigma_rR,\qquad U=\mathcal T(R).
\]

将两个 \(B\) 的二阶项和两个一阶项的交叉项全部保留，得到

\[
\mathbb E[B\Sigma_rB]=S+\frac{C}{n_r}+o(n_r^{-1}),
\]

\[
\boxed{C=RUS+SUR+R\mathcal T(S)R.}
\]

这里不需要样本 Gram 对角，也没有使用对角化估计器的 feedthrough 独立性。

## 4. 四项系数

### 4.1 确定偏差项

由上一节直接得到

\[
\mathbb E[v^\top B\Sigma_rBv]
=v^\top Sv+\frac{v^\top Cv}{n_r}+o(n_r^{-1}).
\]

因此人口主项为

\[
\boxed{R_0=v^\top Sv.}
\]

### 4.2 失配波动平方

两个显式 \(Z_h\) 已经贡献 \(n_h^{-1}\)。因此 \(B\) 只需保留人口项 \(R\)，任何额外波动都进入更高阶：

\[
\mathbb E[\lambda^2\delta^\top Z_hB\Sigma_rBZ_h\delta]
=\frac{\lambda^2}{\rho n_r}
\operatorname{tr}\{S\mathcal T_h(\delta\delta^\top)\}
+o(n_r^{-1}).
\]

即使 \(v=0\) 使人口主项完全抵消，这一项通常仍严格为正。

### 4.3 失配与 resolvent 的交叉项

虽然 \(\mathbb E Z_h=0\)，但 \(B\) 与 \(Z_h\) 相关，不能删除交叉项。每个 \(B\) 只取一阶波动，且只有人域部分能与显式 \(Z_h\) 配对：

\[
\mathbb E[B\Sigma_rBZ_h]
=-\frac{\lambda}{\rho n_r}
\left[R\mathcal T_h(S)+S\mathcal T_h(R)\right]
+o(n_r^{-1}).
\]

乘以原风险中的 \(2\lambda v^\top(\cdot)\delta\)，得到

\[
2\lambda\mathbb E[v^\top B\Sigma_rBZ_h\delta]
=-\frac{2\lambda^2}{\rho n_r}
v^\top[R\mathcal T_h(S)+S\mathcal T_h(R)]\delta
+o(n_r^{-1}).
\]

负号来自 \(\partial A^{-1}\) 中的负号，因子 2 来自向量平方的交叉项。

### 4.4 标签噪声

\(N\) 本身已经是 \(n_r^{-1}\) 阶，所以 \(B\) 和 \(G_e\) 只保留人口主项：

\[
\mathbb E\operatorname{tr}\{\Sigma_rBNB\}
=\frac{(1-\lambda)^2\sigma_r^2}{n_r}\operatorname{tr}(S\Sigma_r)
+\frac{\lambda^2\sigma_h^2}{\rho n_r}\operatorname{tr}(S\Sigma_h)
+O(n_r^{-2}).
\]

噪声与 resolvent 的进一步相关修正属于 \(n_r^{-2}\)，不影响本定理。

## 5. 完整二阶系数

合并四项：

\[
\boxed{
\begin{aligned}
R_1={}&v^\top Cv
+\frac{\lambda^2}{\rho}\operatorname{tr}\{S\mathcal T_h(\delta\delta^\top)\}\\
&-\frac{2\lambda^2}{\rho}v^\top[R\mathcal T_h(S)+S\mathcal T_h(R)]\delta\\
&+(1-\lambda)^2\sigma_r^2\operatorname{tr}(S\Sigma_r)
+\frac{\lambda^2\sigma_h^2}{\rho}\operatorname{tr}(S\Sigma_h).
\end{aligned}}
\]

最后两项正是 T1-b 中的人口标签噪声；前三项是完整二阶风险必须新增的 Gram 和失配修正。

在共同对角谱基下，\(R_0+R_1/n_r\) 的人口部分恢复

\[
\sum_j\frac{a_j}{D_j^2}
\left[\frac{(1-\lambda)^2\sigma_r^2a_j}{n_r}
+\frac{\lambda^2\sigma_h^2b_j}{n_h}
+(\lambda b_j\delta_j-\mu\beta_j^*)^2\right],
\]

但完整 \(R_1\) 还包含非对角样本 Gram 引起的矩阵迹项，不能由简单替换有效噪声得到。

## 6. 好事件、坏事件与一致性

在好事件上，三阶 resolvent 余项的算子范数由 \(\|Z\|^3\) 和 \(\mu^{-1}\) 的固定幂次控制；其期望为 \(O(n_r^{-3/2})=o(n_r^{-1})\)。所有被丢弃的显式三阶或四阶 Wishart 配对同样为 \(o(n_r^{-1})\)。

坏事件有两项：

1. 实际风险在坏事件上的贡献；
2. 被扣除的确定近似在坏事件上的贡献。

第二项由指数小的坏事件概率直接控制。第一项用 Cauchy–Schwarz：\(\|B\|\le\mu^{-1}\)，高斯设计和四阶噪声矩给出实际风险的有限二阶矩，因此其贡献也为指数小量。两项都为 \(o(n_r^{-1})\)。

对 \(\lambda\in[0,1]\)，可统一用 \(\|Z_r\|,\|Z_h\|\le\mu/2\) 定义好事件。所有系数在闭区间上连续，故有相同阶的一致余项。端点 \(\lambda=0,1\) 也可直接代入同一公式。

这是一致**风险**展开。对 \(\lambda\) 求导后的一致余项、优化后的 \(\lambda_n^*\) 和最优风险仍需独立证明。

## 7. 与零点导数定理的一致性

对 \(R_0(\lambda)\) 在零点求导得到

\[
R_0'(0^+)=H_{\mathrm{full}}.
\]

对 \(R_1(\lambda)\) 在零点求导得到

\[
R_1'(0^+)=-K_{\mathrm{full}},
\]

其中 \(H_{\mathrm{full}},K_{\mathrm{full}}\) 正是 `P1-a_full_ridge_零点导数推导.md` 中独立推导的系数。复步长解析求导核验中，\(p=1,2\) 的一致性误差不超过 \(4.44\times10^{-16}\)。

因此，固定 \(\lambda\) 二阶风险和零点导数不是两个互不相关的手工公式，而是相互一致的风险展开。

## 8. 数值核验

复现命令：

```bash
cd /mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/M1_T1定理
python verify_full_ridge_second_order.py --reps 20000 --direct-reps 5000
```

结果文件：`full_ridge_second_order_risk_results.json`。

覆盖范围：

- 一维人口主项精确抵消例；
- \(p=2\) 正向和反向失配；
- 谱系 A 的 \(p=20\) 配置；
- \(\lambda=0,0.1,0.3,0.5,0.8,1\)；
- \(n_r=1000,4000\)，固定 \(n_h/n_r\)；
- 22 个设计级条件风险单元，每个 20000 次重复；
- 3 个直接调用 `gen_two_domain` 和 `e2_full_ridge` 的真实风险单元，每个 5000 次重复。

22 个条件风险单元的“模拟均值减理论预测”全部在 \(\pm2.13\) 个 Monte Carlo 标准误内。三个真实估计器对照分别在 \(-0.78,-0.51,-1.12\) 个标准误处。紧凑 resolvent 公式、独立逐项配对和零点导数一致性检查的最大误差为 \(7.11\times10^{-15}\)。

一维抵消例中 \(R_0=0\)，\(R_1=0.0165289256\)。在 \(n_r=1000\) 时，理论风险为 \(1.65289\times10^{-5}\)，条件模拟为 \(1.65923\times10^{-5}\)，首个非零波动项得到正确恢复。

旧 `MSE_full,num` 也已单独比较：在这 22 个单元上，其相对误差绝对值的中位数为 0.0382%。它在这些大样本单元上可以是很好的数值近似，但这不改变其身份——旧 surrogate 不是已证明对象，新 \(R_0+R_1/n_r\) 才是本记录推导的定理。逐单元结果见 `full_ridge_second_order_vs_legacy_surrogate.json`，复现脚本为 `compare_full_ridge_surrogate.py`。

## 9. 仍未完成的结论

本阶段已经完成固定 \(p\) 的完整二阶风险，不只是一个数值 surrogate。

仍需单独研究：

1. 风险对 \(\lambda\) 的导数及其一致 \(o(n_r^{-1})\) 余项；
2. 最优混合比例和最优风险的扰动，尤其唯一性和非退化曲率；
3. matched-spectrum 剂量律是否只在领先阶为双曲线；
4. 固定正则化汇率的存在性、有限/无限边界及一阶修正；
5. 全局单峰、弃用源数据和比例区间 \(p\asymp n\)。

因此，下一步应做“风险导数与优化后风险”专项推导，而不是直接把旧剂量律、汇率公式改名为 full-ridge 定理。
