# P1-a 阶段记录：full-ridge 零点导数与边界可辨识性

## 1. 本记录解决什么

在固定维度 \(p\)、固定 \(\mu>0\)、高斯设计和共同谱基下，本文证明了标准 full-ridge 混合估计器的零点导数展开

\[
R_{\mathrm{full}}'(0^+;n_r)
=H_{\mathrm{full}}-\frac{K_{\mathrm{full}}}{n_r}+o(n_r^{-1}).
\]

这里 \(K_{\mathrm{full}}\) 不只是对 T1-b 人口风险求导得到的系数，还包含机器人样本 Gram 矩阵波动产生的 \(n_r^{-1}\) 修正。

本记录**没有**完成任意固定 \(\lambda\) 的完整二阶风险展开，也没有自动迁移全局最优比例、剂量律或汇率。

## 2. 精确零点导数

记 \(G=X_r^\top X_r/n_r\)，\(\Sigma=\Sigma_r\)，\(D_0=\Sigma+\mu I\)。在 \(\lambda=0\) 附近，

\[
A_\lambda=(1-\lambda)G+\lambda G_h+\mu I,\qquad
\hat\beta-\beta^*=A_\lambda^{-1}[-\mu\beta^*+\lambda G_h\delta+\eta_\lambda],
\]

其中 \(\eta_\lambda=(1-\lambda)X_r^\top\varepsilon_r/n_r+\lambda X_h^\top\varepsilon_h/n_h\)。写 \(B=(G+\mu I)^{-1}\)。对人域独立随机性取期望后，精确导数可以写成

\[
\begin{aligned}
R_{\mathrm{full}}'(0^+)
={}&-2\mu(\Sigma_h\delta)^\top\mathbb E[B\Sigma B]\beta^*\\
&-2\mu^2\beta^{*\top}\mathbb E[B\Sigma B(\Sigma_h-G)B]\beta^*\\
&-\frac{2\sigma_r^2}{n_r}\operatorname{tr}\Sigma\,\mathbb E[BGB]\\
&-\frac{2\sigma_r^2}{n_r}\operatorname{tr}\Sigma\,\mathbb E[B(\Sigma_h-G)BGB].
\end{aligned}
\]

推导时使用

\[
\partial_\lambda A^{-1}=-A^{-1}(G_h-G)A^{-1},\qquad
\mathbb E[\eta_0\eta_0^\top\mid X_r]=\sigma_r^2G/n_r.
\]

确定性偏差项与噪声项的分母导数符号不同，不能合并后随意换号。人域标签噪声和人域失配波动都带 \(\lambda^2\)，所以不对零点导数产生一阶贡献；人域 Gram 在零点导数中只通过其均值 \(\Sigma_h\) 出现。

## 3. Wishart 二阶矩与 resolvent 展开

令 \(Z=G-\Sigma\)。对任意确定矩阵 \(M\)，实高斯 Wishart 矩给出

\[
\mathbb E[ZMZ]=\frac1{n_r}\mathcal T(M),\qquad
\mathcal T(M)=\Sigma M\Sigma+\operatorname{tr}(\Sigma M)\Sigma.
\]

在 \(\|Z\|\le\mu/2\) 的好事件上，

\[
B=R-RZR+RZRZR+O_p(\|Z\|^3),\qquad R=D_0^{-1}.
\]

坏事件概率随固定 \(p\) 的 \(n_r\) 指数下降；逆矩阵有 \(\|A_\lambda^{-1}\|\le\mu^{-1}\)，并且模型具有所需有限矩，因此坏事件和 Taylor 余项均不进入所保留的阶数。对固定的所有参数，展开系数和余项在近零区间 \(\lambda\in[0,\varepsilon]\) 上一致控制；有界逆矩阵和有限矩也允许交换求导与期望。

## 4. 闭式系数

定义

\[
S=R\Sigma R,\qquad U=\mathcal T(R),\qquad M=\Sigma_h-\Sigma.
\]

二阶 resolvent 配对给出两个修正矩阵：

\[
\mathcal C=RUS+SUR+R\mathcal T(S)R,
\]

以及

\[
\mathcal L=\mathcal J(M)+R\mathcal T(S)R+2SUR,
\]

其中

\[
\begin{aligned}
\mathcal J(N)={}&RUSNR+SURNR+SNRUR\\
&+R\mathcal T(S)RNR+R\mathcal T(SNR)R+S\mathcal T(RNR)R.
\end{aligned}
\]

于是

\[
\mathbb E[B\Sigma B]=S+\frac{\mathcal C}{n_r}+o(n_r^{-1}),
\]

\[
\mathbb E[B\Sigma B(\Sigma_h-G)B]
=SMR+\frac{\mathcal L}{n_r}+o(n_r^{-1}).
\]

收集偏差项和噪声项：

\[
H_{\mathrm{full}}
=-2\mu(\Sigma_h\delta)^\top S\beta^*
-2\mu^2\beta^{*\top}SMR\beta^*,
\]

\[
K_0=2\sigma_r^2\operatorname{tr}\{\Sigma(R\Sigma R+R\Sigma RMR)\},
\]

\[
H_1=-2\mu(\Sigma_h\delta)^\top\mathcal C\beta^*
-2\mu^2\beta^{*\top}\mathcal L\beta^*,
\]

\[
\boxed{K_{\mathrm{full}}=K_0-H_1.}
\]

在论文的共同对角谱基下，这些人口阶量退化为

\[
H_{\mathrm{full}}
=-2\mu\sum_j\frac{a_jb_j\beta_j^*\delta_j}{(a_j+\mu)^2}
-2\mu^2\sum_j\frac{a_j\beta_j^{*2}(b_j-a_j)}{(a_j+\mu)^3},
\]

\[
K_0=2\sigma_r^2\sum_j\frac{a_j^2(b_j+\mu)}{(a_j+\mu)^3}.
\]

所以，**只对人口风险求导会漏掉 \(-H_1/n_r\)**。噪声方差的 Gram 修正自身是 \(n_r^{-2}\) 阶，不进入 \(K_{\mathrm{full}}\)；偏差sandwich中的 Gram 修正才是这里的 \(n_r^{-1}\) 新项。

## 5. 一维手算检查

当 \(p=1\) 时，写 \(a=\Sigma_r\)、\(b=\Sigma_h\)、\(d=a+\mu\)，则

\[
K_0=\frac{2\sigma_r^2a^2(b+\mu)}{d^3},
\]

\[
H_1=-\frac{12\mu a^3b\delta\beta^*}{d^4}
+\frac{12\mu^2a^3(a-\mu-2b)\beta^{*2}}{d^5}.
\]

这与直接展开标量 \(B=(G+\mu)^{-1}\) 所得的 \(\mathbb E B^2\)、\(\mathbb E B^3\)、\(\mathbb E GB^3\) 完全一致。它也说明：即使 \(a=b\) 使人口阶覆盖项为零，其 \(n_r^{-1}\) Gram 修正通常仍不为零。

## 6. 边界可辨识条件

令 \(t=1/n_r\)，渐近导数为 \(\bar g(t)=H_{\mathrm{full}}-K_{\mathrm{full}}t\)。

- 若 \(H_{\mathrm{full}}K_{\mathrm{full}}>0\)，则
  \[
  t^*=H_{\mathrm{full}}/K_{\mathrm{full}}>0,\qquad
  n_r^{\mathrm{crit,full}}=K_{\mathrm{full}}/H_{\mathrm{full}}.
  \]
  且 \(\bar g'(t^*)=-K_{\mathrm{full}}\ne0\)，所以零点两侧的符号在远离根任意固定相对距离时可辨识。
- 若 \(H_{\mathrm{full}}>0\) 但 \(K_{\mathrm{full}}\le0\)，不存在该公式给出的正临界样本量，不能继续套用 \(K/H\) 相图。
- 若 \(H_{\mathrm{full}}<0\)，大 \(n_r\) 的边际混合有益；只有 \(K_{\mathrm{full}}<0\) 时，上述公式才另外给出正根。
- 在根本身，主项为零而理论余项仍为 \(o(n_r^{-1})\)，故本定理不识别精确有限样本根，也不保证离散样本量上存在精确零点。它识别的是**符号翻转的渐近尺度和非退化区域的符号**。

特别地，\(K_{\mathrm{full}}\) 不再像旧对角化公式中的 \(K\) 那样显然为正，必须先计算其符号。

## 7. 数值核验

复现命令：

```bash
cd /mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/M1_T1定理
python verify_full_ridge_zero_derivative.py --reps 10000
```

结果文件：`full_ridge_zero_derivative_results.json`。脚本使用 Bartlett 分解精确生成高斯 Wishart 样本 Gram，并直接计算 full-ridge 的设计级条件导数；人域独立 Gram 已精确取期望。

| 配置 | \(n_r\) | 一阶预测 | Monte Carlo | MC 标准误 |
|---|---:|---:|---:|---:|
| \(p=1\) | 1000 | -0.034956629 | -0.034977578 | 3.51e-5 |
| \(p=1\) | 10000 | -0.033247729 | -0.033235472 | 1.07e-5 |
| \(p=2\) | 1000 | -0.072574612 | -0.072510463 | 7.50e-5 |
| \(p=2\) | 10000 | -0.069337150 | -0.069327678 | 2.33e-5 |
| 谱系 A，\(p=20\)，反向失配 | 1000 | +0.014813560 | +0.014761728 | 1.97e-5 |
| 同上 | 4000 | +0.024029178 | +0.024015679 | 1.03e-5 |
| 同上 | 16000 | +0.026333083 | +0.026332290 | 5.16e-6 |

紧凑矩阵公式与独立逐项配对展开的最大代数误差为 \(1.11\times10^{-16}\)。

另用现有 `gen_two_domain` 和 `e2_full_ridge` 直接生成标签并计算真实风险差分 \([R(10^{-4})-R(0)]/10^{-4}\)，各运行 5000 次：\(p=2\) 得到 \(-0.0723000\pm0.0003548\)（预测 \(-0.0725746\)），\(p=20\) 得到 \(0.0143334\pm0.0003422\)（预测 \(0.0148136\)）。两者分别在约 0.77 和 1.40 个标准误内。结果保存在 `full_ridge_direct_finite_difference_results.json`。该检查直接调用实际估计器，不只验证导数代数。

原 \(p=20\) 配置中：

\[
H_{\mathrm{full}}=0.0271010512,\qquad K_0=12.6413283671,
\]
\[
H_1=0.3538375404,\qquad K_{\mathrm{full}}=12.2874908267.
\]

早期用非零 \(\lambda\) 的风险曲线作二次/三次拟合得到约 \(0.0171\)，其中混入了曲率；它不是纯零点导数，不能用来否定上述展开。

## 8. 当前边界

已完成：固定 \(p\) 的零点导数、显式 \(n_r^{-1}\) 系数、非退化边界可辨识条件及数值核验。

仍需继续：固定 \(\lambda\in(0,1)\) 的完整二阶风险系数、坏事件和余项的详细常数表、优化后风险的独立控制，以及剂量律和汇率是否迁移。只有这些完成后，才能声称 P1-a 的固定 \(p\) 风险展开全部完成。
