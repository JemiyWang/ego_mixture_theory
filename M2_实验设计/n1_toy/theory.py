"""v5 逐坐标公式(忠实移植 M1_T1定理/l1_com_v5.py 的 pred_v5)+ Σ_r 加权总风险。

逐坐标 MSE 公式(v5 口径,w_e 为逐样本权重 w_r=(1-λ)/n_r、w_h=λ/n_h):
  D     = (1-λ)lr + λlh + μ                                  对角权阵期望
  N0    = λ·lh·δ - μβ*                                       偏差分子(失配 + 岭收缩)
  VarN  = (wh·δ)²·2n_h·lh²                (人域失配涨落)
        + wr²(σ_r²+F_r)·n_r·lr + wh²(σ_h²+F_h)·n_h·lh   (噪声 + 馈通串扰,σ̃²=σ²+F)
  CovND = wh²δ·2n_h·lh²                                      N̂ 与 D̂ 的协方差修正
  VarD  = 2((1-λ)²lr²/n_r + λ²lh²/n_h)                       分母涨落(二阶修正)
  MSE_j = N0²/D² + VarN/D² - 4N0·CovND/D³ + 3N0²·VarD/D⁴
馈通有效噪声:F_e,j = Σ_{k≠j} λe_k β_e,k²(坐标外信号对坐标 j 而言就是噪声),β_h=β*+δ。

full-ridge 数值修正模型:把 σ̃² 换回真实 σ²(去掉馈通项),其余同 v5——供 E2 对照(仅报告,不做判据)。
注意:这是 MSE_full,num 计算模型(主阶 T1-b + 对角展开的二阶修正),不是已证明的 full-ridge 二阶定理。

总风险:R̂(λ) = Σ_j a_j·MSE_j,默认 a_j = lr_j(Σ_r 加权 = 机器人域测试风险)。
"""
import numpy as np


def feedthrough(lam_spec, beta):
    """F_j = Σ_{k≠j} λ_k β_k²"""
    e = lam_spec * beta**2
    return e.sum() - e          # 去掉自身坐标


def mse_per_coord(lam, lam_r, lam_h, delta, beta_star, beta_h,
                  n_r, n_h, sig_r2, sig_h2, mu, use_feedthrough=True):
    """v5 逐坐标 MSE(pred_v5 忠实移植);use_feedthrough=False 即 full-ridge 数值修正模型。"""
    wr, wh = (1.0 - lam) / n_r, lam / n_h
    D = (1.0 - lam) * lam_r + lam * lam_h + mu
    N0 = lam * lam_h * delta - mu * beta_star
    F_r = feedthrough(lam_r, beta_star) if use_feedthrough else 0.0   # 机器人域串扰能
    F_h = feedthrough(lam_h, beta_h) if use_feedthrough else 0.0      # 人域串扰能
    # Var(N̂):失配涨落 + (噪声+串扰) 的期望
    VarN = (wh * delta)**2 * 2 * n_h * lam_h**2 \
        + wr**2 * (sig_r2 + F_r) * n_r * lam_r \
        + wh**2 * (sig_h2 + F_h) * n_h * lam_h
    CovND = wh**2 * delta * 2 * n_h * lam_h**2
    VarD = 2 * ((1.0 - lam)**2 * lam_r**2 / n_r + lam**2 * lam_h**2 / n_h)
    return (N0**2 / D**2 + VarN / D**2 - 4 * N0 * CovND / D**3
            + 3 * N0**2 * VarD / D**4)


def mse_v5(lam, *args):
    return mse_per_coord(lam, *args, use_feedthrough=True)


def mse_t1b(lam, *args):
    """full-ridge 数值修正模型:σ̃² → 真实 σ²(无馈通),用于 E2 对照。"""
    return mse_per_coord(lam, *args, use_feedthrough=False)


def risk_from_mse(mse, weights):
    """R̂ = Σ_j a_j·MSE_j"""
    return float(np.sum(weights * mse))
