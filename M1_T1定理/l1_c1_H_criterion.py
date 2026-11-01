"""v11：H 判据的数值确认——失配阈值与 n_r^crit ∝ 1/H。

H := lim_{n_r→∞} R̄'(0⁺)（n_h/n_r 固定，B 项消失）= A∞ − C∞：
  A∞ = -2μ Σ_j a_j β*_j b_j δ_j / D0²     （有符号对齐贡献；本实验 δ_j<0、β*_j>0）
  C∞ = 2μ² Σ_j a_j β*_j² (b_j−a_j) / D0³    （更宽人域谱的收缩减免收益，与 δ 无关）
H>0 ⟹ 存在有限 n_r^crit（有害相区非空）；H≤0 ⟹ 边际混合恒有益。

验证：① 由 v9 已知 ‖δ‖=0.3/0.5 无临界、1.0/2.0 有临界 → 预测阈值 δ* ≈ 0.674；
② n_r^crit ∝ 1/H 检验（v9 两点：9726 @δ=1, 2434 @δ=2；H 比 2.28e-3 vs 9.26e-3，1/H 比 ≈ 4.06 vs 实测 4.00）。
"""
import numpy as np
import json

S_SMOOTH, SIG_R2, SIG_H2 = 1.0, 1.0, 4.0
P, MU = 1000, 0.05
jj = np.arange(1, P + 1, dtype=float)
beta = jj ** (-(S_SMOOTH + 0.5))
a = jj ** -2.0; b = jj ** -0.9
D0 = a + MU

def H_of(delta_norm):
    delta = -delta_norm * np.ones(P) / np.sqrt(P)
    A_inf = float((a * (-2 * MU * beta * b * delta) / D0**2).sum())
    C_inf = float((a * (2 * MU**2 * beta**2 * (b - a)) / D0**3).sum())
    return A_inf, C_inf, A_inf - C_inf

print("=== H 判据 ===")
for dn in [0.3, 0.5, 0.6, 0.674, 0.75, 1.0, 2.0]:
    A_inf, C_inf, H = H_of(dn)
    print(f"‖δ‖={dn:>5}: A∞={A_inf:+.4e} C∞={-C_inf:+.4e} H={H:+.4e} → {'有害相区存在' if H>0 else '恒有益'}")

# n_r^crit ∝ 1/H 检验（v9 数据点复算）
def terms_dR0(n_r, delta_norm, ratio=10.0):
    delta = -delta_norm * np.ones(P) / np.sqrt(P)
    Fr = float((a * beta**2).sum())
    Vr = (SIG_R2 + Fr) * a / n_r
    A_ = float((a * (-2 * MU * beta * b * delta) / D0**2).sum())
    B_ = float((a * (-2 * Vr) / D0**2).sum())
    C_ = float((a * (-2 * (Vr + MU**2 * beta**2) * (b - a)) / D0**3).sum())
    return A_ + B_ + C_

def find_crit(delta_norm):
    lo, hi = 2.0, 7.0
    if terms_dR0(10**hi, delta_norm) < 0: return np.inf
    for _ in range(45):
        mid = (lo + hi) / 2
        if terms_dR0(10**mid, delta_norm) > 0: hi = mid
        else: lo = mid
    return 10**((lo + hi) / 2)

print("\n=== n_r^crit vs 1/H ===")
for dn in [0.75, 1.0, 2.0]:
    _, _, H = H_of(dn)
    nc = find_crit(dn)
    print(f"‖δ‖={dn}: H={H:+.4e}, n_r^crit={nc:.3e}, n_r^crit·H={nc*H:.3f}")
print("（若 n_r^crit·H ≈ 常数 ⟹ n_r^crit ∝ 1/H 成立）")
