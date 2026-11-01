"""v10：C1 失败的数值归因——R̄'(0) 的 A/B/C 逐项分解与真实标度。

v9 用 λ∈[0,0.02] 线性拟合估计 R̄'(0)，混入二阶项；且 μ 区间对应的 j_μ ∈ [1.8,17.7]
非渐近。本脚本：
1. 用精确解析导数（非拟合）；
2. 分解 A（失配交叉）/B（方差）/C（谱宽）三项的和式值；
3. 在大 j_μ 渐近区（p=20000, μ∈[1e-5, 3e-4]）重测 n_r^crit 的 μ/δ 标度。
"""
import numpy as np
import json

S_SMOOTH = 1.0
SIG_R2, SIG_H2 = 1.0, 4.0

def terms(p, mu, n_r, n_h, delta_norm, ar=2.0, ah=0.9):
    jj = np.arange(1, p + 1, dtype=float)
    beta = jj ** (-(S_SMOOTH + 0.5))
    a = jj ** (-ar); b = jj ** (-ah)
    delta = -delta_norm * np.ones(p) / np.sqrt(p)        # 白色反对齐
    D0 = a + mu
    Fr = float((a * beta**2).sum()); Fh = float((b * (beta + delta)**2).sum())
    Vr = (SIG_R2 + Fr) * a / n_r
    # A：失配交叉项（μβ*bδ）；B：方差项；C：谱宽项——均为对 R̄'(0) 的贡献（带 a_j 权重）
    A = float((a * (-2 * mu * beta * b * delta) / D0**2).sum())
    Bterm = float((a * (-2 * Vr) / D0**2).sum())
    Cterm = float((a * (-2 * (Vr + mu**2 * beta**2) * (b - a)) / D0**3).sum())
    return dict(dR0=A + Bterm + Cterm, A=A, B=Bterm, C=Cterm, Fh=Fh)

# --- 诊断 1：v9 配置区（p=1000, μ=0.05, n_h/n_r=10）‖δ‖=0.3 为何无相变 ---
print("=== 诊断1：‖δ‖=0.3, μ=0.05, p=1000，R̄'(0) 随 n_r ===")
for n_r in [1e3, 1e4, 3e4, 1e5, 3e5, 1e6, 1e7]:
    r = terms(1000, 0.05, n_r, 10 * n_r, 0.3)
    print(f"n_r={n_r:>9.0f}: dR0={r['dR0']:+.4e}  (A={r['A']:+.3e} B={r['B']:+.3e} C={r['C']:+.3e})")

# --- 诊断 2：渐近区（p=20000, 大 j_μ）的 μ 标度 ---
print("\n=== 诊断2：p=20000, ‖δ‖=1.0, n_h/n_r=10，渐近区 n_r^crit vs μ ===")
def find_crit(p, mu, delta_norm, ratio=10.0):
    lo, hi = 2.0, 8.0   # log10 n_r ∈ [1e2, 1e8]
    if terms(p, mu, 10**lo, 10**lo * ratio, delta_norm)["dR0"] > 0: return 10**lo
    if terms(p, mu, 10**hi, 10**hi * ratio, delta_norm)["dR0"] < 0: return np.inf
    for _ in range(45):
        mid = (lo + hi) / 2
        if terms(p, mu, 10**mid, 10**mid * ratio, delta_norm)["dR0"] > 0: hi = mid
        else: lo = mid
    return 10**((lo + hi) / 2)

mu_grid = np.logspace(-5, -3.5, 7)   # j_μ = μ^{-0.5} ∈ [56, 316]
ncs = [find_crit(20000, mu, 1.0) for mu in mu_grid]
for mu, nc in zip(mu_grid, ncs):
    print(f"μ={mu:.2e} (j_μ={mu**-0.5:.0f}): n_r^crit={nc:.3e}")
mask = np.isfinite(ncs)
slope = np.polyfit(np.log10(mu_grid[mask]), np.log10(ncs)[mask], 1)[0] if mask.sum() > 2 else np.nan
print(f"n_r^crit ∝ μ^{slope:.3f}")

# --- 诊断 3：同区的 δ 标度 ---
print("\n=== 诊断3：p=20000, μ=1e-4.5，n_r^crit vs ‖δ‖ ===")
dn_grid = np.array([0.3, 0.5, 1.0, 2.0, 4.0])
ncs_d = [find_crit(20000, 10**-4.5, dn) for dn in dn_grid]
for dn, nc in zip(dn_grid, ncs_d):
    print(f"‖δ‖={dn}: n_r^crit={nc:.3e}")
mask = np.isfinite(ncs_d)
slope_d = np.polyfit(np.log10(dn_grid[mask]), np.log10(ncs_d)[mask], 1)[0] if mask.sum() > 2 else np.nan
print(f"n_r^crit ∝ ‖δ‖^{slope_d:.3f}")

json.dump(dict(diag2_mu=list(map(float, mu_grid)), diag2_nc=[float(x) for x in ncs],
               diag2_slope=float(slope), diag3_dn=list(map(float, dn_grid)),
               diag3_nc=[float(x) for x in ncs_d], diag3_slope=float(slope_d)),
          open("/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l1_c1_diag.json", "w"), indent=1)
print("saved l1_c1_diag.json")
