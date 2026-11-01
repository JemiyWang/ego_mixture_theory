"""v12：κ_ex 对谱差 Δα = α_r − α_h 的数值趋势（单调性仅为观察，未作为定理证明）。

固定 n_r=1000, n_h/n_r=10, μ=0.05, ‖δ‖=0.3（白色对齐失配，远离有害相区），
α_h ∈ {0.6,0.8,1.0,1.2}，α_r 扫描，算扩展 κ_ex 与 λ*。
口径（W5 修订）：有限 κ>1 是正常汇率；无有限解时返回 +∞，不截断到 1。
min(κ,1) 仅作为 kappa_ex_cap1 展示量另行保存；固定 μ 地板不等于永久覆盖缺失。
"""
import numpy as np
import json

P = 1000
SIG_R2, SIG_H2 = 1.0, 4.0
MU = 0.05
S = 1.0
N_R, RATIO = 1000, 10.0
LAM_GRID = np.linspace(0, 1, 401)

jj = np.arange(1, P + 1, dtype=float)
beta = jj ** (-(S + 0.5))

def feed(spec, b):
    e = spec * b**2
    return e.sum() - e

def make_delta():
    d = np.ones(P) / np.sqrt(P) * 0.3   # 对齐（正 c）白色失配
    return d

def R_mix(lam, lr, lh, delta, n_r, n_h):
    D = (1 - lam) * lr + lam * lh + MU
    N0 = lam * lh * delta - MU * beta
    Vr = (SIG_R2 + feed(lr, beta)) * lr / n_r
    Vh = (SIG_H2 + feed(lh, beta + delta)) * lh / n_h
    return float((lr * (N0**2 + (1 - lam)**2 * Vr + lam**2 * Vh) / D**2).sum())

def R_pure(lr, delta_unused, n_eff):
    D0 = lr + MU
    Vr = (SIG_R2 + feed(lr, beta)) * lr / n_eff
    return float((lr * (MU**2 * beta**2 + Vr) / D0**2).sum())

def kappa_ex(lr, lh, delta, n_r, n_h):
    curve = np.array([R_mix(l, lr, lh, delta, n_r, n_h) for l in LAM_GRID])
    i_star = int(np.argmin(curve))
    Rb = curve[i_star]
    lam_star = LAM_GRID[i_star]
    floor = R_pure(lr, None, np.inf)  # 固定 μ 的纯机器人收缩地板
    if Rb <= floor + 1e-15:
        return np.inf, lam_star       # 定义方程无有限解，不截断到 1
    lo, hi = 0.0, 1.0
    while R_pure(lr, None, n_r + hi * n_h) > Rb:
        hi *= 2.0
        if hi > 1e16:
            return np.inf, lam_star   # 数值兜底；理论上 Rb>floor 时不应到达
    for _ in range(80):
        mid = (lo + hi) / 2
        if R_pure(lr, None, n_r + mid * n_h) > Rb:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, lam_star

delta = make_delta()
results = []
for ah in [0.6, 0.8, 1.0, 1.2]:
    for ar in np.arange(ah, ah + 2.01, 0.25):
        lr = jj ** (-ar); lh = jj ** (-ah)
        k, ls = kappa_ex(lr, lh, delta, N_R, int(N_R * RATIO))
        results.append(dict(alpha_h=ah, alpha_r=round(float(ar), 2),
                            delta_alpha=round(float(ar - ah), 2),
                            kappa_ex=k, lam_star=ls))

print(f"{'α_h':>4} {'α_r':>5} {'Δα':>5} | {'κ_ex':>7} {'λ*':>6}")
for r in results:
    k_text = "inf" if np.isinf(r["kappa_ex"]) else f"{r['kappa_ex']:.4f}"
    print(f"{r['alpha_h']:>4} {r['alpha_r']:>5} {r['delta_alpha']:>5} | {k_text:>7} {r['lam_star']:.3f}")

# 单调性仅对有限的扩展汇率计算；无穷值单独计数。cap1 只是展示量，不进入定义。
from scipy.stats import spearmanr
for ah in [0.6, 0.8, 1.0, 1.2]:
    sub = [r for r in results if r["alpha_h"] == ah]
    finite = [r for r in sub if np.isfinite(r["kappa_ex"])]
    n_inf = len(sub) - len(finite)
    if len(finite) >= 3:
        rho, pval = spearmanr([r["delta_alpha"] for r in finite], [r["kappa_ex"] for r in finite])
        print(f"α_h={ah}: finite Spearman(Δα, κ_ex) = {rho:.3f} (p={pval:.2e}); ∞ {n_inf}/{len(sub)}")
    else:
        print(f"α_h={ah}: finite values {len(finite)}/{len(sub)}, Spearman not reported; ∞ {n_inf}/{len(sub)}")

json_results = []
for r in results:
    k = r["kappa_ex"]
    json_results.append(dict(
        alpha_h=r["alpha_h"], alpha_r=r["alpha_r"], delta_alpha=r["delta_alpha"],
        kappa_ex=None if np.isinf(k) else round(float(k), 4),
        kappa_ex_infinite=bool(np.isinf(k)),
        kappa_ex_cap1=round(min(float(k), 1.0), 4),
        lam_star=round(float(r["lam_star"]), 3)))
for output_path in [
    "/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l1_kappa_ex_curve.json",
    "/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/M1_T1定理/l1_kappa_ex_curve.json",
]:
    json.dump(json_results, open(output_path, "w"), indent=1)
    print(f"saved {output_path}")
