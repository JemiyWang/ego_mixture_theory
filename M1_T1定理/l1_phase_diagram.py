"""v7：理论相图（纯公式，无 MC）——用 v5 已验证公式计算 λ* 相图、边际符号条件、汇率预览。

逐坐标（对角化估计量、馈通修正有效噪声，v5 验证精度 1–2%）：
  D_j(λ) = (1−λ)a_j + λ b_j + μ        （a_j=λr_j, b_j=λh_j）
  N0_j(λ) = λ b_j δ_j − μ β*_j
  VarN_j(λ) = (1−λ)² Vr_j + λ² Vh_j,  Vr_j = (σ_r²+F_r) a_j/n_r,  Vh_j = (σ_h²+F_h) b_j/n_h
  MSE_j = (N0_j² + VarN_j)/D_j²，R̄(λ) = Σ_j a_j MSE_j（Σ_r 加权）

C1（逐坐标边际收益，解析导数，见 M1 文档 §3 推论 C1）：
  MSE_j'(0) = [−2μβ*_j b_j δ_j − 2Vr_j]/D₀² − 2[Vr_j + μ²β*_j²](b_j−a_j)/D₀³
"""
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = 1000
SIG_R2, SIG_H2 = 1.0, 4.0
MU = 0.05
S_SMOOTH = 1.0
LAM_GRID = np.linspace(0, 1, 501)

jj = np.arange(1, P + 1, dtype=float)
beta = jj ** (-(S_SMOOTH + 0.5))

SPECTRA = dict(A=(2.0, 0.9), B=(1.2, 1.0))
PROFILES = dict(white=np.ones(P), lowfreq=jj ** -1.0, highfreq=(jj / P) ** 2)

def feed(lam_spec, b):
    e = lam_spec * b**2
    return e.sum() - e

def risk_bar(lam, lr, lh, delta, n_r, n_h):
    a, b = lr, lh
    D = (1 - lam) * a + lam * b + MU
    N0 = lam * b * delta - MU * beta
    Vr = (SIG_R2 + feed(lr, beta)) * a / n_r
    Vh = (SIG_H2 + feed(lh, beta + delta)) * b / n_h
    VarN = (1 - lam)**2 * Vr + lam**2 * Vh
    return float((a * (N0**2 + VarN) / D**2).sum())

def dR0_analytic(lr, lh, delta, n_r, n_h):
    """C1：R̄'(0⁺) 的逐坐标解析和。"""
    a, b = lr, lh
    D0 = a + MU
    Vr = (SIG_R2 + feed(lr, beta)) * a / n_r
    term1 = (-2 * MU * beta * b * delta - 2 * Vr) / D0**2
    term2 = -2 * (Vr + MU**2 * beta**2) * (b - a) / D0**3
    return float((a * (term1 + term2)).sum())

def analyze(lr, lh, delta, n_r, n_h):
    R = np.array([risk_bar(l, lr, lh, delta, n_r, n_h) for l in LAM_GRID])
    d0_ana = dR0_analytic(lr, lh, delta, n_r, n_h)
    d0_fd = (R[1] - R[0]) / LAM_GRID[1]
    lam_star = float(LAM_GRID[np.argmin(R)])
    # 汇率 κ_ex：R̄(λ*; n_r, n_h) = R̄(0; n_r + κ n_h) 解 κ（二分）
    R_best = R.min()
    def R_pure_robot(n_r_eff):
        a = lr
        D0 = a + MU
        Vr = (SIG_R2 + feed(lr, beta)) * a / n_r_eff
        return float((a * ((MU * beta)**2 + Vr) / D0**2).sum())
    lo, hi = 0.0, 1.0
    if R_pure_robot(n_r + n_h) > R_best:
        kappa = 1.0     # 即使全部折算也不够好 → κ 上界 1（人域数据优于折算）
    else:
        for _ in range(60):
            mid = (lo + hi) / 2
            if R_pure_robot(n_r + mid * n_h) < R_best: hi = mid
            else: lo = mid
        kappa = (lo + hi) / 2
    return dict(dR0_ana=d0_ana, dR0_fd=float(d0_fd), lam_star=lam_star,
                R0=float(R[0]), R_best=float(R_best), kappa_ex=float(kappa))

results = {}
# 与 v2 经验对照的两个基准配置
for sp_name, (ar, ah) in SPECTRA.items():
    lr = jj ** (-ar); lh = jj ** (-ah)
    for pf_name, dvec in PROFILES.items():
        delta = dvec / np.linalg.norm(dvec) * 0.5
        r = analyze(lr, lh, delta, 1000, 6000)
        results[f"{sp_name}|{pf_name}|v2config"] = r
        print(f"{sp_name}|{pf_name}: dR0 ana/fd = {r['dR0_ana']:.2e}/{r['dR0_fd']:.2e}, "
              f"λ*={r['lam_star']:.3f}, κ_ex={r['kappa_ex']:.3f}")

# 相图：谱系 A，三失配形态，n_r × (n_h/n_r) 网格
NR_GRID = [100, 300, 1000, 3000, 10000, 30000]
RHO_GRID = [1, 3, 10, 30, 100]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (pf_name, dvec) in zip(axes, PROFILES.items()):
    delta = dvec / np.linalg.norm(dvec) * 0.5
    lr = jj ** (-2.0); lh = jj ** (-0.9)
    Z = np.zeros((len(NR_GRID), len(RHO_GRID)))
    S = np.full_like(Z, np.nan)
    for i, n_r in enumerate(NR_GRID):
        for k, rho in enumerate(RHO_GRID):
            r = analyze(lr, lh, delta, n_r, int(n_r * rho))
            Z[i, k] = r["lam_star"]
            S[i, k] = np.sign(r["dR0_ana"])
    im = ax.pcolormesh(np.log10(RHO_GRID), np.log10(NR_GRID), Z, cmap="viridis")
    # λ*=0 的有害相区边界
    ax.contour(np.log10(RHO_GRID), np.log10(NR_GRID), Z, levels=[0.001],
               colors="red", linewidths=2)
    ax.set_title(f"Spectra A | mismatch={pf_name}", fontsize=11)
    ax.set_xlabel("log10(n_h/n_r)"); ax.set_ylabel("log10(n_r)")
    fig.colorbar(im, ax=ax, label="λ*")
fig.suptitle("Theory phase diagram (v5 formula, no MC): lambda* over (n_r, n_h/n_r, mismatch profile); red line = boundary of harmful region (lambda*=0)")
fig.tight_layout()
fig.savefig("/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l1_phase_diagram.png", dpi=130)

with open("/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l1_phase_results.json", "w") as f:
    json.dump(dict(v2config=results,
                   note="相图网格数据在 PNG 生成过程中计算；基准配置含 κ_ex"),
              f, indent=1, default=str)
print("saved l1_phase_diagram.png + l1_phase_results.json")
