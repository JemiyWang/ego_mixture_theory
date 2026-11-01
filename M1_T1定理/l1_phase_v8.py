"""v8：有害相区扫描——反对齐失配（c<0）× 大 ‖δ‖ × 大 n_r 网格。

同 v7 公式，但 δ 方向取负（anti-aligned：δ = −profile·norm，c = Σδ_jβ*_j < 0），
Δ ∈ {0.5, 1.0, 2.0}，n_r 到 1e5，标记 λ*≈0 的有害相区。
输出：相图 PNG（3 形态 × 3 Δ 强度）+ 边界数据 JSON。
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

PROFILES = dict(white=np.ones(P), lowfreq=jj ** -1.0, highfreq=(jj / P) ** 2)
DELTAS = [0.5, 1.0, 2.0]
NR_GRID = [100, 300, 1000, 3000, 10000, 30000, 100000]
RHO_GRID = [0.5, 1, 3, 10, 30, 100]

def feed(lam_spec, b):
    e = lam_spec * b**2
    return e.sum() - e

def risk_bar(lam, lr, lh, delta, n_r, n_h):
    a, b = lr, lh
    D = (1 - lam) * a + lam * b + MU
    N0 = lam * b * delta - MU * beta
    Vr = (SIG_R2 + feed(lr, beta)) * a / n_r
    Vh = (SIG_H2 + feed(lh, beta + delta)) * b / n_h
    return float((a * ((N0**2 + (1 - lam)**2 * Vr + lam**2 * Vh)) / D**2).sum())

def analyze(lr, lh, delta, n_r, n_h):
    R = np.array([risk_bar(l, lr, lh, delta, n_r, n_h) for l in LAM_GRID])
    d0 = (R[1] - R[0]) / LAM_GRID[1]
    return float(LAM_GRID[np.argmin(R)]), float(d0), float(R.min()), float(R[0])

lr = jj ** (-2.0); lh = jj ** (-0.9)   # 谱系 A
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
boundary_data = {}
for row, dnorm in enumerate(DELTAS):
    for col, (pf_name, dvec) in enumerate(PROFILES.items()):
        delta = -dvec / np.linalg.norm(dvec) * dnorm   # 反对齐
        Z = np.zeros((len(NR_GRID), len(RHO_GRID)))
        for i, n_r in enumerate(NR_GRID):
            for k, rho in enumerate(RHO_GRID):
                ls, d0, _, _ = analyze(lr, lh, delta, n_r, int(n_r * rho))
                Z[i, k] = ls
        ax = axes[row][col]
        im = ax.pcolormesh(np.log10(RHO_GRID), np.log10(NR_GRID), Z, cmap="viridis",
                           vmin=0, vmax=1)
        cs = ax.contour(np.log10(RHO_GRID), np.log10(NR_GRID), Z, levels=[0.01],
                        colors="red", linewidths=2)
        n_harm = int((Z < 0.01).sum())
        ax.set_title(f"anti-aligned |delta|={dnorm} | {pf_name} | harmful cells: {n_harm}",
                     fontsize=10)
        ax.set_xlabel("log10(n_h/n_r)"); ax.set_ylabel("log10(n_r)")
        fig.colorbar(im, ax=ax, label="lambda*")
        boundary_data[f"{dnorm}|{pf_name}"] = dict(harmful_cells=n_harm,
                                                   lam_min=float(Z.min()), lam_max=float(Z.max()))
fig.suptitle("v8 phase diagram (spectra A, ANTI-aligned mismatch): red contour = harmful region (lambda*<0.01)")
fig.tight_layout()
fig.savefig("/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l1_phase_v8_anti.png", dpi=120)
with open("/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l1_phase_v8_results.json", "w") as f:
    json.dump(boundary_data, f, indent=1)
print(json.dumps(boundary_data, indent=1))
