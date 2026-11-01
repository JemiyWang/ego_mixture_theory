"""核验脚本 2：v0.3 边界配置的 MC 噪声探针 + 割线偏差演示。"""
import numpy as np
import json

# ---------- 割线 vs 导数（在理论曲线上演示，零 MC 噪声） ----------
print("=" * 70)
print("[J] 前向割线对 R'(0+) 的系统性偏差（理论曲线演示）")
print("=" * 70)
p, mu, B = 200, 0.05, 1.0
for (n_r, s2, c, dphi, h) in [(50, 2.0, 0.5, 0.25, 0.005),
                              (2000, 2.0, -2.0, 4.0, 0.001)]:
    D = lambda l: (1 - l) + l * s2 + mu
    R = lambda l: ((l**2 * s2**2 * dphi - 2 * l * s2 * mu * c + mu**2 * B)
                   + p * ((1 - l)**2 / n_r + l**2 * s2 / (5 * n_r))) / D(l)**2
    dR = (-2 * s2 * mu * c * (1 + mu) - 2 * mu**2 * B * (s2 - 1)
          - 2 * p * (s2 + mu) / n_r) / (1 + mu)**3
    sec = (R(h) - R(0)) / h
    print(f"n_r={n_r},s2={s2},c={c}: 真导数={dR:+.4f}  割线(h={h})={sec:+.4f}  "
          f"|割线|/|导数|={abs(sec/dR):.3f}")

# ---------- v0.3 边界配置的 MC 标准误 ----------
print()
print("=" * 70)
print("[H/J] v0.3 n_r=2000 anti 配置：dR_emp 的 Monte Carlo 标准误（新种子）")
print("=" * 70)
rng = np.random.default_rng(12345)
P, SIG, MU, REPS = 200, 1.0, 0.05, 100
beta_star = rng.normal(size=P)
beta_star /= np.linalg.norm(beta_star)

def probe(n_r, n_h, s2, delta, lam_grid):
    beta_h = beta_star + delta
    per_rep_secants = []
    Rm = np.zeros((REPS, len(lam_grid)))
    for rep in range(REPS):
        Xr = rng.normal(size=(n_r, P))
        yr = Xr @ beta_star + SIG * rng.normal(size=n_r)
        Xh = rng.normal(scale=np.sqrt(s2), size=(n_h, P))
        yh = Xh @ beta_h + SIG * rng.normal(size=n_h)
        Sr, Sh = Xr.T @ Xr, Xh.T @ Xh
        br, bh = Xr.T @ yr, Xh.T @ yh
        risks = []
        for lam in lam_grid:
            wr, wh = (1 - lam) / n_r, lam / n_h
            A = wr * Sr + wh * Sh + MU * np.eye(P)
            b = wr * br + wh * bh
            diff = np.linalg.solve(A, b) - beta_star
            risks.append(diff @ diff)
        Rm[rep] = risks
        per_rep_secants.append((risks[1] - risks[0]) / lam_grid[1])
    sec = np.array(per_rep_secants)
    return sec.mean(), sec.std(ddof=1) / np.sqrt(REPS), Rm

delta_anti = -2.0 * beta_star  # c=-2, Delta=4
m, se, _ = probe(2000, 10000, 2.0, delta_anti, [0.0, 0.001])
print(f"n_r=2000 anti (c=-2): dR_emp = {m:+.4f} ± {se:.4f} (SE, reps=100, 新种子)")
print(f"  JSON 记录值 = -0.0040；理论 = +0.0043")
print(f"  => |观测值| 约为 {abs(m)/se:.1f} 倍 SE" if se > 0 else "")
