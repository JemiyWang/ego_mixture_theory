"""L0 模型 M0-ISO 的 Monte Carlo 数值验证（对应 L0模型与边际收益命题.md §3/§5）。

模型：x_r ~ N(0, I_p), y_r = x·β* + ε_r (σ_r)
      x_h ~ N(0, s²I_p), y_h = x·β_h + ε_h (σ_h), β_h = β* + δ, Δ_φ = ‖δ‖²
混合 ridge：min w_r‖y_r−X_rβ‖² + w_h‖y_h−X_hβ‖² + μ‖β‖²，w_r=(1−λ)/n_r, w_h=λ/n_h
目标：R(λ) = E‖β̂−β*‖²（Σ_r = I）

理论预测（天真确定性等价，A≈EA，γ→0 时精确）：
  D(λ) = (1−λ) + λs² + μ
  bias² = [λ²s⁴Δ − 2λs²μc + μ²B] / D²,  c = δᵀβ*, B = ‖β*‖²
  Var   = p[σ_r²(1−λ)²/(n_rD²) + σ_h²λ²s²/(n_hD²)]
  R'(0+) = [−2s²μc(1+μ) − 2μ²B(s²−1) − 2pσ_r²(s²+μ)/n_r] / (1+μ)³
"""
import numpy as np
import json
import os

rng = np.random.default_rng(1)

P = 200
SIG_R, SIG_H = 1.0, 1.0
MU = 0.05
REPS = 100
LAMBDAS = np.array([0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4])

# 固定 β* 保证 c 稳定
beta_star = rng.normal(size=P)
beta_star /= np.linalg.norm(beta_star)   # B = 1
B = float(beta_star @ beta_star)


def make_delta(mode, mag):
    """构造失配方向：aligned(c>0) / anti(c<0) / orth(c=0)，‖δ‖=mag。"""
    if mode == "aligned":
        d = beta_star.copy()
    elif mode == "anti":
        d = -beta_star.copy()
    else:
        d = rng.normal(size=P)
        d -= (d @ beta_star) * beta_star
    d /= np.linalg.norm(d)
    return mag * d


def run_config(n_r, n_h, s2, delta_mode, delta_mag=2.0):
    delta = make_delta(delta_mode, delta_mag)
    beta_h = beta_star + delta
    c = float(delta @ beta_star)
    dphi = float(delta @ delta)
    risks = np.zeros((REPS, len(LAMBDAS)))
    for rep in range(REPS):
        Xr = rng.normal(size=(n_r, P))
        yr = Xr @ beta_star + SIG_R * rng.normal(size=n_r)
        Xh = rng.normal(scale=np.sqrt(s2), size=(n_h, P))
        yh = Xh @ beta_h + SIG_H * rng.normal(size=n_h)
        Sr = Xr.T @ Xr
        Sh = Xh.T @ Xh
        br = Xr.T @ yr
        bh = Xh.T @ yh
        for i, lam in enumerate(LAMBDAS):
            wr = (1 - lam) / n_r
            wh = lam / n_h
            A = wr * Sr + wh * Sh + MU * np.eye(P)
            b = wr * br + wh * bh
            bh_hat = np.linalg.solve(A, b)
            diff = bh_hat - beta_star
            risks[rep, i] = diff @ diff
    R = risks.mean(axis=0)

    # 理论曲线
    D = (1 - LAMBDAS) + LAMBDAS * s2 + MU
    bias2 = (LAMBDAS**2 * s2**2 * dphi - 2 * LAMBDAS * s2 * MU * c + MU**2 * B) / D**2
    var = P * (SIG_R**2 * (1 - LAMBDAS)**2 / (n_r * D**2)
               + SIG_H**2 * LAMBDAS**2 * s2 / (n_h * D**2))
    R_theory = bias2 + var

    # R'(0+)：经验值用最小非零 λ 的差分；理论值用闭式
    dR_emp = (R[1] - R[0]) / LAMBDAS[1]
    dR_th = (-2 * s2 * MU * c * (1 + MU) - 2 * MU**2 * B * (s2 - 1)
             - 2 * P * SIG_R**2 * (s2 + MU) / n_r) / (1 + MU)**3

    lam_star_emp = float(LAMBDAS[np.argmin(R)])
    lam_star_th = float(LAMBDAS[np.argmin(R_theory)])
    return dict(n_r=n_r, n_h=n_h, s2=s2, mode=delta_mode, c=c,
                dR_emp=float(dR_emp), dR_theory=float(dR_th),
                R_emp=list(map(float, R)), R_theory=list(map(float, R_theory)),
                lam_star_emp=lam_star_emp, lam_star_theory=lam_star_th)


configs = []
for n_r in [2000, 10000]:
    for s2 in [2.0]:
        for mode in ["anti", "orth"]:
            configs.append(dict(n_r=n_r, n_h=5 * n_r, s2=s2, delta_mode=mode))

results = [run_config(**cfg) for cfg in configs]

out = "/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l0_mc_v03_results.json"
with open(out, "w") as f:
    json.dump(dict(p=P, mu=MU, sig_r=SIG_R, sig_h=SIG_H, reps=REPS,
                   lambdas=list(map(float, LAMBDAS)), results=results), f, indent=1)

print(f"{'n_r':>5} {'s²':>4} {'mode':>8} {'c':>6} | {'dR/dλ emp':>10} {'theory':>10} | "
      f"{'λ* emp':>7} {'λ* th':>7} | sign agree")
for r in results:
    agree = (r["dR_emp"] < 0) == (r["dR_theory"] < 0)
    print(f"{r['n_r']:>5} {r['s2']:>4} {r['mode']:>8} {r['c']:>6.2f} | "
          f"{r['dR_emp']:>10.3f} {r['dR_theory']:>10.3f} | "
          f"{r['lam_star_emp']:>7.3f} {r['lam_star_theory']:>7.3f} | {agree}")
print("saved:", out)
