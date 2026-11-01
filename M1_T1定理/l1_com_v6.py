"""v6：完整 ridge（A⁻¹b）vs 对角化估计量 vs v5 预测——界定对角化近似适用范围。

3 个对象在同一批数据上对比：
  full：β̂ = A⁻¹b（A 含非对角元）
  diag：β̂_j = (w_rT_rj + w_hT_hj)/(w_rS_rjj + w_hS_hjj + μ)（对角化近似）
  v5 预测：馈通修正的逐坐标公式
配置：A|lowfreq 与 B|lowfreq，λ=0.3，REPS=50。
"""
import numpy as np
import json

rng = np.random.default_rng(31)

P = 1000
N_R, N_H = 1000, 6000
SIG_R2, SIG_H2 = 1.0, 4.0
MU = 0.05
REPS = 50
LAM = 0.3
S_SMOOTH = 1.0
NBINS = 16

jj = np.arange(1, P + 1, dtype=float)
beta_star = jj ** (-(S_SMOOTH + 0.5))
edges = np.unique(np.round(np.logspace(0, np.log10(P), NBINS + 1)).astype(int))
bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

CASES = {
    "A(αr2.0,αh0.9)|lowfreq": (dict(alpha_r=2.0, alpha_h=0.9), jj ** -1.0),
    "B(αr1.2,αh1.0)|lowfreq": (dict(alpha_r=1.2, alpha_h=1.0), jj ** -1.0),
}
DELTA_NORM = 0.5

def feedthrough(lam_spec, beta):
    e = lam_spec * beta**2
    return e.sum() - e

def pred_v5(lam, lr, lh, delta, beta_h):
    wr, wh = (1 - lam) / N_R, lam / N_H
    D = (1 - lam) * lr + lam * lh + MU
    N0 = lam * lh * delta - MU * beta_star
    F_r = feedthrough(lr, beta_star); F_h = feedthrough(lh, beta_h)
    VarN = (wh * delta)**2 * 2 * N_H * lh**2 \
        + wr**2 * (SIG_R2 + F_r) * N_R * lr \
        + wh**2 * (SIG_H2 + F_h) * N_H * lh
    CovND = wh**2 * delta * 2 * N_H * lh**2
    VarD = 2 * ((1 - lam)**2 * lr**2 / N_R + lam**2 * lh**2 / N_H)
    return (N0**2 / D**2 + VarN / D**2 - 4 * N0 * CovND / D**3
            + 3 * N0**2 * VarD / D**4)

out_all = {}
for case_name, (cc, dvec) in CASES.items():
    lr = jj ** (-cc["alpha_r"]); lh = jj ** (-cc["alpha_h"])
    delta = dvec / np.linalg.norm(dvec) * DELTA_NORM
    beta_h = beta_star + delta
    mse_full = np.zeros((REPS, P)); mse_diag = np.zeros((REPS, P))
    for rep in range(REPS):
        Xr = rng.normal(size=(N_R, P)) * np.sqrt(lr)
        yr = Xr @ beta_star + np.sqrt(SIG_R2) * rng.normal(size=N_R)
        Xh = rng.normal(size=(N_H, P)) * np.sqrt(lh)
        yh = Xh @ beta_h + np.sqrt(SIG_H2) * rng.normal(size=N_H)
        wr, wh = (1 - LAM) / N_R, LAM / N_H
        Gr = Xr.T @ Xr; Gh = Xh.T @ Xh
        br = Xr.T @ yr;  bh = Xh.T @ yh
        A = wr * Gr + wh * Gh + MU * np.eye(P)
        b = wr * br + wh * bh
        beta_full = np.linalg.solve(A, b)
        beta_diag = b / np.diag(A)
        mse_full[rep] = (beta_full - beta_star) ** 2
        mse_diag[rep] = (beta_diag - beta_star) ** 2
    ef = mse_full.mean(0); ed = mse_diag.mean(0); p5 = pred_v5(LAM, lr, lh, delta, beta_h)
    rows = []
    for (a, b_) in bins:
        sl = slice(a - 1, b_)
        rows.append(dict(j=(int(a), int(b_)),
                         emp_full=float(ef[sl].mean()), emp_diag=float(ed[sl].mean()),
                         v5=float(p5[sl].mean())))
    relf = [abs(r["emp_full"] - r["v5"]) / max(r["v5"], 1e-300) for r in rows]
    reld = [abs(r["emp_diag"] - r["v5"]) / max(r["v5"], 1e-300) for r in rows]
    ratio_fd = [r["emp_full"] / max(r["emp_diag"], 1e-300) for r in rows]
    out_all[case_name] = dict(bins=rows,
        bin_rel_median_full_vs_v5=float(np.median(relf)),
        bin_rel_median_diag_vs_v5=float(np.median(reld)),
        bin_median_full_over_diag=float(np.median(ratio_fd)),
        total_full=float(ef.sum()), total_diag=float(ed.sum()), total_v5=float(p5.sum()),
        total_w_full=float((ef * lr).sum()), total_w_diag=float((ed * lr).sum()),
        total_w_v5=float((p5 * lr).sum()))

with open("/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l1_com_v6_results.json", "w") as f:
    json.dump(dict(p=P, n_r=N_R, n_h=N_H, mu=MU, lam=LAM, reps=REPS, results=out_all), f, indent=1)

for k, v in out_all.items():
    print(f"{k}: full/diag 中位比 {v['bin_median_full_over_diag']:.3f} | "
          f"bin-med |full−v5|/v5 {v['bin_rel_median_full_vs_v5']:.3f} | "
          f"|diag−v5|/v5 {v['bin_rel_median_diag_vs_v5']:.3f}")
    print(f"   总量 full/diag/v5: {v['total_full']:.4f}/{v['total_diag']:.4f}/{v['total_v5']:.4f} | "
          f"Σr加权: {v['total_w_full']:.5f}/{v['total_w_diag']:.5f}/{v['total_w_v5']:.5f}")
print("saved l1_com_v6_results.json")
