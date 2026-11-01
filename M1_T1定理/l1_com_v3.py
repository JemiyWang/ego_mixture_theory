"""L0-COM 逐坐标验证 v3：分箱平均消除 MC 噪声 + Σ_r 加权风险。

设计：3 个配置（A|lowfreq, A|highfreq, B|lowfreq），λ ∈ {0.3, 0.5}，REPS=200。
把坐标按 j 分成 16 个 octave 箱，比较箱内"经验 MSE 均值" vs "预测 MSE 均值"，
消除单坐标 MC 噪声（40 reps 时单坐标噪声地板 ~22%）。
同时报告 Σ_r 加权风险（λ^r_j 加权）与非加权风险两套总量。
"""
import numpy as np
import json

rng = np.random.default_rng(23)

P = 1000
N_R, N_H = 1000, 6000
SIG_R2, SIG_H2 = 1.0, 4.0
MU = 0.05
REPS = 200
LAMBDAS = [0.3, 0.5]
S_SMOOTH = 1.0
NBINS = 16

jj = np.arange(1, P + 1, dtype=float)
beta_star = jj ** (-(S_SMOOTH + 0.5))
# 等距对数分箱
edges = np.unique(np.round(np.logspace(0, np.log10(P), NBINS + 1)).astype(int))
bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

CASES = {
    "A(αr2.0,αh0.9)|lowfreq":  (dict(alpha_r=2.0, alpha_h=0.9), jj ** -1.0),
    "A(αr2.0,αh0.9)|highfreq": (dict(alpha_r=2.0, alpha_h=0.9), (jj / P) ** 2),
    "B(αr1.2,αh1.0)|lowfreq":  (dict(alpha_r=1.2, alpha_h=1.0), jj ** -1.0),
}
DELTA_NORM = 0.5

def pred_per_coord(lam, lr, lh, delta):
    D = (1 - lam) * lr + lam * lh + MU
    bias = (lam * lh * delta - MU * beta_star) / D
    var = ((1 - lam) ** 2 * SIG_R2 * lr / N_R + lam**2 * SIG_H2 * lh / N_H) / D**2
    return var + bias**2

out_all = {}
for case_name, (cc, dvec) in CASES.items():
    lr = jj ** (-cc["alpha_r"]); lh = jj ** (-cc["alpha_h"])
    delta = dvec / np.linalg.norm(dvec) * DELTA_NORM
    S_r = np.zeros((REPS, P)); S_h = np.zeros((REPS, P))
    T_r = np.zeros((REPS, P)); T_h = np.zeros((REPS, P))
    for rep in range(REPS):
        Xr = rng.normal(size=(N_R, P)) * np.sqrt(lr)
        yr = Xr @ beta_star + np.sqrt(SIG_R2) * rng.normal(size=N_R)
        Xh = rng.normal(size=(N_H, P)) * np.sqrt(lh)
        yh = Xh @ (beta_star + delta) + np.sqrt(SIG_H2) * rng.normal(size=N_H)
        S_r[rep] = (Xr**2).sum(0); S_h[rep] = (Xh**2).sum(0)
        T_r[rep] = Xr.T @ yr;      T_h[rep] = Xh.T @ yh
    rec = dict()
    for lam in LAMBDAS:
        wr = (1 - lam) / N_R; wh = lam / N_H
        beta_hat = (wr * T_r + wh * T_h) / (wr * S_r + wh * S_h + MU)
        mse_emp = ((beta_hat - beta_star) ** 2).mean(axis=0)       # 逐坐标
        mse_pred = pred_per_coord(lam, lr, lh, delta)
        # 分箱对比
        bin_tbl = []
        for (a, b) in bins:
            sl = slice(a - 1, b)
            emp_b = float(mse_emp[sl].mean()); pred_b = float(mse_pred[sl].mean())
            bin_tbl.append(dict(j_lo=int(a), j_hi=int(b), emp=emp_b, pred=pred_b,
                                rel=abs(emp_b - pred_b) / max(pred_b, 1e-300)))
        # 总量：Σ_r 加权与非加权
        rec[f"lam={lam}"] = dict(
            bins=bin_tbl,
            bin_rel_median=float(np.median([t["rel"] for t in bin_tbl])),
            total_emp_unweighted=float(mse_emp.sum()),
            total_pred_unweighted=float(mse_pred.sum()),
            total_emp_weighted=float((mse_emp * lr).sum()),
            total_pred_weighted=float((mse_pred * lr).sum()),
        )
    out_all[case_name] = rec

with open("/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l1_com_v3_results.json", "w") as f:
    json.dump(dict(p=P, n_r=N_R, n_h=N_H, mu=MU, reps=REPS, lambdas=LAMBDAS,
                   delta_norm=DELTA_NORM, results=out_all), f, indent=1)

for k, v in out_all.items():
    for lam in LAMBDAS:
        r = v[f"lam={lam}"]
        print(f"{k} λ={lam}: 分箱中位相对误差 {r['bin_rel_median']:.3f} | "
              f"非加权 emp/pred {r['total_emp_unweighted']:.4f}/{r['total_pred_unweighted']:.4f} | "
              f"Σ_r加权 emp/pred {r['total_emp_weighted']:.5f}/{r['total_pred_weighted']:.5f}")
print("saved l1_com_v3_results.json")
