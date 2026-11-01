"""L0-COM 逐坐标验证 v5：串扰修正（标签经样本 Gram 非对角的馈通）。

v4 发现：二阶比率涨落修正几乎不缩小差距 → 主误差源不是比率涨落。
v5 修正：T_ej = X_ejᵀy_e 含非对角馈通 Σ_{k≠j} S_e,jk β_e,k，
其方差 = w_e² n_e λe_j · F_e,j，F_e,j := Σ_{k≠j} λe_k β_e,k²（β_h = β*+δ）。
⇒ 逐坐标有效噪声 σ̃_e,j² = σ_e² + F_e,j（坐标外信号对坐标 j 而言就是噪声）。
与 D̂ 的协方差：S_e,jj 与 S_e,jk（k≠j）独立 → 无额外协方差项。

v5 预测 = v4 公式中 VarN 的噪声部分 σ_e² 换成 σ̃_e,j²。
"""
import numpy as np
import json

rng = np.random.default_rng(23)   # 与 v3/v4 同种子

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
edges = np.unique(np.round(np.logspace(0, np.log10(P), NBINS + 1)).astype(int))
bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

CASES = {
    "A(αr2.0,αh0.9)|lowfreq":  (dict(alpha_r=2.0, alpha_h=0.9), jj ** -1.0),
    "A(αr2.0,αh0.9)|highfreq": (dict(alpha_r=2.0, alpha_h=0.9), (jj / P) ** 2),
    "B(αr1.2,αh1.0)|lowfreq":  (dict(alpha_r=1.2, alpha_h=1.0), jj ** -1.0),
}
DELTA_NORM = 0.5

def feedthrough(lam_spec, beta):
    """F_j = Σ_{k≠j} λ_k β_k²"""
    e = lam_spec * beta**2
    return e.sum() - e          # 去掉自身坐标

def pred_v5(lam, lr, lh, delta, beta_h):
    wr, wh = (1 - lam) / N_R, lam / N_H
    D = (1 - lam) * lr + lam * lh + MU
    N0 = lam * lh * delta - MU * beta_star
    F_r = feedthrough(lr, beta_star)          # 机器人域串扰能
    F_h = feedthrough(lh, beta_h)             # 人域串扰能
    # Var(N̂)：失配涨落 + (噪声+串扰) 的期望
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
    S_r = np.zeros((REPS, P)); S_h = np.zeros((REPS, P))
    T_r = np.zeros((REPS, P)); T_h = np.zeros((REPS, P))
    for rep in range(REPS):
        Xr = rng.normal(size=(N_R, P)) * np.sqrt(lr)
        yr = Xr @ beta_star + np.sqrt(SIG_R2) * rng.normal(size=N_R)
        Xh = rng.normal(size=(N_H, P)) * np.sqrt(lh)
        yh = Xh @ beta_h + np.sqrt(SIG_H2) * rng.normal(size=N_H)
        S_r[rep] = (Xr**2).sum(0); S_h[rep] = (Xh**2).sum(0)
        T_r[rep] = Xr.T @ yr;      T_h[rep] = Xh.T @ yh
    rec = dict()
    for lam in LAMBDAS:
        wr, wh = (1 - lam) / N_R, lam / N_H
        beta_hat = (wr * T_r + wh * T_h) / (wr * S_r + wh * S_h + MU)
        mse_emp = ((beta_hat - beta_star) ** 2).mean(axis=0)
        p5 = pred_v5(lam, lr, lh, delta, beta_h)
        rows = []
        for (a, b) in bins:
            sl = slice(a - 1, b)
            e = float(mse_emp[sl].mean()); v = float(p5[sl].mean())
            rows.append(dict(j=(int(a), int(b)), emp=e, v5=v,
                             rel=abs(e - v) / max(v, 1e-300)))
        rec[f"lam={lam}"] = dict(
            bins=rows,
            bin_rel_median_v5=float(np.median([r["rel"] for r in rows])),
            total_emp=float(mse_emp.sum()), total_v5=float(p5.sum()),
            total_w_emp=float((mse_emp * lr).sum()), total_w_v5=float((p5 * lr).sum()))
    out_all[case_name] = rec

with open("/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l1_com_v5_results.json", "w") as f:
    json.dump(dict(p=P, n_r=N_R, n_h=N_H, mu=MU, reps=REPS, results=out_all), f, indent=1)

print(f"{'case':<28} {'λ':>4} | {'bin-med rel v5':>14} | {'tot emp':>8} {'v5':>8} | {'Σr加权 emp/v5'}")
for k, v in out_all.items():
    for lam in LAMBDAS:
        r = v[f"lam={lam}"]
        print(f"{k:<28} {lam:>4.1f} | {r['bin_rel_median_v5']:>14.3f} | "
              f"{r['total_emp']:>8.4f} {r['total_v5']:>8.4f} | "
              f"{r['total_w_emp']:.5f}/{r['total_w_v5']:.5f}")
print("saved l1_com_v5_results.json")
