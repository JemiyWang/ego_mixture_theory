"""L0-COM 逐坐标验证 v4：二阶 delta-method 涨落修正的闭式检验。

目标：检验 L3 的证明结构——对 E[N̂²/D̂²] 做二阶展开是否闭合 v3 发现的系统差距。

逐坐标（设计矩阵层随机量）：
  D̂ = w_r S_rj + w_h S_hj + μ,  S_ej ~ λe_j χ²_{n_e}
  N̂ = w_h S_hj δ_j − μβ*_j + (w_r X_rjᵀε_r + w_h X_hjᵀε_h)
  MSE_j = E[N̂²/D̂²]

二阶展开（f(N,D)=N²/D²，N0=EN̂，D=E D̂）：
  E[f] ≈ N0²/D² + Var(N̂)/D² − 4·N0·Cov(N̂,D̂)/D³ + 3·N0²·Var(D̂)/D⁴
其中（全部闭式）：
  Var(N̂) = (w_h δ_j)²·2n_h(λh_j)² + E[w_r²σ_r²S_rj + w_h²σ_h²S_hj]
  Cov(N̂,D̂) = w_h² δ_j · 2n_h(λh_j)²
  Var(D̂) = 2[(1−λ)²(λr_j)²/n_r + λ²(λh_j)²/n_h]
  E[w_e²σ_e²S_ej] = w_e²σ_e² n_e λe_j = （权重化简后）(1−λ)²σ_r²λr_j/n_r 等

对照：v3 的零阶（总体代入）预测 vs v4 二阶修正预测 vs 经验值（REPS=200）。
"""
import numpy as np
import json

rng = np.random.default_rng(23)   # 与 v3 同种子，经验侧完全一致可比

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

def pred_v3(lam, lr, lh, delta):
    """零阶（总体代入）"""
    D = (1 - lam) * lr + lam * lh + MU
    bias = (lam * lh * delta - MU * beta_star) / D
    var = ((1 - lam) ** 2 * SIG_R2 * lr / N_R + lam**2 * SIG_H2 * lh / N_H) / D**2
    return var + bias**2

def pred_v4(lam, lr, lh, delta):
    """二阶 delta-method 修正"""
    wr, wh = (1 - lam) / N_R, lam / N_H
    ES_r = N_R * lr; ES_h = N_H * lh
    D = wr * ES_r + wh * ES_h + MU                       # = (1−λ)lr + λlh + μ
    N0 = wh * ES_h * delta - MU * beta_star              # = λlh·δ − μβ*
    VarS_r = 2 * N_R * lr**2; VarS_h = 2 * N_H * lh**2
    # Var(N̂)：系统性部分 + 条件噪声的期望
    VarN = (wh * delta)**2 * VarS_h + (wr**2 * SIG_R2 * ES_r + wh**2 * SIG_H2 * ES_h)
    CovND = (wh**2) * delta * VarS_h
    VarD = wr**2 * VarS_r + wh**2 * VarS_h
    return (N0**2 / D**2 + VarN / D**2 - 4 * N0 * CovND / D**3
            + 3 * N0**2 * VarD / D**4)

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
        wr, wh = (1 - lam) / N_R, lam / N_H
        beta_hat = (wr * T_r + wh * T_h) / (wr * S_r + wh * S_h + MU)
        mse_emp = ((beta_hat - beta_star) ** 2).mean(axis=0)
        p0 = pred_v3(lam, lr, lh, delta)
        p2 = pred_v4(lam, lr, lh, delta)
        rows = []
        for (a, b) in bins:
            sl = slice(a - 1, b)
            e = float(mse_emp[sl].mean()); v0 = float(p0[sl].mean()); v2 = float(p2[sl].mean())
            rows.append(dict(j=(int(a), int(b)), emp=e, v0=v0, v2=v2,
                             rel0=abs(e - v0) / max(v0, 1e-300),
                             rel2=abs(e - v2) / max(v2, 1e-300)))
        rec[f"lam={lam}"] = dict(
            bins=rows,
            bin_rel_median_v0=float(np.median([r["rel0"] for r in rows])),
            bin_rel_median_v2=float(np.median([r["rel2"] for r in rows])),
            total_emp=float(mse_emp.sum()), total_v0=float(p0.sum()), total_v2=float(p2.sum()),
            total_w_emp=float((mse_emp * lr).sum()), total_w_v0=float((p0 * lr).sum()),
            total_w_v2=float((p2 * lr).sum()))
    out_all[case_name] = rec

with open("/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l1_com_v4_results.json", "w") as f:
    json.dump(dict(p=P, n_r=N_R, n_h=N_H, mu=MU, reps=REPS, results=out_all), f, indent=1)

print(f"{'case':<28} {'λ':>4} | {'bin-med rel v0':>14} {'v2(2阶)':>9} | "
      f"{'tot emp':>8} {'v0':>7} {'v2':>7} | {'Σr加权 emp/v0/v2'}")
for k, v in out_all.items():
    for lam in LAMBDAS:
        r = v[f"lam={lam}"]
        print(f"{k:<28} {lam:>4.1f} | {r['bin_rel_median_v0']:>14.3f} {r['bin_rel_median_v2']:>9.3f} | "
              f"{r['total_emp']:>8.4f} {r['total_v0']:>7.4f} {r['total_v2']:>7.4f} | "
              f"{r['total_w_emp']:.5f}/{r['total_w_v0']:.5f}/{r['total_w_v2']:.5f}")
print("saved l1_com_v4_results.json")
