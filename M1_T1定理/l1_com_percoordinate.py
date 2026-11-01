"""M0-COM 逐坐标数值验证：检验猜想 P0.2 的结构。

模型（对角可交换谱，坐标 j 精确解耦，μ≈0）：
  x_rj ~ N(0, λr_j), λr_j = j^{-α_r};  y_rj = x·β*_j + ε_r
  x_hj ~ N(0, λh_j), λh_j = j^{-α_h};  y_hj = x·(β*_j + δ_j) + ε_h
逐坐标估计（权重 w_r=(1−λ)/n_r, w_h=λ/n_h）：
  β̂_j = (w_r S_rj + w_h S_hj + μ)⁻¹ (w_r T_rj + w_h T_hj)

P0.2 预测（总体版本）：
  w_j^h = λ λh_j / [(1−λ)λr_j + λ λh_j]
  bias_j = w_j^h δ_j
  Var_j = [(1−λ)²σ_r²λr_j/n_r + λ²σ_h²λh_j/n_h] / [(1−λ)λr_j + λλh_j]²
  MSE_j ≈ Var_j + bias_j²
"""
import numpy as np
import json

rng = np.random.default_rng(7)

P = 1000
N_R, N_H = 400, 4000
SIG_R2 = SIG_H2 = 1.0
MU = 1e-8
REPS = 100
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
S_SMOOTH = 1.0  # β*_j = j^{-(s+1/2)}

jj = np.arange(1, P + 1, dtype=float)
beta_star = jj ** (-(S_SMOOTH + 0.5))

CASES = {
    "A_window(δ=1.1)": dict(alpha_r=2.0, alpha_h=0.9),
    "B_outside(δ=0.2)": dict(alpha_r=1.2, alpha_h=1.0),
}
MISMATCH = {
    "white":   lambda: np.ones(P),
    "lowfreq": lambda: jj ** -1.0,
    "highfreq": lambda: (jj / P) ** 2,
}
DELTA_NORM = 0.3

def per_coord_pred(lam, lr, lh, delta):
    den = (1 - lam) * lr + lam * lh
    w_h = lam * lh / den
    var = ((1 - lam) ** 2 * SIG_R2 * lr / N_R + lam**2 * SIG_H2 * lh / N_H) / den**2
    bias = w_h * delta
    return var + bias**2

out_all = {}
for case_name, cc in CASES.items():
    lr = jj ** (-cc["alpha_r"]); lh = jj ** (-cc["alpha_h"])
    for mm_name, mk in MISMATCH.items():
        delta = mk(); delta = delta / np.linalg.norm(delta) * DELTA_NORM
        # 预生成数据的充分统计量（每 rep 复用于所有 λ）
        S_r = np.zeros((REPS, P)); S_h = np.zeros((REPS, P))
        T_r = np.zeros((REPS, P)); T_h = np.zeros((REPS, P))
        for rep in range(REPS):
            Xr = rng.normal(size=(N_R, P)) * np.sqrt(lr)
            yr = Xr @ beta_star + np.sqrt(SIG_R2) * rng.normal(size=N_R)
            Xh = rng.normal(size=(N_H, P)) * np.sqrt(lh)
            yh = Xh @ (beta_star + delta) + np.sqrt(SIG_H2) * rng.normal(size=N_H)
            S_r[rep] = (Xr**2).sum(0); S_h[rep] = (Xh**2).sum(0)
            T_r[rep] = Xr.T @ yr;      T_h[rep] = Xh.T @ yh
        key = f"{case_name}|{mm_name}"
        out_all[key] = dict(per_coord=[], total_emp=[], total_pred=[], lam_star={})
        for lam in LAMBDAS:
            wr = (1 - lam) / N_R; wh = lam / N_H
            beta_hat = (wr * T_r + wh * T_h) / (wr * S_r + wh * S_h + MU)
            mse_emp = ((beta_hat - beta_star) ** 2).mean(axis=0)      # 逐坐标
            mse_pred = per_coord_pred(lam, lr, lh, delta)
            # 逐坐标一致性：log 空间相对误差的中位数与 90% 分位
            rel = np.abs(mse_emp - mse_pred) / np.maximum(mse_pred, 1e-300)
            out_all[key]["per_coord"].append(dict(
                lam=lam, rel_median=float(np.median(rel)),
                rel_p90=float(np.quantile(rel, 0.9))))
            out_all[key]["total_emp"].append(float(mse_emp.sum()))
            out_all[key]["total_pred"].append(float(mse_pred.sum()))
        i_e = int(np.argmin(out_all[key]["total_emp"]))
        i_p = int(np.argmin(out_all[key]["total_pred"]))
        out_all[key]["lam_star"] = dict(emp=LAMBDAS[i_e], pred=LAMBDAS[i_p])

with open("/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l1_com_results.json", "w") as f:
    json.dump(dict(p=P, n_r=N_R, n_h=N_H, lambdas=LAMBDAS, reps=REPS,
                   delta_norm=DELTA_NORM, s=S_SMOOTH, results=out_all), f, indent=1)

print(f"{'case':<26} | {'med rel-err @λ=.2/.5':>22} | {'λ* emp':>6} {'λ* pred':>7} | "
      f"{'R(0)':>8} {'R(λ*e)':>8} {'R_pred(λ*p)':>10}")
for k, v in out_all.items():
    pc = {d["lam"]: d["rel_median"] for d in v["per_coord"]}
    print(f"{k:<26} | {pc[0.2]:>10.3f} {pc[0.5]:>10.3f} | "
          f"{v['lam_star']['emp']:>6.2f} {v['lam_star']['pred']:>7.2f} | "
          f"{v['total_emp'][0]:>8.4f} {min(v['total_emp']):>8.4f} {min(v['total_pred']):>10.4f}")
print("saved l1_com_results.json")
