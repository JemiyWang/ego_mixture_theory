"""M0-COM 逐坐标数值验证 v2：带 ridge μ 的猜想 P0.2 结构检验。

相对 v1 的修正：μ=0.05（控制低信息坐标爆炸）；n_r=1000（γ_r=1）；σ_h=2（人域标签更噪）；
失配范数 0.5（更强）；按坐标信息充裕度分层报告预测精度。

逐坐标预测（含 μ 的总体版）：
  D_j = (1−λ)λr_j + λλh_j + μ
  bias_j = [λ λh_j δ_j − μ β*_j] / D_j
  Var_j = [(1−λ)²σ_r²λr_j/n_r + λ²σ_h²λh_j/n_h] / D_j²
  MSE_j ≈ Var_j + bias_j²
"""
import numpy as np
import json

rng = np.random.default_rng(11)

P = 1000
N_R, N_H = 1000, 6000
SIG_R2, SIG_H2 = 1.0, 4.0   # σ_h = 2
MU = 0.05
REPS = 40
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
S_SMOOTH = 1.0

jj = np.arange(1, P + 1, dtype=float)
beta_star = jj ** (-(S_SMOOTH + 0.5))

CASES = {
    "A_window(δ=1.1)": dict(alpha_r=2.0, alpha_h=0.9),
    "B_outside(δ=0.2)": dict(alpha_r=1.2, alpha_h=1.0),
}
MISMATCH = {
    "white":    lambda: np.ones(P),
    "lowfreq":  lambda: jj ** -1.0,
    "highfreq": lambda: (jj / P) ** 2,
}
DELTA_NORM = 0.5

def per_coord_pred(lam, lr, lh, delta):
    D = (1 - lam) * lr + lam * lh + MU
    bias = (lam * lh * delta - MU * beta_star) / D
    var = ((1 - lam) ** 2 * SIG_R2 * lr / N_R + lam**2 * SIG_H2 * lh / N_H) / D**2
    return var + bias**2

out_all = {}
for case_name, cc in CASES.items():
    lr = jj ** (-cc["alpha_r"]); lh = jj ** (-cc["alpha_h"])
    for mm_name, mk in MISMATCH.items():
        delta = mk(); delta = delta / np.linalg.norm(delta) * DELTA_NORM
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
        rec = dict(per_coord=[], total_emp=[], total_pred=[], lam_star={})
        for lam in LAMBDAS:
            wr = (1 - lam) / N_R; wh = lam / N_H
            beta_hat = (wr * T_r + wh * T_h) / (wr * S_r + wh * S_h + MU)
            mse_emp = ((beta_hat - beta_star) ** 2).mean(axis=0)
            mse_pred = per_coord_pred(lam, lr, lh, delta)
            # 坐标信息充裕度分层：信息充裕 = 数据项 ≥ 5μ（涨落可控）
            info = (1 - lam) * lr + lam * lh
            rich = info >= 5 * MU
            rel_rich = np.abs(mse_emp[rich] - mse_pred[rich]) / mse_pred[rich] if rich.any() else [np.nan]
            rel_poor = np.abs(mse_emp[~rich] - mse_pred[~rich]) / mse_pred[~rich] if (~rich).any() else [np.nan]
            rec["per_coord"].append(dict(
                lam=lam,
                rel_median_rich=float(np.median(rel_rich)),
                rel_median_poor=float(np.median(rel_poor)),
                n_rich=int(rich.sum())))
            rec["total_emp"].append(float(mse_emp.sum()))
            rec["total_pred"].append(float(mse_pred.sum()))
        i_e = int(np.argmin(rec["total_emp"])); i_p = int(np.argmin(rec["total_pred"]))
        rec["lam_star"] = dict(emp=LAMBDAS[i_e], pred=LAMBDAS[i_p])
        out_all[key] = rec

with open("/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l1_com_v2_results.json", "w") as f:
    json.dump(dict(p=P, n_r=N_R, n_h=N_H, sig_h2=SIG_H2, mu=MU, lambdas=LAMBDAS,
                   reps=REPS, delta_norm=DELTA_NORM, results=out_all), f, indent=1)

print(f"{'case':<26} | {'rich-coord med rel-err @.2/.5':>30} | {'λ*e':>4} {'λ*p':>4} | "
      f"{'R(0)':>8} {'R(λ*e)':>8} {'Rth(λ*p)':>8}")
for k, v in out_all.items():
    pc = {d["lam"]: d["rel_median_rich"] for d in v["per_coord"]}
    print(f"{k:<26} | {pc[0.2]:>14.3f} {pc[0.5]:>14.3f} | "
          f"{v['lam_star']['emp']:>4.2f} {v['lam_star']['pred']:>4.2f} | "
          f"{v['total_emp'][0]:>8.4f} {min(v['total_emp']):>8.4f} {min(v['total_pred']):>8.4f}")
print("saved l1_com_v2_results.json")
