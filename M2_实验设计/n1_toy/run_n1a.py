"""N1a 风险曲线复现(全程:闭式 200 reps + Adam 粗批单 seed s0)。

配置(实施卡 §4 N1a):谱系 A(α_r=2.0/α_h=0.9),低频 δ,‖δ‖=0.5,反对齐;
n_r=1000/n_h=6000;μ=0.05。
- E1/E2 × 9 λ × 200 reps(每次独立采样数据;2026-10-25 修订:20→200);
- E3(Adam 线性,显式 l2)× 9 λ,单 seed s0=23:l2=0.05(主)+ l2=0(健全门);
- E4(两层 MLP,显式 l2=0.05)× 9 λ,单 seed s0=23,只报告;
- 理论线 v5(配 E1)与 T1-b 变体(配 E2)用 201 点密网格。
经验风险:线性估计量 = Σ_j a_j(β̂_j-β*_j)²(a_j=lr_j,Σ_r 加权);E4 非线性
用 50k 蒙特卡洛测试集 E_x[(f(x)-xᵀβ*)²](同一 s0 rng 流,无噪声标签)。
E3/E4 与 E2 的比较均为同一份 s0 数据的配对比较。

检查门(判据自动算进 JSON):
1. E1 vs v5 公式中位相对误差 ≤2%(硬门,不过不进 Adam);
2. E3 健全门:E3(l2=0) vs E2(μ=0) 中位相对误差(报告数值);
3. N1a 判据:E3(l2=0.05) vs E2(μ=0.05) 中位相对误差 ≤5%;
4. E2 vs T1-b、E4 vs E2:仅报告;报告理论 λ*(v5)。

用法:python3 run_n1a.py            # 全程计算 + 落盘 + 绘图
     python3 run_n1a.py --plot-only  # 只从已存 JSON 重画图(不重算)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C
from gen_data import gen_two_domain
from estimators import (e1_diag_ridge, e2_full_ridge, e2_grams,
                        e3_sgd_linear, e4_mlp, mc_risk_model)
from theory import mse_v5, mse_t1b, risk_from_mse

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
JSON_PATH = OUT / "n1a_risk_curves.json"
PNG_PATH = OUT / "n1a_risk_curves.png"
PLOT_ONLY = "--plot-only" in sys.argv

# --- N1a 配置 ---
ALPHA_R, ALPHA_H = C.SPECTRA["A"]
DELTA_SHAPE, DELTA_NORM = "lowfreq", 0.5
REPS = C.REPS_CLOSED

beta_star = C.make_beta_star()
lam_r, lam_h = C.make_spectra(ALPHA_R, ALPHA_H)
delta = C.make_delta(DELTA_SHAPE, DELTA_NORM, beta_star, anti_align=True)
beta_h = beta_star + delta
c_align = float(delta @ beta_star)
a_w = lam_r.copy()                    # Σ_r 加权权重 a_j = lr_j
lams = np.array(C.LAMBDAS)

theory_args = (lam_r, lam_h, delta, beta_star, beta_h,
               C.N_R, C.N_H, C.SIG_R2, C.SIG_H2, C.MU)

def theory_curve(lams, fn):
    return np.array([risk_from_mse(fn(l, *theory_args), a_w) for l in lams])

def mean_se(x):
    return x.mean(0), x.std(0, ddof=1) / np.sqrt(x.shape[0])

def rel_err(a, b):
    return np.abs(a - b) / np.maximum(b, 1e-300)


# --- E3/E4 worker:每个子进程用同一 default_rng(s0) 流重建同一份训练数据
#     与 MC 测试集 ⇒ 9 个 λ 之间、E3/E4 与 E2 之间均为配对比较 ---
def _sgd_data():
    rng = np.random.default_rng(C.S0)
    rob, hum = gen_two_domain(rng, C.N_R, C.N_H, lam_r, lam_h,
                              beta_star, delta, C.SIG_R2, C.SIG_H2)
    return rob, hum, rng

def _e3_task(args):
    lam, l2 = args
    import torch
    torch.set_num_threads(4)   # 112 核机:9 worker × 4 线程,避免 112 线程微批抖动
    rob, hum, _ = _sgd_data()
    b = e3_sgd_linear(rob, hum, lam, l2=l2, steps=C.E3_STEPS, lr0=C.E3_LR, seed=C.S0)
    return lam, l2, risk_from_mse((b - beta_star) ** 2, a_w)

def _e4_task(lam):
    import torch
    torch.set_num_threads(4)
    rob, hum, rng = _sgd_data()
    m = e4_mlp(rob, hum, lam, l2=C.L2_N1A, width=C.E4_WIDTH,
               steps=C.E4_STEPS, lr0=C.E4_LR, batch=C.E4_BATCH, seed=C.S0)
    Xte = rng.normal(size=(C.N_TEST_MC, C.P)) * np.sqrt(lam_r)
    return lam, mc_risk_model(m, Xte, Xte @ beta_star)


def compute():
    # 理论线(201 点密网格)+ 理论 λ*(v5)
    dense = C.grid_dense()
    risk_v5_dense = theory_curve(dense, mse_v5)
    risk_t1b_dense = theory_curve(dense, mse_t1b)
    i_star = int(np.argmin(risk_v5_dense))
    lam_star_v5 = float(dense[i_star])
    risk_star_v5 = float(risk_v5_dense[i_star])
    risk_v5_at = theory_curve(lams, mse_v5)
    risk_t1b_at = theory_curve(lams, mse_t1b)

    # E1/E2:200 reps(单一 rng 流,种子 23,v3–v5 同约定)
    rng = np.random.default_rng(C.SEED)
    emp = {name: np.zeros((REPS, len(lams))) for name in ("E1", "E2")}
    t0 = time.time()
    for rep in range(REPS):
        rob, hum = gen_two_domain(rng, C.N_R, C.N_H, lam_r, lam_h,
                                  beta_star, delta, C.SIG_R2, C.SIG_H2)
        grams = e2_grams(rob, hum)
        for i, lam in enumerate(lams):
            b1 = e1_diag_ridge(rob, hum, lam, C.MU)
            b2 = e2_full_ridge(rob, hum, lam, C.MU, grams=grams)
            emp["E1"][rep, i] = risk_from_mse((b1 - beta_star) ** 2, a_w)
            emp["E2"][rep, i] = risk_from_mse((b2 - beta_star) ** 2, a_w)
        if (rep + 1) % 50 == 0:
            print(f"[E1/E2] rep {rep + 1:>3}/{REPS}", flush=True)
    mean_e1, se_e1 = mean_se(emp["E1"])
    mean_e2, se_e2 = mean_se(emp["E2"])
    rel_e1 = rel_err(mean_e1, risk_v5_at)
    rel_e2 = rel_err(mean_e2, risk_t1b_at)
    med_rel_e1 = float(np.median(rel_e1))
    med_rel_e2 = float(np.median(rel_e2))
    gate_e1 = bool(med_rel_e1 <= 0.02)
    print(f"[E1/E2] done in {time.time() - t0:.0f}s; "
          f"E1 vs v5 median rel = {med_rel_e1:.4%}", flush=True)

    # E2 配对参照(s0 数据,μ=0.05 与 μ=0)
    rob0, hum0, _ = _sgd_data()
    grams0 = e2_grams(rob0, hum0)
    risk_e2_mu = np.array([risk_from_mse(
        (e2_full_ridge(rob0, hum0, l, C.MU, grams=grams0) - beta_star) ** 2, a_w)
        for l in lams])
    risk_e2_mu0 = np.array([risk_from_mse(
        (e2_full_ridge(rob0, hum0, l, 0.0, grams=grams0) - beta_star) ** 2, a_w)
        for l in lams])
    print("[E2 paired] μ=0.05 / μ=0 on s0 data done", flush=True)

    # E3/E4:Adam 训练 9 路并行
    from concurrent.futures import ProcessPoolExecutor, as_completed
    risk_e3 = np.zeros(len(lams)); risk_e3_l0 = np.zeros(len(lams))
    risk_e4 = np.zeros(len(lams))
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=9) as ex:
        futs = {ex.submit(_e3_task, (l, C.L2_N1A)): ("e3", i)
                for i, l in enumerate(lams)}
        futs.update({ex.submit(_e3_task, (l, 0.0)): ("e3l0", i)
                     for i, l in enumerate(lams)})
        futs.update({ex.submit(_e4_task, l): ("e4", i)
                     for i, l in enumerate(lams)})
        for fut in as_completed(futs):
            kind, i = futs[fut]
            if kind == "e3":
                _, _, risk_e3[i] = fut.result()
            elif kind == "e3l0":
                _, _, risk_e3_l0[i] = fut.result()
            else:
                _, risk_e4[i] = fut.result()
            print(f"[Adam] {kind} λ={lams[i]:<4.2f} done "
                  f"({time.time() - t1:.0f}s)", flush=True)

    rel_e3_e2 = rel_err(risk_e3, risk_e2_mu)          # N1a 判据 ≤5%
    rel_e3san = rel_err(risk_e3_l0, risk_e2_mu0)      # 健全门(报告)
    rel_e4_e2 = rel_err(risk_e4, risk_e2_mu)          # E4 仅报告
    med_e3_e2 = float(np.median(rel_e3_e2))
    med_e3san = float(np.median(rel_e3san))
    med_e4_e2 = float(np.median(rel_e4_e2))
    gate_e3 = bool(med_e3_e2 <= 0.05)

    result = dict(
        experiment="N1a 风险曲线复现(全程:闭式 200 reps + Adam 粗批单 seed s0)",
        config=dict(p=C.P, s=C.S_SMOOTH, sig_r2=C.SIG_R2, sig_h2=C.SIG_H2, mu=C.MU,
                    n_r=C.N_R, n_h=C.N_H, seed=C.SEED, reps=REPS, s0=C.S0,
                    spectrum_family="A", alpha_r=ALPHA_R, alpha_h=ALPHA_H,
                    delta_shape=DELTA_SHAPE, delta_norm=DELTA_NORM,
                    anti_aligned=True, c_delta_beta=c_align,
                    risk_weights="a_j = lam_r,j = j^{-alpha_r}(Σ_r 加权)",
                    lambdas=list(map(float, lams)),
                    e3=dict(impl="Linear(p,1,no bias),零初始化,全批 GD,Adam lr=3e-3,"
                                 "20k 步 cosine;损失 Σw_i(·)²+l2‖β‖²",
                            l2=C.L2_N1A, l2_sanity=0.0),
                    e4=dict(impl="Linear(p,512)-ReLU-Linear(512,1),默认初始化,Adam lr=1e-3,"
                                 "batch 512(有放回逐步随机批),5k 步 cosine;显式 l2 同 E3,"
                                 "L2 只作用于权重矩阵;风险为 50k MC 测试风险",
                            l2=C.L2_N1A, n_test_mc=C.N_TEST_MC),
                    comparison_semantics="E3/E4 vs E2 均为同一 s0 数据上的配对比较"),
        theory=dict(
            lambda_dense=list(map(float, dense)),
            risk_v5=list(map(float, risk_v5_dense)),
            risk_t1b=list(map(float, risk_t1b_dense)),
            lambda_star_v5=lam_star_v5,
            risk_at_lambda_star_v5=risk_star_v5,
            risk_v5_at_lambdas=list(map(float, risk_v5_at)),
            risk_t1b_at_lambdas=list(map(float, risk_t1b_at)),
        ),
        E1=dict(risk_mean=list(map(float, mean_e1)), risk_se=list(map(float, se_e1)),
                theory="v5", rel_err_pointwise=list(map(float, rel_e1)),
                rel_err_median=med_rel_e1),
        E2=dict(risk_mean=list(map(float, mean_e2)), risk_se=list(map(float, se_e2)),
                theory="MSE_full,num(主阶T1-b+二阶数值修正,非已证二阶定理)", rel_err_pointwise=list(map(float, rel_e2)),
                rel_err_median=med_rel_e2),
        E3=dict(l2=C.L2_N1A, risk=list(map(float, risk_e3)),
                e2_paired_mu=list(map(float, risk_e2_mu)),
                rel_err_pointwise=list(map(float, rel_e3_e2)),
                rel_err_median=med_e3_e2),
        E3_sanity=dict(l2=0.0, risk=list(map(float, risk_e3_l0)),
                       e2_paired_mu0=list(map(float, risk_e2_mu0)),
                       rel_err_pointwise=list(map(float, rel_e3san)),
                       rel_err_median=med_e3san),
        E4=dict(l2=C.L2_N1A, risk=list(map(float, risk_e4)),
                e2_paired_mu=list(map(float, risk_e2_mu)),
                rel_err_pointwise=list(map(float, rel_e4_e2)),
                rel_err_median=med_e4_e2),
        gates=dict(
            E1_vs_v5_median_rel_err=dict(
                value=med_rel_e1, threshold=0.02,
                verdict="PASS" if gate_e1 else "FAIL",
                note="硬门:复现 v5 精度(reps=200,2026-10-25 修订)"),
            E3_sanity_l2_0_vs_E2_mu_0=dict(
                value=med_e3san, verdict="REPORT_ONLY",
                note="健全门:l2=0 充分训练应逼近 μ≈0 的 E2 曲线。"
                     "λ=0 处 p=n_r=1000 无正则 OLS 病态爆炸(E2 μ=0 风险 ~4e3),"
                     "有限步 Adam 自零初始化的隐式早停正则不会到达该解——该点为预期分歧"),
            E3_vs_E2_median_rel_err=dict(
                value=med_e3_e2, threshold=0.05,
                verdict="PASS" if gate_e3 else "FAIL",
                note="N1a 判据:E3(l2=0.05) vs E2(μ=0.05),配对 s0"),
            E2_vs_t1b_median_rel_err=dict(value=med_rel_e2, verdict="REPORT_ONLY"),
            E4_vs_E2_median_rel_err=dict(value=med_e4_e2, verdict="REPORT_ONLY"),
            lambda_star_v5=dict(value=lam_star_v5, risk=risk_star_v5),
        ),
        deviations_from_v5=[
            "β* 归一到 Σ j^{2s}β_j²=1(实施卡 A2 椭球约束);v5 脚本未归一。以实施卡为准。",
            "δ 取负号使 c=δᵀβ*<0(实施卡反对齐约定);v5 lowfreq 例为取正。以实施卡为准。",
        ],
    )
    with open(JSON_PATH, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"saved {JSON_PATH}", flush=True)
    return result


def plot(result):
    dense = np.array(result["theory"]["lambda_dense"])
    lam_star_v5 = result["theory"]["lambda_star_v5"]
    ls = np.array(result["config"]["lambdas"])
    m1 = np.array(result["E1"]["risk_mean"]); s1 = np.array(result["E1"]["risk_se"])
    m2 = np.array(result["E2"]["risk_mean"]); s2 = np.array(result["E2"]["risk_se"])
    r3 = np.array(result["E3"]["risk"]); r3l0 = np.array(result["E3_sanity"]["risk"])
    r2mu = np.array(result["E3"]["e2_paired_mu"])
    r4 = np.array(result["E4"]["risk"])
    r2m0 = np.array(result["E3_sanity"]["e2_paired_mu0"])
    g = result["gates"]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(dense, result["theory"]["risk_v5"], "-", color="C0", lw=1.8,
            label="theory v5 (feedthrough, for E1)")
    ax.plot(dense, result["theory"]["risk_t1b"], "--", color="C1", lw=1.8,
            label="full-ridge numerical surrogate (for E2)")
    ax.errorbar(ls, m1, yerr=s1, fmt="o", color="C0", ms=5, capsize=3,
                label=f"E1 diag ridge ({REPS} reps)")
    ax.errorbar(ls, m2, yerr=s2, fmt="s", color="C1", ms=5, capsize=3,
                label=f"E2 full ridge ({REPS}-replicate mean)")
    ax.plot(ls, r2mu, "-", color="black", lw=1.4,
            label="E2 full ridge (s0, paired)")
    ax.plot(ls, r3, "^", color="C2", ms=6, label=f"E3 full-batch Adam linear l2={C.L2_N1A} (s0)")
    ax.plot(ls, r4, "v", color="C3", ms=6, label=f"E4 MLP l2={C.L2_N1A} (s0)")
    ax.plot(ls, r2m0, "-.d", color="gray", lw=1.2, ms=5, alpha=0.8,
            label="E2 μ=0 (s0, sanity ref)")
    ax.plot(ls, r3l0, "x", color="C4", ms=7, label="E3 l2=0 (s0, sanity)")
    ax.axvline(lam_star_v5, color="gray", ls=":", lw=1.2,
               label=fr"theory $\lambda^*$ (v5) = {lam_star_v5:.3f}")
    # λ=0 处 E2(μ=0) 的 p=n OLS 爆炸点会压扁纵轴:裁剪并标注
    ymax = 1.18 * float(max(np.max(result["theory"]["risk_v5"]),
                            m2.max(), r3.max(), r4.max()))
    ax.set_ylim(0.0, ymax)
    off = [f"E2 μ=0, λ=0 ≈ {r2m0[0]:.2g} (off-scale)" if r2m0[0] > ymax else None,
           f"E3 l2=0, λ=0 ≈ {r3l0[0]:.2g} (off-scale)" if r3l0[0] > ymax else None]
    off = [t for t in off if t]
    if off:
        ax.annotate("\n".join(off), xy=(0.98, 0.04), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=7.5, color="dimgray")
    ax.set_xlabel(r"$\lambda$ (human-data mixing weight)")
    ax.set_ylabel(r"$\Sigma_r$-weighted risk")
    ax.set_title(
        f"N1a risk curves — family A (2.0/0.9), lowfreq anti-aligned δ "
        f"(‖δ‖=0.5, c={c_align:.3f}), n_r=1000/n_h=6000, μ=0.05\n"
        f"E1 vs v5 = {g['E1_vs_v5_median_rel_err']['value']:.3%} "
        f"({g['E1_vs_v5_median_rel_err']['verdict']} ≤2%) | "
        f"E3 vs E2 = {g['E3_vs_E2_median_rel_err']['value']:.3%} "
        f"({g['E3_vs_E2_median_rel_err']['verdict']} ≤5%)\n"
        f"E3 sanity(l2=0 vs μ=0) = {g['E3_sanity_l2_0_vs_E2_mu_0']['value']:.3%} | "
        f"E4 vs E2 = {g['E4_vs_E2_median_rel_err']['value']:.3%} (report)",
        fontsize=9)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    print(f"saved {PNG_PATH}", flush=True)


def summarize(result):
    g = result["gates"]
    print(f"\n[gates] E1 vs v5 median rel err = {g['E1_vs_v5_median_rel_err']['value']:.4%} "
          f"(≤2% hard gate, {REPS} reps): {g['E1_vs_v5_median_rel_err']['verdict']}")
    print(f"[gates] E3 sanity: E3(l2=0) vs E2(μ=0) median rel err = "
          f"{g['E3_sanity_l2_0_vs_E2_mu_0']['value']:.4%} (report; λ=0 off-scale 见 JSON note)")
    print(f"[gates] E3(l2=0.05) vs E2(μ=0.05) median rel err = "
          f"{g['E3_vs_E2_median_rel_err']['value']:.4%} (≤5%): "
          f"{g['E3_vs_E2_median_rel_err']['verdict']}")
    print(f"[gates] E2 vs T1-b median rel err = {g['E2_vs_t1b_median_rel_err']['value']:.4%} (report only)")
    print(f"[gates] E4 vs E2 median rel err = {g['E4_vs_E2_median_rel_err']['value']:.4%} (report only)")
    print(f"[gates] theory λ* (v5, dense grid) = {g['lambda_star_v5']['value']:.3f}, "
          f"risk* = {g['lambda_star_v5']['risk']:.6f}")


if __name__ == "__main__":
    if PLOT_ONLY:
        with open(JSON_PATH) as f:
            result = json.load(f)
    else:
        result = compute()
    plot(result)
    summarize(result)
