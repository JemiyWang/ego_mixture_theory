"""N1f 基失配压力测试(A1 边界标定,粗批)。

配置(实施卡 §4 N1f):谱系 A(2.0/0.9),低频 δ=0.5 反对齐(同 N1a);
n_r=1000/n_h=6000;μ=0.05(E3 显式 l2=0.05);E2(200 reps)+E3(s0=23);
λ 网格 9 点;θ∈{0,15,30,45,60,75,90}°。

旋转构造:U_θ=exp(θ_rad·A),A 只作用前 20 坐标的随机反对称矩阵
(M−Mᵀ,谱范数归一 ⇒ θ 为最大主角;固定种子 N1F_A_SEED=20261025 落档)。
人域 x_h=U_θ diag(√b_j) u、标签系数 β_h'=U_θ(β*+δ) 同步旋转
(人域自身物理不变:y_h=⟨Uβ_h,Ux⟩=⟨β_h,x⟩);机器人域不动。
实现上 yh 用未旋转数据生成后再 Xh←Xh·U_θᵀ ⇒ θ=0 数据流与 N1a 逐位一致。
对照:v5 公式忽略旋转(仍按共享基、原 b_j/δ),预测曲线与 θ 无关。

度量:每 θ 下 E2(200 reps 均值)/E3(s0)实测风险曲线 vs v5 公式曲线的相对误差
(逐 λ 与中位数);benefit 符号预测(公式固定为 N1a 的有害符号,实测逐 θ 判定)。
判据:无通过线(边界标定);报告两个 θ 阈值——公式中位误差翻倍处、
benefit 符号预测开始翻转处(公式预测恒为有害;实测变有益即翻转)。

用法:python3 run_n1f.py [--from-cache | --plot-only]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

import common as C
from gen_data import gen_two_domain
from estimators import e2_full_ridge, e2_grams, e3_sgd_linear
from theory import mse_v5, risk_from_mse

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
JSON_PATH = OUT / "n1f_basis_rotation.json"
PNG_PATH = OUT / "n1f_basis_rotation.png"
CACHE_PATH = OUT / "n1f_raw_cache.npz"
PLOT_ONLY = "--plot-only" in sys.argv
FROM_CACHE = "--from-cache" in sys.argv

THETAS_DEG = [0, 15, 30, 45, 60, 75, 90]
ROT_DIM = 20
N1F_A_SEED = 20261025
LAMS = np.array(C.LAMBDAS)
REPS = C.REPS_CLOSED

# --- N1a 配置 ---
ALPHA_R, ALPHA_H = C.SPECTRA["A"]
BETA_STAR = C.make_beta_star()
LAM_R, LAM_H = C.make_spectra(ALPHA_R, ALPHA_H)
DELTA = C.make_delta("lowfreq", 0.5, BETA_STAR, anti_align=True)
BETA_H = BETA_STAR + DELTA

# --- 固定随机反对称 A(前 20 坐标,谱范数归一) ---
_rngA = np.random.default_rng(N1F_A_SEED)
_M = _rngA.normal(size=(ROT_DIM, ROT_DIM))
_A20 = _M - _M.T
_A20 = _A20 / np.linalg.norm(_A20, 2)
_U_CACHE = {}


def make_U(theta_deg):
    if theta_deg not in _U_CACHE:
        U = np.eye(C.P)
        if theta_deg != 0:
            U[:ROT_DIM, :ROT_DIM] = expm(np.deg2rad(theta_deg) * _A20)
        _U_CACHE[theta_deg] = U
    return _U_CACHE[theta_deg]


# --- 理论对照:v5 公式忽略旋转(θ 无关),即 N1a 配置 ---
THEORY_ARGS = (LAM_R, LAM_H, DELTA, BETA_STAR, BETA_H,
               C.N_R, C.N_H, C.SIG_R2, C.SIG_H2, C.MU)
DENSE = C.grid_dense()
THEORY_DENSE = np.array([risk_from_mse(mse_v5(l, *THEORY_ARGS), LAM_R) for l in DENSE])
THEORY_AT9 = np.array([risk_from_mse(mse_v5(l, *THEORY_ARGS), LAM_R) for l in LAMS])
THEORY_BENEFIT = float(THEORY_AT9[0] - THEORY_DENSE.min())
THEORY_SIGN = 1 if THEORY_BENEFIT > 0 else -1     # N1a 配置:有害(−1),λ*=0


def gen_rotated(rng, theta_deg):
    """θ=0 时与 gen_two_domain 逐位一致(先生成再转 Xh)。"""
    rob, hum = gen_two_domain(rng, C.N_R, C.N_H, LAM_R, LAM_H,
                              BETA_STAR, DELTA, C.SIG_R2, C.SIG_H2)
    if theta_deg != 0:
        hum = (hum[0] @ make_U(theta_deg).T, hum[1])
    return rob, hum


def overlap_c(theta_deg):
    """诊断:c(θ)=β*ᵀU_θ(β*+δ)(旋转使人域系数偏离机器人基的程度)"""
    return float(BETA_STAR @ (make_U(theta_deg) @ BETA_H))


# ---------------- workers ----------------
def _e2_cell(theta):
    rng = np.random.default_rng(C.SEED)
    risks = np.zeros((REPS, len(LAMS)))
    for rep in range(REPS):
        rob, hum = gen_rotated(rng, theta)
        grams = e2_grams(rob, hum)
        for i, lam in enumerate(LAMS):
            b2 = e2_full_ridge(rob, hum, lam, C.MU, grams=grams)
            risks[rep, i] = risk_from_mse((b2 - BETA_STAR) ** 2, LAM_R)
    return theta, risks


def _e3_task(args):
    theta, lam = args
    import torch
    torch.set_num_threads(4)
    rng = np.random.default_rng(C.S0)
    rob, hum = gen_rotated(rng, theta)
    b = e3_sgd_linear(rob, hum, lam, l2=C.L2_N1A,
                      steps=C.E3_STEPS, lr0=C.E3_LR, seed=C.S0)
    return theta, lam, risk_from_mse((b - BETA_STAR) ** 2, LAM_R)


def compute_raw():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.environ.update(OPENBLAS_NUM_THREADS="8", OMP_NUM_THREADS="8",
                      MKL_NUM_THREADS="8")
    mp = __import__("multiprocessing").get_context("spawn")
    raw = {"e2": {}, "e3": {}}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=7, mp_context=mp) as ex:
        futs = {ex.submit(_e2_cell, th): th for th in THETAS_DEG}
        for fut in as_completed(futs):
            th, risks = fut.result()
            raw["e2"][th] = risks
            print(f"[E2] θ={th}° done ({time.time()-t0:.0f}s)", flush=True)
    for th in THETAS_DEG:
        raw["e3"][th] = np.zeros(len(LAMS))
    tasks = [(th, float(l)) for th in THETAS_DEG for l in LAMS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=14, mp_context=mp) as ex:
        futs = {ex.submit(_e3_task, t): t for t in tasks}
        for fut in as_completed(futs):
            th, lam_r, risk = fut.result()
            raw["e3"][th][int(np.argmin(np.abs(LAMS - lam_r)))] = risk
            print(f"[E3] θ={th}° λ={lam_r:<4.2f} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(CACHE_PATH,
             **{f"e2::{th}": raw["e2"][th] for th in THETAS_DEG},
             **{f"e3::{th}": raw["e3"][th] for th in THETAS_DEG})
    print(f"saved {CACHE_PATH}", flush=True)
    return raw


def load_cache():
    z = np.load(CACHE_PATH)
    raw = {"e2": {}, "e3": {}}
    for key in z.files:
        k, th = key.split("::")
        raw[k][int(th)] = z[key]
    return raw


def first_crossing(vals, thresh):
    for th, v in zip(THETAS_DEG, vals):
        if v >= thresh:
            return th
    return None


def assemble(raw):
    per_theta = []
    for th in THETAS_DEG:
        risks = raw["e2"][th]
        m2, s2 = risks.mean(0), risks.std(0, ddof=1) / np.sqrt(REPS)
        r3 = raw["e3"][th]
        rel2 = np.abs(m2 - THEORY_AT9) / THEORY_AT9
        rel3 = np.abs(r3 - THEORY_AT9) / THEORY_AT9
        ben2 = float(m2[0] - m2.min())
        ben3 = float(r3[0] - r3.min())
        per_theta.append(dict(
            theta=th, overlap_c=overlap_c(th),
            E2=dict(risk_mean=list(map(float, m2)), risk_se=list(map(float, s2)),
                    rel_err_pointwise=list(map(float, rel2)),
                    rel_err_median=float(np.median(rel2)),
                    benefit=ben2, sign=(1 if ben2 > 0 else -1),
                    lambda_star_9pt=float(LAMS[int(m2.argmin())])),
            E3=dict(risk=list(map(float, r3)),
                    rel_err_pointwise=list(map(float, rel3)),
                    rel_err_median=float(np.median(rel3)),
                    benefit=ben3, sign=(1 if ben3 > 0 else -1),
                    lambda_star_9pt=float(LAMS[int(r3.argmin())])),
        ))
    med2 = [c["E2"]["rel_err_median"] for c in per_theta]
    med3 = [c["E3"]["rel_err_median"] for c in per_theta]
    thresholds = {}
    for est, meds, key in (("E2", med2, "E2"), ("E3", med3, "E3")):
        base = meds[0]
        thresholds[key] = dict(
            err_at_theta0=base,
            err_doubling_theta=first_crossing(meds, 2 * base),
            sign_flip_theta=next((c["theta"] for c in per_theta
                                  if c[key]["sign"] != THEORY_SIGN), None),
        )
    result = dict(
        experiment="N1f 基失配压力测试(谱系 A,低频 δ=0.5 反对齐;公式忽略旋转)",
        config=dict(p=C.P, mu=C.MU, l2=C.L2_N1A, n_r=C.N_R, n_h=C.N_H,
                    seed=C.SEED, reps=REPS, s0=C.S0, thetas_deg=THETAS_DEG,
                    rot_dim=ROT_DIM, rot_seed=N1F_A_SEED,
                    rot_construction="A=(M−Mᵀ)/‖M−Mᵀ‖₂ 作用前 20 坐标;U=exp(θ_rad·A);"
                                     "yh 先生成后转 Xh ⇒ θ=0 与 N1a 数据流逐位一致",
                    lambdas=list(map(float, LAMS)),
                    theory_semantics="v5 公式忽略旋转,曲线 θ 无关"),
        theory=dict(risk_at_lambdas=list(map(float, THEORY_AT9)),
                    benefit=THEORY_BENEFIT, sign=THEORY_SIGN,
                    lambda_star=0.0),
        per_theta=per_theta,
        thresholds=thresholds,
        gates=dict(verdict="REPORT_ONLY",
                   note="N1f 为边界标定,无通过线;报告误差翻倍与符号翻转的 θ 阈值"),
    )
    with open(JSON_PATH, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"saved {JSON_PATH}", flush=True)
    return result


def plot(result):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axL.plot(DENSE, THEORY_DENSE, "k-", lw=2.0, label="theory v5 (θ-independent)")
    show = [0, 30, 60, 90]
    cmap = plt.get_cmap("viridis")
    for i, th in enumerate(show):
        c = next(c for c in result["per_theta"] if c["theta"] == th)
        color = cmap(i / (len(show) - 1))
        axL.plot(LAMS, c["E2"]["risk_mean"], "o", color=color, ms=5,
                 label=f"E2 θ={th}°")
        axL.plot(LAMS, c["E3"]["risk"], "--", color=color, lw=1.0, alpha=0.75,
                 label=f"E3 θ={th}°")
    axL.set_xlabel(r"$\lambda$"); axL.set_ylabel(r"$\Sigma_r$-weighted risk")
    axL.set_title("risk curves at representative θ vs theory")
    axL.legend(fontsize=7.5, ncol=2); axL.grid(alpha=0.3)

    ths = [c["theta"] for c in result["per_theta"]]
    for est, color, mk in (("E2", "C1", "s"), ("E3", "C2", "^")):
        meds = [c[est]["rel_err_median"] * 100 for c in result["per_theta"]]
        axR.plot(ths, meds, mk + "-", color=color, ms=6, lw=1.3,
                 label=f"{est} median rel err vs v5")
        base = result["thresholds"][est]["err_at_theta0"]
        axR.axhline(2 * base * 100, color=color, ls=":", lw=1.0,
                    label=f"{est} 2× baseline ({base * 100:.2f}%)")
        flips = [c["theta"] for c in result["per_theta"]
                 if c[est]["sign"] != result["theory"]["sign"]]
        for f in flips:
            axR.axvline(f, color=color, ls="--", lw=1.0, alpha=0.6)
        if flips:
            axR.plot([], [], color=color, ls="--",
                     label=f"{est} sign flip θ={flips[0]}°")
    axR.set_xlabel(r"$\theta$ (deg)"); axR.set_ylabel("median rel err vs v5 (%)")
    axR.set_title("formula error growth & benefit-sign flips vs θ")
    axR.legend(fontsize=7.5); axR.grid(alpha=0.3)
    t = result["thresholds"]
    fig.suptitle(
        f"N1f basis rotation — doubling θ: E2 {t['E2']['err_doubling_theta']}°, "
        f"E3 {t['E3']['err_doubling_theta']}° | sign-flip θ: E2 {t['E2']['sign_flip_theta']}°, "
        f"E3 {t['E3']['sign_flip_theta']}° (formula sign fixed: "
        f"{'beneficial' if result['theory']['sign'] > 0 else 'harmful'})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PNG_PATH, dpi=140)
    print(f"saved {PNG_PATH}", flush=True)


def summarize(result):
    t = result["thresholds"]
    print("\n=== N1f (boundary calibration, no pass line) ===")
    print(f"theory sign fixed: {'beneficial' if result['theory']['sign'] > 0 else 'harmful'} "
          f"(benefit={result['theory']['benefit']:.2e}, λ*=0)")
    for est in ("E2", "E3"):
        print(f"[{est}] median rel err vs θ: " +
              " ".join(f"{c['theta']}°:{c[est]['rel_err_median'] * 100:.2f}%"
                       for c in result["per_theta"]))
        print(f"     doubling θ = {t[est]['err_doubling_theta']}, "
              f"sign-flip θ = {t[est]['sign_flip_theta']}")
    print("\nper-θ: overlap c(θ) | E2 benefit/sign | E3 benefit/sign")
    for c in result["per_theta"]:
        print(f"  θ={c['theta']:>3}° c={c['overlap_c']:+.4f} | "
              f"E2 {c['E2']['benefit']:+.5f}({c['E2']['sign']:+d}) | "
              f"E3 {c['E3']['benefit']:+.5f}({c['E3']['sign']:+d})")


if __name__ == "__main__":
    if PLOT_ONLY:
        with open(JSON_PATH) as f:
            result = json.load(f)
    else:
        raw = load_cache() if FROM_CACHE else compute_raw()
        result = assemble(raw)
    plot(result)
    summarize(result)
