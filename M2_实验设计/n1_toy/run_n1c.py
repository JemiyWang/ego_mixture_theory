"""N1c 剂量律形状(粗批;2026-10-25 修订量程 n_h≤32000 后重赛)。

配置(实施卡 §4 N1c,2026-10-25 修订):匹配谱 α_r=α_h=2.0;白色**对齐** δ
(j^0 归一、取正号 c=δᵀβ*>0,‖δ‖=0.3);n_r=1000;
n_h∈{30,60,120,250,500,1000,2000,4000,8000,16000,32000}(上限 ≈8× 理论半饱和点 3865);
E2(200 reps)+E3(s0=23,显式 l2=0.05);λ 网格 9 点;p=1000、s=1.0、σ²=1.0/4.0、μ=0.05。

度量(N1b 修正口径):每 n_h 下 g(n_h)=R(0)−min_λ R(λ),在估计器均值(E2)/
单 seed(E3)曲线上取 min,不逐 rep 先取 min。E2 的 g 对 E2 自洽
(E2 自己的 R(0) 与 E2 自己的 min R;E2 基线相对 v5 有 T1-b 偏移,见 N1a/d3);
理论线照旧用 v5 公式(201 点密网格)。
理论双曲参数(匹配谱二次型系数,口径移植自 M1 verify_dosage_law.py;
u=aδ,w=μβ*,vr=σ_r²a/n_r,D0=a+μ):
  A1 = Σ a·(−2(vr+u·w))/D0²,A2r = Σ a(vr+u²)/D0²,B = Σ a(σ_h²a+2u²)/D0²
  g(n_h)=A1²/(4(A2r+B/n_h)),λ*(n_h)=−A1/(2(A2r+B/n_h)),n_h^{1/2}=B/A2r
  (二次型系数不含馈通/VarD 修正,与参照一致;v5 曲线含)。

拟合竞赛(线性尺度 R²):① 双曲 a/(b+c/n_h)(半饱和点 c/b)② 指数饱和
g_inf(1−e^{−n_h/τ})③ 对数 a·log(1+n_h/b)。
1/n_h 线性检验:卡的等价表述"双曲 ⟺ 对 1/n_h 严格线性"的数学正确形式是
**1/g 对 1/n_h 线性**(1/g = 4(A2r+B/n_h)/A1²);字面 g~1/n_h 对严格双曲的
R² 仅 ~0.34(数值实证),故判据落在 1/g~1/n_h 上,字面版同时报告并标注。
g=0 的格(λ-网格退化:n_h=30/60 的 λ*<0.05 不可分辨)不参与 1/g 拟合。

判据(预注册,修订):E2——双曲 R² 三者最高 ∧ 半饱和点与理论差 ≤2 倍
∧ 1/g~1/n_h 线性 R²≥0.995;E3 粗批仅报告(g 量级 ≈ 单 seed 噪声,不判定)。

用法:python3 run_n1c.py               # 增量计算(只补缓存缺的格)+ 落盘 + 绘图
     python3 run_n1c.py --from-cache   # 全部用缓存重组 JSON/图
     python3 run_n1c.py --plot-only    # 只从已存 JSON 重画图
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
from scipy.optimize import curve_fit

import common as C
from gen_data import gen_two_domain
from estimators import e2_full_ridge, e2_grams, e3_sgd_linear
from theory import mse_v5, mse_t1b, risk_from_mse

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
JSON_PATH = OUT / "n1c_dosage.json"
PNG_PATH = OUT / "n1c_dosage.png"
CACHE_PATH = OUT / "n1c_raw_cache.npz"
PLOT_ONLY = "--plot-only" in sys.argv
FROM_CACHE = "--from-cache" in sys.argv

N_HS = [30, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
DELTA_NORM, DELTA_SHAPE = 0.3, "white"
LAMS = np.array(C.LAMBDAS)
REPS = C.REPS_CLOSED
LIN_R2_THRESHOLD = 0.995

BETA_STAR = C.make_beta_star()
LAM_R = LAM_H = C.make_spectra(2.0, 2.0)[0]          # 匹配谱 α_r=α_h=2.0
DELTA = C.make_delta(DELTA_SHAPE, DELTA_NORM, BETA_STAR, anti_align=False)
BETA_H = BETA_STAR + DELTA
C_ALIGN = float(DELTA @ BETA_STAR)


def quad_coeffs():
    """匹配谱二次型系数(verify_dosage_law.py 口径;n_h 无关部分)"""
    a = LAM_R
    D0 = a + C.MU
    u = a * DELTA
    w = C.MU * BETA_STAR
    vr = C.SIG_R2 * a / C.N_R
    A1 = float((a * (-2 * (vr + u * w)) / D0**2).sum())
    A2r = float((a * (vr + u**2) / D0**2).sum())
    B = float((a * (C.SIG_H2 * a + 2 * u**2) / D0**2).sum())
    return A1, A2r, B


A1, A2R, B_COEF = quad_coeffs()
NH_HALF_THEORY = B_COEF / A2R


def g_hyper_law(nh):
    return A1**2 / (4 * (A2R + B_COEF / nh))


def lam_star_law(nh):
    return float(np.clip(-A1 / (2 * (A2R + B_COEF / nh)), 0.0, 1.0))


def theory_cell(nh):
    args = (LAM_R, LAM_H, DELTA, BETA_STAR, BETA_H,
            C.N_R, nh, C.SIG_R2, C.SIG_H2, C.MU)
    dense = C.grid_dense()
    curve = np.array([risk_from_mse(mse_v5(l, *args), LAM_R) for l in dense])
    curve_t1b = np.array([risk_from_mse(mse_t1b(l, *args), LAM_R) for l in dense])
    return dict(risk_dense=list(map(float, curve)),
                g=float(curve[0] - curve.min()),
                g_t1b=float(curve_t1b[0] - curve_t1b.min()),
                g_hyper=float(g_hyper_law(nh)),
                lambda_star=float(dense[int(np.argmin(curve))]),
                lambda_star_law=lam_star_law(nh))


# ---------------- workers ----------------
def _e2_cell(nh):
    rng = np.random.default_rng(C.SEED)   # 各 n_h 格共享同一随机流 ⇒ 跨格配对
    risks = np.zeros((REPS, len(LAMS)))
    for rep in range(REPS):
        rob, hum = gen_two_domain(rng, C.N_R, nh, LAM_R, LAM_H,
                                  BETA_STAR, DELTA, C.SIG_R2, C.SIG_H2)
        grams = e2_grams(rob, hum)
        for i, lam in enumerate(LAMS):
            b2 = e2_full_ridge(rob, hum, lam, C.MU, grams=grams)
            risks[rep, i] = risk_from_mse((b2 - BETA_STAR) ** 2, LAM_R)
    return nh, risks


def _e3_task(args):
    nh, lam = args
    import torch
    torch.set_num_threads(4)
    rng = np.random.default_rng(C.S0)
    rob, hum = gen_two_domain(rng, C.N_R, nh, LAM_R, LAM_H,
                              BETA_STAR, DELTA, C.SIG_R2, C.SIG_H2)
    b = e3_sgd_linear(rob, hum, lam, l2=C.L2_N1A,
                      steps=C.E3_STEPS, lr0=C.E3_LR, seed=C.S0)
    return nh, lam, risk_from_mse((b - BETA_STAR) ** 2, LAM_R)


def compute_raw():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.environ.update(OPENBLAS_NUM_THREADS="8", OMP_NUM_THREADS="8",
                      MKL_NUM_THREADS="8")
    mp = __import__("multiprocessing").get_context("spawn")
    raw = load_cache() if CACHE_PATH.exists() else {"e2": {}, "e3": {}}
    raw.setdefault("e2", {}); raw.setdefault("e3", {})
    miss2 = [nh for nh in N_HS if nh not in raw["e2"]]
    miss3 = [nh for nh in N_HS if nh not in raw["e3"]]
    print(f"[cache] reuse e2={sorted(raw['e2'])} e3={sorted(raw['e3'])}; "
          f"compute missing e2={miss2} e3={miss3}", flush=True)
    if miss2:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=9, mp_context=mp) as ex:
            futs = {ex.submit(_e2_cell, nh): nh for nh in miss2}
            for fut in as_completed(futs):
                nh, risks = fut.result()
                raw["e2"][nh] = risks
                print(f"[E2] n_h={nh} done ({time.time()-t0:.0f}s)", flush=True)
    if miss3:
        for nh in miss3:
            raw["e3"][nh] = np.zeros(len(LAMS))
        tasks = [(nh, float(l)) for nh in miss3 for l in LAMS]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=14, mp_context=mp) as ex:
            futs = {ex.submit(_e3_task, t): t for t in tasks}
            for fut in as_completed(futs):
                nh, lam_r, risk = fut.result()
                raw["e3"][nh][int(np.argmin(np.abs(LAMS - lam_r)))] = risk
                print(f"[E3] n_h={nh} λ={lam_r:<4.2f} ({time.time()-t0:.0f}s)",
                      flush=True)
    np.savez(CACHE_PATH,
             **{f"e2::{nh}": raw["e2"][nh] for nh in N_HS},
             **{f"e3::{nh}": raw["e3"][nh] for nh in N_HS},
             **{f"e2dense41::{nh}": a for nh, a in raw.get("e2dense41", {}).items()})
    print(f"saved {CACHE_PATH}", flush=True)
    return raw


def load_cache():
    z = np.load(CACHE_PATH)
    raw = {"e2": {}, "e3": {}, "e2dense41": {}}
    for key in z.files:
        k, nh = key.split("::")
        raw[k][int(nh)] = z[key]
    return raw


# ---------------- 拟合竞赛 ----------------
def _r2(g, pred):
    ss_res = float(np.sum((g - pred) ** 2))
    ss_tot = float(np.sum((g - g.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-300)


def fit_all(nh, g):
    nh = np.asarray(nh, float)
    g = np.asarray(g, float)
    out = {}

    def try_fit(fn, p0):
        try:
            p, _ = curve_fit(fn, nh, g, p0=p0, maxfev=60000)
            return p, _r2(g, fn(nh, *p))
        except Exception:
            return None, float("nan")

    p, r2 = try_fit(lambda x, a, b, c: a / (b + c / x),
                    [max(g.max(), 1e-8), 1.0, float(np.median(nh))])
    out["hyperbola"] = dict(params=None if p is None else list(map(float, p)),
                            r2=r2,
                            half_sat=(float(p[2] / p[1])
                                      if p is not None and p[1] > 0 else float("nan")))
    p, r2 = try_fit(lambda x, gi, tau: gi * (1 - np.exp(-x / tau)),
                    [max(g.max(), 1e-8), 500.0])
    out["exp"] = dict(params=None if p is None else list(map(float, p)), r2=r2)
    p, r2 = try_fit(lambda x, a, b: a * np.log1p(x / b),
                    [max(g.max(), 1e-8) / 5.0, 100.0])
    out["log"] = dict(params=None if p is None else list(map(float, p)), r2=r2)
    return out


def fit_inv_linear(nh, g):
    """1/n_h 线性检验:判据版 = 1/g 对 1/n_h(与双曲严格等价);
    字面版 = g 对 1/n_h(对严格双曲仅 R²≈0.34,见 config 注记)。g≤0 的格剔除。"""
    nh = np.asarray(nh, float)
    g = np.asarray(g, float)
    m = g > 0
    x, y = 1.0 / nh[m], 1.0 / g[m]
    p = np.polyfit(x, y, 1)
    r2_inv = _r2(y, np.polyval(p, x))
    pl = np.polyfit(1.0 / nh, g, 1)
    r2_lit = _r2(g, np.polyval(pl, 1.0 / nh))
    return dict(r2_inv_linear=float(r2_inv), r2_literal_g_vs_inv=float(r2_lit),
                n_used=int(m.sum()), n_total=len(g),
                excluded_zero_g=[float(v) for v in nh[~m]],
                params_inv=list(map(float, p)))


def fit_curve(kind, params, nh_grid):
    nh_grid = np.asarray(nh_grid, float)
    if params is None:
        return np.full_like(nh_grid, np.nan)
    if kind == "hyperbola":
        a, b, c = params
        return a / (b + c / nh_grid)
    if kind == "exp":
        gi, tau = params
        return gi * (1 - np.exp(-nh_grid / tau))
    a, b = params
    return a * np.log1p(nh_grid / b)


def judge(fits, lin):
    r2s = {k: fits[k]["r2"] for k in ("hyperbola", "exp", "log")}
    hyper_best = r2s["hyperbola"] > r2s["exp"] and r2s["hyperbola"] > r2s["log"]
    hs = fits["hyperbola"]["half_sat"]
    ratio = hs / NH_HALF_THEORY if np.isfinite(hs) else float("nan")
    half_ok = bool(np.isfinite(ratio) and 0.5 <= ratio <= 2.0)
    lin_ok = bool(lin["r2_inv_linear"] >= LIN_R2_THRESHOLD)
    ok = hyper_best and half_ok and lin_ok
    return dict(r2=r2s, half_sat_fit=hs, half_sat_theory=NH_HALF_THEORY,
                half_sat_ratio=ratio, hyperbola_best=bool(hyper_best),
                half_sat_within_2x=half_ok,
                r2_inv_linear=lin["r2_inv_linear"],
                r2_inv_linear_threshold=LIN_R2_THRESHOLD,
                r2_inv_linear_ok=lin_ok,
                verdict="PASS" if ok else "FAIL")


def assemble(raw):
    per_nh = []
    for nh in N_HS:
        risks = raw["e2"][nh]
        mean, se = risks.mean(0), risks.std(0, ddof=1) / np.sqrt(REPS)
        r3 = raw["e3"][nh]
        th = theory_cell(nh)
        per_nh.append(dict(
            n_h=nh, theory=th,
            E2=dict(risk_mean=list(map(float, mean)), risk_se=list(map(float, se)),
                    risk_reps=list(map(float, risks.ravel())),
                    g=float(mean[0] - mean.min()),
                    lambda_star_9pt=float(LAMS[int(mean.argmin())])),
            E3=dict(risk=list(map(float, r3)), g=float(r3[0] - r3.min()),
                    lambda_star_9pt=float(LAMS[int(r3.argmin())])),
        ))
    g_e2 = np.array([c["E2"]["g"] for c in per_nh])
    g_e3 = np.array([c["E3"]["g"] for c in per_nh])
    g_th = np.array([c["theory"]["g"] for c in per_nh])
    fits_e2 = fit_all(N_HS, g_e2)
    fits_e3 = fit_all(N_HS, g_e3)
    lin_e2 = fit_inv_linear(N_HS, g_e2)
    lin_e3 = fit_inv_linear(N_HS, g_e3)

    # 诊断(不改预注册判据):
    #  d1/d1b 对 v5 / T1-b 理论 g 做同一三族竞赛(无噪声对照,检验量程可分辨性);
    #  d2 去掉 λ-网格退化零点(n_h=30/60,λ*<0.05)后的 E2 敏感性;
    #  d3 E2 双曲/指数拟合逐点残差;
    #  d4 E2 在 41 点密 λ 网格的 benefit(缓存有的格)。
    fits_theory_v5 = fit_all(N_HS, g_th)
    fits_theory_t1b = fit_all(N_HS, [c["theory"]["g_t1b"] for c in per_nh])
    mask = np.array([c["n_h"] >= 120 for c in per_nh])
    fits_e2_trim = fit_all(np.array(N_HS)[mask], g_e2[mask])
    nh_arr = np.asarray(N_HS, float)
    resid = dict(
        hyper=list(map(float, g_e2 - fit_curve("hyperbola", fits_e2["hyperbola"]["params"], nh_arr))),
        exp=list(map(float, g_e2 - fit_curve("exp", fits_e2["exp"]["params"], nh_arr))))
    d4 = None
    if raw.get("e2dense41"):
        have = sorted(raw["e2dense41"])
        D41 = np.linspace(0.0, 1.0, 41)
        g_d = [float((lambda m: m[0] - m.min())(raw["e2dense41"][nh].mean(0)))
               for nh in have]
        lam_d = [float(D41[int(raw["e2dense41"][nh].mean(0).argmin())]) for nh in have]
        fits_d = fit_all(have, g_d)
        d4 = dict(n_hs=have, lams=list(map(float, D41)), g=g_d, lambda_star=lam_d,
                  fits=fits_d, judge=judge(fits_d, fit_inv_linear(have, g_d)),
                  note="只覆盖缓存中的旧格(≤8000);新增 16000/32000 未补 dense 网格")
    #  d5 1/g 线性检验的量程敏感性(逐步剔除低端小格):分解杠杆点与 p≈n 残差的贡献;
    #  d6 每格 E2 g 与 v5/T1-b 理论 g 的逐格比值(T1-b 偏移的自洽性)。
    nh_pos = [c["n_h"] for c in per_nh if c["E2"]["g"] > 0]
    g_pos = [c["E2"]["g"] for c in per_nh if c["E2"]["g"] > 0]
    d5 = []
    for lo in (30, 120, 250, 500, 1000):
        sel = [i for i, v in enumerate(nh_pos) if v >= lo]
        if len(sel) >= 4:
            r = fit_inv_linear([nh_pos[i] for i in sel], [g_pos[i] for i in sel])
            d5.append(dict(min_nh=lo, n_used=r["n_used"],
                           r2_inv_linear=r["r2_inv_linear"]))
    g_t1b = np.array([c["theory"]["g_t1b"] for c in per_nh])
    d6 = [dict(n_h=c["n_h"], g_e2=c["E2"]["g"], g_v5=c["theory"]["g"],
               g_t1b=float(gt), ratio_e2_over_v5=c["E2"]["g"] / max(c["theory"]["g"], 1e-300),
               ratio_e2_over_t1b=c["E2"]["g"] / max(gt, 1e-300))
          for c, gt in zip(per_nh, g_t1b)]

    gates = dict(
        E2=judge(fits_e2, lin_e2),
        E3=dict(r2={k: fits_e3[k]["r2"] for k in ("hyperbola", "exp", "log")},
                half_sat_fit=fits_e3["hyperbola"]["half_sat"],
                half_sat_theory=NH_HALF_THEORY,
                r2_inv_linear=lin_e3["r2_inv_linear"],
                verdict="REPORT_ONLY",
                note="修订:E3 粗批仅报告(g 量级 ≈ 单 seed 噪声,形状判别留验证批)"),
        A1_negative=dict(A1=A1, verdict="PASS" if A1 < 0 else "FAIL",
                         note="前提 A1=R̄'(0⁺)<0(白色对齐 δ 下 u·w=aδ·μβ*>0 保证)"),
        monotonicity=dict(
            E2_g_nondecreasing=bool(np.all(np.diff(g_e2) >= -1e-12)),
            E3_g_nondecreasing=bool(np.all(np.diff(g_e3) >= -1e-12)),
            E2_g_positive=bool(np.all(g_e2 > 0)),
            E3_g_positive=bool(np.all(g_e3 > 0)),
            note="n_h=30/60 的 g=0 是 λ-网格退化(λ*≈0.01–0.015<0.05)所致,已记录"),
    )
    result = dict(
        experiment="N1c 剂量律形状(匹配谱 2.0/2.0,白色对齐 δ,‖δ‖=0.3;"
                   "2026-10-25 修订量程 n_h≤32000 重赛)",
        config=dict(p=C.P, s=C.S_SMOOTH, sig_r2=C.SIG_R2, sig_h2=C.SIG_H2, mu=C.MU,
                    l2=C.L2_N1A, n_r=C.N_R, n_hs=N_HS, seed=C.SEED, reps=REPS, s0=C.S0,
                    alpha_r=2.0, alpha_h=2.0, delta_shape=DELTA_SHAPE,
                    delta_norm=DELTA_NORM, aligned=True, c_delta_beta=C_ALIGN,
                    lambdas=list(map(float, LAMS)),
                    benefit_semantics="N1b 修正口径:估计器均值/单 seed 曲线上取 min_λ;"
                                      "E2 的 g 对 E2 自洽(E2 自己的 R(0) 与 min R)",
                    theory_line_semantics="理论线照旧为 v5 公式;E2 基线 R(0) 相对 v5 有 "
                                          "T1-b 偏移(N1a:E2 不吃馈通项),两者关系即此偏移",
                    linearity_semantics="卡的「双曲⟺对1/n_h严格线性」的数学正确形式为 "
                                        "1/g 对 1/n_h(判据所用);字面 g~1/n_h 对严格双曲 "
                                        "R²≈0.34(数值实证),同时报告于 fits 的 r2_literal",
                    note="二次型系数不含馈通/VarD(与 verify_dosage_law.py 一致);"
                         "v5 曲线含。各 n_h 格共享 default_rng(23) 流(跨格配对,"
                         "n_h 格的人域数据为小格的前缀)。缓存增量复用"),
        theory_coeffs=dict(A1=A1, A2r=A2R, B=B_COEF, nh_half_theory=NH_HALF_THEORY,
                           g_inf_theory=float(A1**2 / (4 * A2R))),
        per_nh=per_nh,
        fits=dict(E2=fits_e2, E3=fits_e3, linearity=dict(E2=lin_e2, E3=lin_e3)),
        diagnostics=dict(
            note="不改变预注册判据;d1/d1b 为无噪声理论对照",
            d1_fits_on_theory_v5_g=fits_theory_v5,
            d1b_fits_on_theory_t1b_g=fits_theory_t1b,
            d2_fits_E2_without_nh30_60=fits_e2_trim,
            d3_E2_pointwise_residuals=resid,
            d4_E2_dense41_lambda=d4,
            d5_inv_linear_range_sensitivity=d5,
            d6_E2_g_vs_theory_per_nh=d6,
        ),
        gates=gates,
    )
    with open(JSON_PATH, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"saved {JSON_PATH}", flush=True)
    return result


def plot(result):
    nh = np.array([c["n_h"] for c in result["per_nh"]], float)
    nh_grid = np.geomspace(nh.min(), nh.max(), 200)
    g_th = np.array([c["theory"]["g"] for c in result["per_nh"]])
    g_e2 = np.array([c["E2"]["g"] for c in result["per_nh"]])
    g_e3 = np.array([c["E3"]["g"] for c in result["per_nh"]])
    d4 = result.get("diagnostics", {}).get("d4_E2_dense41_lambda")

    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(17.5, 5.0))
    # 左:g(n_h)
    axL.plot(nh_grid, [g_hyper_law(x) for x in nh_grid], "-", color="gray", lw=1.4,
             label=fr"hyperbola law $A_1^2/4(A_2^r+B/n_h)$")
    axL.plot(nh, g_th, "o-", color="C0", ms=5, lw=1.5, label="theory v5 (dense)")
    axL.plot(nh, g_e2, "s", color="C1", ms=6, label="E2 (200 reps)")
    axL.plot(nh, g_e3, "^", color="C2", ms=6, label="E3 (s0)")
    for kind, ls, lb in (("hyperbola", "-", "fit hyperbola"),
                         ("exp", "--", "fit exp"), ("log", ":", "fit log")):
        p = result["fits"]["E2"][kind]["params"]
        axL.plot(nh_grid, fit_curve(kind, p, nh_grid), ls, color="C1", lw=1.1,
                 alpha=0.75, label=f"E2 {lb}")
    p = result["fits"]["E3"]["hyperbola"]["params"]
    axL.plot(nh_grid, fit_curve("hyperbola", p, nh_grid), "-.", color="C2", lw=1.1,
             alpha=0.8, label="E3 fit hyperbola")
    axL.axvline(NH_HALF_THEORY, color="gray", ls=":", lw=1.0,
                label=fr"theory $n_h^{{1/2}}$={NH_HALF_THEORY:.0f}")
    axL.set_xscale("log")
    axL.set_xlabel(r"$n_h$"); axL.set_ylabel(r"$g(n_h)=R(0)-\min_\lambda R(\lambda)$")
    axL.set_title("dose law g(n_h): measured vs theory vs fits")
    axL.legend(fontsize=7); axL.grid(alpha=0.3, which="both")

    # 中:λ*(n_h)
    axM.plot(nh_grid, [lam_star_law(x) for x in nh_grid], "-", color="gray", lw=1.4,
             label=r"law $-A_1/2A_2(n_h)$")
    axM.plot(nh, [c["theory"]["lambda_star"] for c in result["per_nh"]],
             "o-", color="C0", ms=5, lw=1.5, label="theory v5 argmin (dense)")
    axM.plot(nh, [c["E2"]["lambda_star_9pt"] for c in result["per_nh"]],
             "s--", color="C1", ms=6, label="E2 argmin (9-pt)")
    axM.plot(nh, [c["E3"]["lambda_star_9pt"] for c in result["per_nh"]],
             "^--", color="C2", ms=6, label="E3 argmin (9-pt)")
    if d4:
        axM.plot(d4["n_hs"], d4["lambda_star"], "D-", color="C4", ms=4, lw=1.0,
                 mfc="none", label="E2 argmin (dense-41λ, ≤8000)")
    axM.set_xscale("log")
    axM.set_xlabel(r"$n_h$"); axM.set_ylabel(r"$\lambda^*(n_h)$")
    axM.set_title(r"$\lambda^*(n_h)$ growth: measured vs theory")
    axM.legend(fontsize=7); axM.grid(alpha=0.3, which="both")

    # 右:1/g vs 1/n_h 线性检验(判据版)
    lin = result["fits"]["linearity"]
    for g_est, color, mk, lb in ((g_e2, "C1", "s", "E2"), (g_e3, "C2", "^", "E3")):
        m = g_est > 0
        axR.plot(1 / nh[m], 1 / g_est[m], mk, color=color, ms=6, label=lb)
        p = lin[lb]["params_inv"]
        xs = np.linspace((1 / nh[m]).min(), (1 / nh[m]).max(), 50)
        axR.plot(xs, np.polyval(p, xs), "-", color=color, lw=1.1, alpha=0.7,
                 label=fr"{lb} linear fit ($R^2$={lin[lb]['r2_inv_linear']:.4f})")
    xs = np.linspace((1 / nh).min(), (1 / nh[g_th > 0]).max(), 50)
    axR.plot(xs, 4 * (A2R + B_COEF * xs) / A1**2, "--", color="gray", lw=1.4,
             label="theory law (exactly linear)")
    axR.set_xlabel(r"$1/n_h$"); axR.set_ylabel(r"$1/g(n_h)$")
    axR.set_title(r"linearity check: $1/g$ vs $1/n_h$ "
                  r"($\Leftrightarrow$ hyperbola)")
    axR.legend(fontsize=7); axR.grid(alpha=0.3)

    g = result["gates"]
    fig.suptitle(
        f"N1c dosage (rematch, n_h≤32000) — matched spectra, white aligned δ=0.3 "
        f"(c={C_ALIGN:.3f}), A1={A1:.2e}<0 ✓ | E2: hyper-best="
        f"{'✓' if g['E2']['hyperbola_best'] else '✗'} half-sat {g['E2']['half_sat_fit']:.0f}"
        f"/{NH_HALF_THEORY:.0f} {'✓' if g['E2']['half_sat_within_2x'] else '✗'} "
        f"1/g-lin R²={g['E2']['r2_inv_linear']:.4f} "
        f"{'✓' if g['E2']['r2_inv_linear_ok'] else '✗'} → {g['E2']['verdict']} | "
        f"E3: report only", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(PNG_PATH, dpi=140)
    print(f"saved {PNG_PATH}", flush=True)


def summarize(result):
    g = result["gates"]
    print("\n=== N1c gates (rematch) ===")
    print(f"A1 = {A1:+.4e} (<0 前提): {g['A1_negative']['verdict']}")
    print(f"theory: A2r={A2R:.4e} B={B_COEF:.4e} n_h^(1/2)={NH_HALF_THEORY:.1f} "
          f"g_inf={result['theory_coeffs']['g_inf_theory']:.4e}")
    j = g["E2"]
    print(f"[E2] R²: hyper={j['r2']['hyperbola']:.5f} exp={j['r2']['exp']:.5f} "
          f"log={j['r2']['log']:.5f} | hyper_best={j['hyperbola_best']} | "
          f"half-sat {j['half_sat_fit']:.1f} vs {j['half_sat_theory']:.1f} "
          f"(ratio {j['half_sat_ratio']:.2f}, ok={j['half_sat_within_2x']}) | "
          f"1/g-lin R²={j['r2_inv_linear']:.5f} (≥{j['r2_inv_linear_threshold']}, "
          f"ok={j['r2_inv_linear_ok']}) → {j['verdict']}")
    j = g["E3"]
    print(f"[E3] (report only) R²: hyper={j['r2']['hyperbola']:.5f} "
          f"exp={j['r2']['exp']:.5f} log={j['r2']['log']:.5f} | "
          f"half-sat={j['half_sat_fit']:.1f} | 1/g-lin R²={j['r2_inv_linear']:.5f}")
    print(f"monotonicity: {g['monotonicity']}")
    print("\nper n_h: theory g | E2 g | E3 g | theory λ* | E2 λ* | E3 λ*")
    for c in result["per_nh"]:
        print(f"  n_h={c['n_h']:>5}: {c['theory']['g']:.5f} | {c['E2']['g']:.5f} | "
              f"{c['E3']['g']:.5f} | {c['theory']['lambda_star']:.3f} | "
              f"{c['E2']['lambda_star_9pt']:.2f} | {c['E3']['lambda_star_9pt']:.2f}")


if __name__ == "__main__":
    if PLOT_ONLY:
        with open(JSON_PATH) as f:
            result = json.load(f)
    else:
        raw = load_cache() if FROM_CACHE else compute_raw()
        result = assemble(raw)
    plot(result)
    summarize(result)
