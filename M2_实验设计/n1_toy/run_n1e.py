"""N1e weight decay ↔ μ 映射(粗批,仅 E3)。

配置(实施卡 §4 N1e):wd 八档 {0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2} × 两配置——
A(谱系 A 2.0/0.9,低频 δ=0.5 反对齐,同 N1a)、B(匹配谱 2.0/2.0,白色对齐 δ=0.3,
同 N1c);仅 E3(torch Linear(1000,1,no bias),s0=23,全批,20k 步,Adam 系)。
与 N1a–N1d 的关键差别:wd 走**优化器层**,显式 L2=0;优化器用 **AdamW**
(decoupled weight decay);可选对照:Adam(L2-耦合)在 wd∈{1e-3,1e-2} 两档
(只报不判定)。

方法:每个 (配置 × wd) 训 9 个 λ 点的风险曲线;以同一 s0 数据上的 E2 闭式曲线族
(μ 对数网格 1e-5..1 共 81 点 + 黄金分割细化)为参照,对 log μ 做最小二乘反解
μ_eff(最小化 Σ_λ(r3−r2)²),报告 μ_eff 与相对残差 RMS(‖r3−r2‖/‖r3‖)。
判据(预注册):wd→μ_eff 单调、反解残差 ≤10%、产出 wd→μ 换算曲线与 log-log 斜率。
注:卡写"单调递减",物理上方差衰减方向应为 wd 越大有效正则越强(μ_eff 越大,
单调**递增**);实测方向如实报告,判定按|单调性|给,方向分歧在 JSON 注记。
W6 口径:μ_eff 只是风险曲线反解得到的标量 ridge 系数,不代表 AdamW 的实际
正则化强度;未做学习率、训练步数、参数尺度、多种子敏感性时不得推广为
"AdamW 基本无正则化"或"实际流水线应取 μ→0"。

用法:python3 run_n1e.py [--from-cache | --plot-only]
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
from scipy.optimize import minimize_scalar

import common as C
from gen_data import gen_two_domain
from estimators import e2_grams, e3_sgd_wd
from theory import risk_from_mse

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
JSON_PATH = OUT / "n1e_wd_mu.json"
PNG_PATH = OUT / "n1e_wd_mu.png"
CACHE_PATH = OUT / "n1e_raw_cache.npz"
PLOT_ONLY = "--plot-only" in sys.argv
FROM_CACHE = "--from-cache" in sys.argv

WD_GRID = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
MU_GRID = np.logspace(-8.0, 0.0, 121)   # 下界需低于未正则匹配区(config B wd=0 的匹配 μ<1e-5)
LAMS = np.array(C.LAMBDAS)
RESID_TOL = 0.10
CONFIGS = {
    "A": dict(alphas=C.SPECTRA["A"], shape="lowfreq", norm=0.5, anti=True),
    "B": dict(alphas=(2.0, 2.0), shape="white", norm=0.3, anti=False),
}
ADAM_COMPARE_WD = [1e-3, 1e-2]     # 可选 Adam(L2 耦合)对照,只报不判定

BETA_STAR = C.make_beta_star()


def cfg_setup(cfg):
    cc = CONFIGS[cfg]
    lr, lh = C.make_spectra(*cc["alphas"])
    delta = C.make_delta(cc["shape"], cc["norm"], BETA_STAR, anti_align=cc["anti"])
    return lr, lh, delta


# ---------------- workers ----------------
def _e3_task(args):
    cfg, wd, lam, kind = args
    import torch
    torch.set_num_threads(4)
    lr, lh, delta = cfg_setup(cfg)
    rng = np.random.default_rng(C.S0)
    rob, hum = gen_two_domain(rng, C.N_R, C.N_H, lr, lh,
                              BETA_STAR, delta, C.SIG_R2, C.SIG_H2)
    b = e3_sgd_wd(rob, hum, lam, wd=wd, steps=C.E3_STEPS, lr0=C.E3_LR,
                  seed=C.S0, kind=kind)
    return cfg, wd, lam, kind, risk_from_mse((b - BETA_STAR) ** 2, lr)


def compute_raw():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.environ.update(OPENBLAS_NUM_THREADS="8", OMP_NUM_THREADS="8",
                      MKL_NUM_THREADS="8")
    mp = __import__("multiprocessing").get_context("spawn")
    raw = {}
    if CACHE_PATH.exists():
        z = np.load(CACHE_PATH)
        raw = {k: float(z[k]) for k in z.files}
    tasks = []
    for cfg in CONFIGS:
        for wd in WD_GRID:
            for lam in LAMS:
                if f"{cfg}::adamw::{wd}::{lam}" not in raw:
                    tasks.append((cfg, float(wd), float(lam), "adamw"))
        for wd in ADAM_COMPARE_WD:      # 可选 Adam(L2 耦合)对照
            for lam in LAMS:
                if f"{cfg}::adam::{wd}::{lam}" not in raw:
                    tasks.append((cfg, float(wd), float(lam), "adam"))
    print(f"[cache] reuse {len(raw)} points; missing {len(tasks)} trainings",
          flush=True)
    if tasks:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=14, mp_context=mp) as ex:
            futs = {ex.submit(_e3_task, t): t for t in tasks}
            for fut in as_completed(futs):
                cfg, wd, lam, kind, risk = fut.result()
                raw[f"{cfg}::{kind}::{wd}::{lam}"] = risk
                print(f"[{kind}] {cfg} wd={wd} λ={lam:<4.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        np.savez(CACHE_PATH, **raw)
        print(f"saved {CACHE_PATH}", flush=True)
    return raw


def load_cache():
    z = np.load(CACHE_PATH)
    return {k: float(z[k]) for k in z.files}


# ---------------- E2 参照族与反解 ----------------
def e2_risk_family(cfg):
    """同一 s0 数据上,每 λ 预组合 Gram,返回 risk_E2(λ, μ) 的向量化求值器"""
    lr, lh, delta = cfg_setup(cfg)
    rng = np.random.default_rng(C.S0)
    rob, hum = gen_two_domain(rng, C.N_R, C.N_H, lr, lh,
                              BETA_STAR, delta, C.SIG_R2, C.SIG_H2)
    G_r, G_h, b_r, b_h = e2_grams(rob, hum)
    GA, gb = [], []
    for lam in LAMS:
        GA.append((1 - lam) * G_r + lam * G_h)
        gb.append((1 - lam) * b_r + lam * b_h)
    p = G_r.shape[0]
    eye = np.eye(p)

    def risk_at(i_lam, mu):
        return risk_from_mse(
            (np.linalg.solve(GA[i_lam] + mu * eye, gb[i_lam]) - BETA_STAR) ** 2, lr)

    return risk_at


def invert_mu(risk_at, r3):
    """log μ 网格最小二乘 + 黄金分割细化;返回 (mu_eff, resid_rel, per-lam residual)"""
    R2 = np.array([[risk_at(i, mu) for mu in MU_GRID] for i in range(len(LAMS))])
    r3 = np.asarray(r3)
    obj = (R2 - r3[:, None]) ** 2
    tot = obj.sum(0)
    i0 = int(np.argmin(tot))
    lo, hi = np.log10(MU_GRID[max(i0 - 1, 0)]), np.log10(MU_GRID[min(i0 + 1, len(MU_GRID) - 1)])
    if hi > lo:
        res = minimize_scalar(
            lambda x: sum((risk_at(i, 10 ** x) - r3[i]) ** 2 for i in range(len(LAMS))),
            bounds=(lo, hi), method="bounded",
            options=dict(xatol=1e-6))
        mu_eff = 10 ** res.x
        resid2 = res.fun
    else:
        mu_eff, resid2 = MU_GRID[i0], tot[i0]
    resid_rel = float(np.sqrt(resid2 / np.sum(r3 ** 2)))
    per_lam = [float(r3[i] - risk_at(i, mu_eff)) for i in range(len(LAMS))]
    resid_excl0 = float(np.sqrt(sum(per_lam[1:][i] ** 2 for i in range(len(per_lam) - 1))
                                / np.sum(r3[1:] ** 2)))
    return float(mu_eff), resid_rel, per_lam, resid_excl0


def assemble(raw):
    fams = {cfg: e2_risk_family(cfg) for cfg in CONFIGS}
    out_cfgs = {}
    for cfg in CONFIGS:
        risk_at = fams[cfg]
        rows = []
        for wd in WD_GRID:
            r3 = [raw[f"{cfg}::adamw::{wd}::{l}"] for l in LAMS]
            mu_eff, resid_rel, per_lam, resid_excl0 = invert_mu(risk_at, r3)
            rows.append(dict(wd=wd, risk=list(map(float, r3)), mu_eff=mu_eff,
                             resid_rel=resid_rel, resid_rel_excl_lam0=resid_excl0,
                             resid_per_lam=per_lam,
                             mu_eff_at_grid_floor=bool(mu_eff <= MU_GRID[0] * 1.01)))
        # 可选 Adam(L2 耦合)对照
        adam_rows = []
        for wd in ADAM_COMPARE_WD:
            r3 = [raw[f"{cfg}::adam::{wd}::{l}"] for l in LAMS]
            mu_eff, resid_rel, _, resid_excl0 = invert_mu(risk_at, r3)
            adam_rows.append(dict(wd=wd, mu_eff=mu_eff, resid_rel=resid_rel,
                                  resid_rel_excl_lam0=resid_excl0))
        out_cfgs[cfg] = dict(adamw=rows, adam_coupled_optional=adam_rows)

    result = dict(
        experiment="N1e weight decay ↔ μ 映射(E3,优化器层 wd)",
        config=dict(p=C.P, s=C.S_SMOOTH, mu=C.MU, n_r=C.N_R, n_h=C.N_H, s0=C.S0,
                    wd_grid=WD_GRID, lambdas=list(map(float, LAMS)),
                    mu_grid=[float(MU_GRID[0]), float(MU_GRID[-1]), len(MU_GRID)],
                    optimizer="AdamW(decoupled weight decay;显式 L2=0; Adam 耦合对照可选)",
                    configs={k: dict(alphas=v["alphas"], shape=v["shape"],
                                     norm=v["norm"], anti_aligned=v["anti"])
                             for k, v in CONFIGS.items()},
                    inversion="同一 s0 数据的 E2 闭式曲线族,log μ 网格+黄金分割细化,"
                              "目标 Σ_λ(r3−r2)²;残差=‖r3−r2‖/‖r3‖"),
        results=out_cfgs,
    )
    # ---- 判定 ----
    gates = {}
    for cfg in CONFIGS:
        rows = out_cfgs[cfg]["adamw"]
        mus = np.array([r["mu_eff"] for r in rows])
        resids = np.array([r["resid_rel"] for r in rows])
        mono_inc = bool(np.all(np.diff(mus) >= 0))
        mono_dec = bool(np.all(np.diff(mus) <= 0))
        nz = np.array([r["wd"] for r in rows[1:]])
        log_mu = np.log10(np.maximum(mus[1:], MU_GRID[0]))
        slope, intercept = np.polyfit(np.log10(nz), log_mu, 1)
        gates[cfg] = dict(
            monotone_increasing=mono_inc, monotone_decreasing=mono_dec,
            monotone=mono_inc or mono_dec,
            direction_note="卡写「单调递减」;物理上 wd 越大有效正则越强,"
                           "实测方向如实报告,判定按|单调|给",
            resid_rel_per_wd=list(map(float, resids)),
            resid_rel_max=float(resids.max()),
            resid_rel_max_nonzero_wd=float(resids[1:].max()),
            resid_rel_excl_lam0_per_wd=[r["resid_rel_excl_lam0"] for r in rows],
            resid_rel_excl_lam0_max=float(max(r["resid_rel_excl_lam0"] for r in rows)),
            resid_tol=RESID_TOL,
            loglog_slope=float(slope), loglog_intercept=float(intercept),
            verdict=("PASS" if ((mono_inc or mono_dec)
                                and float(resids[1:].max()) <= RESID_TOL)
                     else "FAIL"),
            verdict_note="判定=|单调|(方向见 direction_note) ∧ 非零 wd 档残差 ≤10%;"
                         "wd=0 行预期残差大(E3 零初始化+有限步的隐式早停正则不在 "
                         "E2 μ 族内,λ=0 处 p=n 无正则 OLS 病态),单列不入判定",
        )
    result["gates"] = gates
    with open(JSON_PATH, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"saved {JSON_PATH}", flush=True)
    return result


def plot(result):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    cfg = "A"
    cc = CONFIGS[cfg]
    fam = e2_risk_family(cfg)
    show_wd = [0.0, 1e-4, 1e-3, 1e-2]
    rows = {r["wd"]: r for r in result["results"][cfg]["adamw"]}
    for i, wd in enumerate(show_wd):
        r = rows[wd]
        color = f"C{i}"
        axL.plot(LAMS, r["risk"], "o", color=color, ms=5,
                 label=f"E3 wd={wd:g} (μ_eff={r['mu_eff']:.2e})")
        e2_curve = [fam(i_l, r["mu_eff"]) for i_l in range(len(LAMS))]
        axL.plot(LAMS, e2_curve, "-", color=color, lw=1.2, alpha=0.8,
                 label=f"E2 μ={r['mu_eff']:.2e} (fit)")
    axL.set_xlabel(r"$\lambda$"); axL.set_ylabel(r"$\Sigma_r$-weighted risk")
    axL.set_title(f"config A (family A, lowfreq δ=0.5 anti): E3(wd) vs matched E2(μ_eff)")
    axL.legend(fontsize=7.5); axL.grid(alpha=0.3)

    for cfg, mk in (("A", "o"), ("B", "s")):
        rows = result["results"][cfg]["adamw"]
        wd_nz = np.array([r["wd"] for r in rows[1:]])
        mu_nz = np.array([r["mu_eff"] for r in rows[1:]])
        axR.plot(wd_nz, mu_nz, mk + "-", ms=6, lw=1.3, label=f"config {cfg}")
        g = result["gates"][cfg]
        xs = np.geomspace(wd_nz.min(), wd_nz.max(), 50)
        axR.plot(xs, 10 ** (g["loglog_intercept"] + g["loglog_slope"] * np.log10(xs)),
                 "--", lw=1.0, alpha=0.7,
                 label=f"{cfg} fit slope={g['loglog_slope']:.2f}")
        axR.annotate(f"{cfg}: μ_eff(wd=0)={rows[0]['mu_eff']:.2e}",
                     xy=(0.03, 0.97 - 0.07 * (cfg == "B")), xycoords="axes fraction",
                     fontsize=7.5, color="dimgray")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel("weight decay (optimizer-layer, AdamW)")
    axR.set_ylabel(r"inverted $\mu_{\mathrm{eff}}$")
    axR.set_title("wd → μ_eff map (log-log)")
    axR.legend(fontsize=8); axR.grid(alpha=0.3, which="both")
    fig.suptitle("N1e scalar ridge inversion (two linear settings only; not an AdamW regularization verdict)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PNG_PATH, dpi=140)
    print(f"saved {PNG_PATH}", flush=True)


def summarize(result):
    print("\n=== N1e wd→μ_eff map ===")
    for cfg in CONFIGS:
        g = result["gates"][cfg]
        print(f"\n[config {cfg}] monotone: inc={g['monotone_increasing']} "
              f"dec={g['monotone_decreasing']} | resid max(all/nonzero-wd) = "
              f"{g['resid_rel_max']:.3%}/{g['resid_rel_max_nonzero_wd']:.3%} "
              f"(tol {g['resid_tol']:.0%}) | log-log slope={g['loglog_slope']:.3f}")
        print(f"  {'wd':>8} {'μ_eff':>10} {'resid%':>8} {'resid_excl_λ0%':>13}")
        for r in result["results"][cfg]["adamw"]:
            print(f"  {r['wd']:>8.0e} {r['mu_eff']:>10.3e} {r['resid_rel']:>8.2%} "
                  f"{r['resid_rel_excl_lam0']:>13.2%}")
        for r in result["results"][cfg]["adam_coupled_optional"]:
            print(f"  adam-coupled wd={r['wd']:.0e}: μ_eff={r['mu_eff']:.3e} "
                  f"resid={r['resid_rel']:.2%}")


if __name__ == "__main__":
    if PLOT_ONLY:
        with open(JSON_PATH) as f:
            result = json.load(f)
    else:
        raw = load_cache() if FROM_CACHE else compute_raw()
        result = assemble(raw)
    plot(result)
    summarize(result)
