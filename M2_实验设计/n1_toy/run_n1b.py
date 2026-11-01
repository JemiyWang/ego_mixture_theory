"""N1b 失配位置效应(零参数测试,粗批)。

配置(实施卡 §4 N1b):谱系 A(α_r=2.0/α_h=0.9);‖δ‖∈{0.5,1.0,2.0}
× 位置{低频 j^{-1} / 白色 j^0 / 高频 (j/P)²} × 反对齐 = 9 格;
p=1000、s=1.0、σ²=1.0/4.0、μ=0.05(E3/E4 显式 l2=0.05)、n_r=1000/n_h=6000;
λ 网格 9 点;E2 用 200 reps,E3/E4 单 seed s0=23。

度量:每格 benefit = R(0) − min_λ R(λ),其中 R(λ) 为该估计器的风险曲线
(E2 取 200 reps 均值曲线;E3/E4 为 s0 单条曲线),min 取 9 点网格。
注意:逐 rep 先取 min 再平均的口径带 Jensen 正偏(噪声曲线的 min 系统性偏低),
仅作次要统计量保留(benefit_perrep_*),符号判定一律用主口径。
理论侧符号由 v5 公式(201 点密网格)benefit 符号给出,并记录 H 泛函值
(H = A∞ − C∞,按本卡归一 β* 与每格实际 δ 计算:
A∞=-2μΣ a_jβ*_j b_jδ_j/D0²,C∞=2μ²Σ a_jβ*_j²(b_j−a_j)/D0³,
D0=a+μ,a=lr、b=lh;本实验 δ_j<0、β*_j>0,与旧的绝对值特化数值相同)。
benefit 符号约定:>0 有益(+1),≤0(含 λ*=0 的恒等 0)有害(−1)。

判据(预注册):9 格"理论符号 vs 实测符号"一致率 100%(允许 ±1 格时记 90%);
钉死格:‖δ‖=2 低频格必须有害(λ*=0)、‖δ‖=2 高频格必须有益。
按估计器分别判定:E2(200 reps 均值)、E3(s0)、E4(s0)。

复现约定:每格的 E2 reps 与 E3/E4 训练数据都用 default_rng(23) 独立重建
(各格共享同一随机流 ⇒ 跨格配对,仅 δ 不同);E3/E4 训练在 spawn 子进程并行
(worker 内 torch 4 线程,避免 112 核微批抖动);E4 风险为 50k MC 测试风险。
原始数组缓存 outputs/n1b_raw_cache.npz,支持不重算地重组 JSON/图。

用法:python3 run_n1b.py               # 全程计算 + 落盘 + 绘图
     python3 run_n1b.py --from-cache   # 用缓存原始数组重组 JSON/图(不重算)
     python3 run_n1b.py --plot-only    # 只从已存 JSON 重画图
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

import common as C
from gen_data import gen_two_domain
from estimators import (e2_full_ridge, e2_grams,
                        e3_sgd_linear, e4_mlp, mc_risk_model)
from theory import mse_v5, risk_from_mse

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
JSON_PATH = OUT / "n1b_location_grid.json"
PNG_PATH = OUT / "n1b_location_grid.png"
CACHE_PATH = OUT / "n1b_raw_cache.npz"
PLOT_ONLY = "--plot-only" in sys.argv
FROM_CACHE = "--from-cache" in sys.argv

DELTA_NORMS = [0.5, 1.0, 2.0]
SHAPES = ["lowfreq", "white", "highfreq"]
CELLS = [(dn, sh) for dn in DELTA_NORMS for sh in SHAPES]
LAMS = np.array(C.LAMBDAS)
ALPHA_R, ALPHA_H = C.SPECTRA["A"]
REPS = C.REPS_CLOSED


def cell_setup(delta_norm, shape):
    beta_star = C.make_beta_star()
    lr, lh = C.make_spectra(ALPHA_R, ALPHA_H)
    delta = C.make_delta(shape, delta_norm, beta_star, anti_align=True)
    return beta_star, lr, lh, delta, beta_star + delta


def sgn(benefit):
    """benefit 符号:>0 有益(+1),≤0(含 λ*=0 的恒等 0)有害(−1)。"""
    return 1 if benefit > 0 else -1


def H_functional(lr, lh, delta, beta_star, mu):
    """H = A∞ − C∞(n_r→∞ 时 R̄'(0⁺) 的符号预测子;口径见 l1_c1_H_criterion.py)"""
    a, b = lr, lh
    D0 = a + mu
    A_inf = float((a * (-2 * mu * beta_star * b * delta) / D0**2).sum())
    C_inf = float((a * (2 * mu**2 * beta_star**2 * (b - a)) / D0**3).sum())
    return A_inf, C_inf, A_inf - C_inf


# ---------------- 理论侧(廉价,assemble 时重算) ----------------
def theory_cell(delta_norm, shape):
    beta_star, lr, lh, delta, beta_h = cell_setup(delta_norm, shape)
    args = (lr, lh, delta, beta_star, beta_h, C.N_R, C.N_H, C.SIG_R2, C.SIG_H2, C.MU)
    dense = C.grid_dense()
    curve = np.array([risk_from_mse(mse_v5(l, *args), lr) for l in dense])
    benefit = float(curve[0] - curve.min())
    A_inf, C_inf, H = H_functional(lr, lh, delta, beta_star, C.MU)
    return dict(risk_dense=list(map(float, curve)),
                benefit=benefit, sign=sgn(benefit),
                lambda_star=float(dense[int(np.argmin(curve))]),
                H=H, A_inf=A_inf, C_inf=C_inf,
                c_delta_beta=float(delta @ beta_star))


# ---------------- workers ----------------
def _e2_cell(cell):
    dn, sh = cell
    beta_star, lr, lh, delta, _ = cell_setup(dn, sh)
    rng = np.random.default_rng(C.SEED)   # 各格共享同一随机流 ⇒ 跨格配对
    risks = np.zeros((REPS, len(LAMS)))
    for rep in range(REPS):
        rob, hum = gen_two_domain(rng, C.N_R, C.N_H, lr, lh,
                                  beta_star, delta, C.SIG_R2, C.SIG_H2)
        grams = e2_grams(rob, hum)
        for i, lam in enumerate(LAMS):
            b2 = e2_full_ridge(rob, hum, lam, C.MU, grams=grams)
            risks[rep, i] = risk_from_mse((b2 - beta_star) ** 2, lr)
    return dn, sh, risks


def _e3_task(args):
    dn, sh, lam = args
    import torch
    torch.set_num_threads(4)
    beta_star, lr, lh, delta, _ = cell_setup(dn, sh)
    rng = np.random.default_rng(C.S0)
    rob, hum = gen_two_domain(rng, C.N_R, C.N_H, lr, lh,
                              beta_star, delta, C.SIG_R2, C.SIG_H2)
    b = e3_sgd_linear(rob, hum, lam, l2=C.L2_N1A,
                      steps=C.E3_STEPS, lr0=C.E3_LR, seed=C.S0)
    return dn, sh, lam, risk_from_mse((b - beta_star) ** 2, lr)


def _e4_task(args):
    dn, sh, lam = args
    import torch
    torch.set_num_threads(4)
    beta_star, lr, lh, delta, _ = cell_setup(dn, sh)
    rng = np.random.default_rng(C.S0)
    rob, hum = gen_two_domain(rng, C.N_R, C.N_H, lr, lh,
                              beta_star, delta, C.SIG_R2, C.SIG_H2)
    m = e4_mlp(rob, hum, lam, l2=C.L2_N1A, width=C.E4_WIDTH,
               steps=C.E4_STEPS, lr0=C.E4_LR, batch=C.E4_BATCH, seed=C.S0)
    Xte = rng.normal(size=(C.N_TEST_MC, C.P)) * np.sqrt(lr)
    return dn, sh, lam, mc_risk_model(m, Xte, Xte @ beta_star)


def compute_raw():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    # spawn 子进程重建 numpy/torch 运行环境 ⇒ BLAS 线程限额在此生效
    os.environ.update(OPENBLAS_NUM_THREADS="8", OMP_NUM_THREADS="8",
                      MKL_NUM_THREADS="8")
    mp = __import__("multiprocessing").get_context("spawn")

    raw = {"e2": {}, "e3": {}, "e4": {}}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=9, mp_context=mp) as ex:
        futs = {ex.submit(_e2_cell, cell): cell for cell in CELLS}
        for fut in as_completed(futs):
            dn, sh, risks = fut.result()
            raw["e2"][(dn, sh)] = risks
            print(f"[E2] cell ‖δ‖={dn} {sh} done ({time.time()-t0:.0f}s)", flush=True)

    for k in ("e3", "e4"):
        raw[k] = {(dn, sh): np.zeros(len(LAMS)) for dn, sh in CELLS}
    tasks = [(dn, sh, float(l)) for dn, sh in CELLS for l in LAMS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=14, mp_context=mp) as ex:
        futs = {ex.submit(_e3_task, t): ("e3", t) for t in tasks}
        futs.update({ex.submit(_e4_task, t): ("e4", t) for t in tasks})
        for fut in as_completed(futs):
            kind, (dn, sh, lam) = futs[fut]
            _, _, lam_r, risk = fut.result()
            i = int(np.argmin(np.abs(LAMS - lam_r)))
            raw[kind][(dn, sh)][i] = risk
            print(f"[{kind}] ‖δ‖={dn} {sh} λ={lam_r:<4.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    np.savez(CACHE_PATH,
             **{f"{k}::{dn}::{sh}": raw[k][(dn, sh)]
                for k in ("e2", "e3", "e4") for dn, sh in CELLS})
    print(f"saved {CACHE_PATH}", flush=True)
    return raw


def load_cache():
    z = np.load(CACHE_PATH)
    raw = {"e2": {}, "e3": {}, "e4": {}}
    for key in z.files:
        k, dn, sh = key.split("::")
        raw[k][(float(dn), sh)] = z[key]
    return raw


def assemble(raw):
    cells = []
    agree = {"E2": 0, "E3": 0, "E4": 0}
    for dn, sh in CELLS:
        th = theory_cell(dn, sh)
        risks = raw["e2"][(dn, sh)]
        mean, se = risks.mean(0), risks.std(0, ddof=1) / np.sqrt(REPS)
        ben_e2 = float(mean[0] - mean.min())          # 主口径:均值曲线上取 min
        ben_rep = risks[:, 0] - risks.min(axis=1)     # 次要口径(Jensen 正偏,仅参照)
        r3, r4 = raw["e3"][(dn, sh)], raw["e4"][(dn, sh)]
        ben_e3, ben_e4 = float(r3[0] - r3.min()), float(r4[0] - r4.min())
        signs = {"E2": sgn(ben_e2), "E3": sgn(ben_e3), "E4": sgn(ben_e4)}
        for k in agree:
            agree[k] += int(signs[k] == th["sign"])
        cells.append(dict(
            delta_norm=dn, shape=sh, theory=th,
            E2=dict(risk_mean=list(map(float, mean)), risk_se=list(map(float, se)),
                    risk_reps=list(map(float, risks.ravel())),
                    benefit=ben_e2, sign=signs["E2"],
                    benefit_perrep_mean=float(ben_rep.mean()),
                    benefit_perrep_se=float(ben_rep.std(ddof=1) / np.sqrt(REPS)),
                    lambda_star_9pt=float(LAMS[int(mean.argmin())])),
            E3=dict(risk=list(map(float, r3)), benefit=ben_e3, sign=signs["E3"],
                    lambda_star_9pt=float(LAMS[int(r3.argmin())])),
            E4=dict(risk=list(map(float, r4)), benefit=ben_e4, sign=signs["E4"],
                    lambda_star_9pt=float(LAMS[int(r4.argmin())])),
        ))

    def agree_rec(n):
        return dict(match=n, of=len(CELLS), rate=n / len(CELLS),
                    verdict="PASS" if n == 9 else ("PASS(90%,±1 格)" if n == 8 else "FAIL"))

    def pinned(dn, sh, want):
        c = next(c for c in cells if c["delta_norm"] == dn and c["shape"] == sh)
        per = {"theory": int(c["theory"]["sign"]),
               "E2": c["E2"]["sign"], "E3": c["E3"]["sign"], "E4": c["E4"]["sign"]}
        ok = all(s == want for s in per.values())
        if want < 0:   # 有害格另要求 λ*=0
            ok = ok and c["theory"]["lambda_star"] == 0.0 \
                and all(c[k]["lambda_star_9pt"] == 0.0 for k in ("E2", "E3", "E4"))
        return dict(required="harmful(λ*=0)" if want < 0 else "beneficial",
                    signs=per, lambda_stars=dict(theory=c["theory"]["lambda_star"],
                                                 **{k: c[k]["lambda_star_9pt"]
                                                    for k in ("E2", "E3", "E4")}),
                    benefit=dict(theory=c["theory"]["benefit"],
                                 **{k: c[k]["benefit"] for k in ("E2", "E3", "E4")}),
                    H=c["theory"]["H"], verdict="PASS" if ok else "FAIL")

    result = dict(
        experiment="N1b 失配位置效应(零参数测试;谱系 A,反对齐,9 格)",
        config=dict(p=C.P, s=C.S_SMOOTH, sig_r2=C.SIG_R2, sig_h2=C.SIG_H2, mu=C.MU,
                    l2=C.L2_N1A, n_r=C.N_R, n_h=C.N_H, seed=C.SEED, reps=REPS, s0=C.S0,
                    spectrum_family="A", alpha_r=ALPHA_R, alpha_h=ALPHA_H,
                    delta_norms=DELTA_NORMS, shapes=SHAPES, anti_aligned=True,
                    lambdas=list(map(float, LAMS)),
                    benefit_semantics="主口径:估计器风险曲线(E2=200-rep 均值曲线)上 "
                                      "R(0)−min_λ R(λ);benefit_perrep_* 为逐 rep 先取 min "
                                      "再平均的次要口径(Jensen 正偏,仅参照)",
                    note="每格 E2/E3/E4 用同一 default_rng(23) 流重建数据(跨格配对);"
                         "H 泛函按本卡归一 β* 与每格实际 δ 计算(参照文件为白色 δ+未归一 β*)"),
        cells=cells,
        gates=dict(
            agreement_E2=agree_rec(agree["E2"]),
            agreement_E3=agree_rec(agree["E3"]),
            agreement_E4=agree_rec(agree["E4"]),
            pinned_norm2_lowfreq_harmful=pinned(2.0, "lowfreq", -1),
            pinned_norm2_highfreq_beneficial=pinned(2.0, "highfreq", +1),
        ),
    )
    with open(JSON_PATH, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"saved {JSON_PATH}", flush=True)
    return result


def plot(result):
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.5), sharex=True)
    dense = np.linspace(0.0, 1.0, C.N_DENSE)
    for idx, c in enumerate(result["cells"]):
        ax = axes[idx // 3][idx % 3]
        th = c["theory"]
        ax.plot(dense, th["risk_dense"], "-", color="C0", lw=1.6, label="theory v5")
        ax.errorbar(LAMS, c["E2"]["risk_mean"], yerr=c["E2"]["risk_se"],
                    fmt="s", color="C1", ms=4, capsize=2, label="E2 (200 reps)")
        ax.plot(LAMS, c["E3"]["risk"], "^", color="C2", ms=5, label="E3 (s0)")
        ax.plot(LAMS, c["E4"]["risk"], "v", color="C3", ms=5, label="E4 (s0)")
        ax.axvline(th["lambda_star"], color="gray", ls=":", lw=1.0)
        def mk(est):
            ok = c[est]["sign"] == th["sign"]
            return f"{est}{'+' if c[est]['sign'] > 0 else '-'}{'✓' if ok else '✗'}"
        ax.set_title(f"‖δ‖={c['delta_norm']} {c['shape']} | thy"
                     f"{'+' if th['sign'] > 0 else '-'} (H={th['H']:+.1e}) | "
                     f"{mk('E2')} {mk('E3')} {mk('E4')}", fontsize=9)
        ax.grid(alpha=0.3)
        if idx // 3 == 2:
            ax.set_xlabel(r"$\lambda$")
        if idx % 3 == 0:
            ax.set_ylabel(r"$\Sigma_r$-weighted risk")
    g = result["gates"]
    fig.suptitle(
        f"N1b location grid — family A (2.0/0.9), anti-aligned; "
        f"agreement: E2 {g['agreement_E2']['match']}/9, E3 {g['agreement_E3']['match']}/9, "
        f"E4 {g['agreement_E4']['match']}/9 | "
        f"pinned: 2.0-lowfreq {g['pinned_norm2_lowfreq_harmful']['verdict']}, "
        f"2.0-highfreq {g['pinned_norm2_highfreq_beneficial']['verdict']}", fontsize=11)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(PNG_PATH, dpi=130)
    print(f"saved {PNG_PATH}", flush=True)


def summarize(result):
    g = result["gates"]
    print("\n=== N1b gates ===")
    for k in ("agreement_E2", "agreement_E3", "agreement_E4"):
        print(f"[{k}] {g[k]['match']}/9 = {g[k]['rate']:.0%} → {g[k]['verdict']}")
    for k in ("pinned_norm2_lowfreq_harmful", "pinned_norm2_highfreq_beneficial"):
        print(f"[{k}] {g[k]['verdict']} signs={g[k]['signs']} λ*={g[k]['lambda_stars']}")
    print("\ncell details (theory benefit / H | E2 E3 E4 benefit):")
    for c in result["cells"]:
        print(f"  ‖δ‖={c['delta_norm']:<3} {c['shape']:<9} thy {c['theory']['benefit']:+.5f}"
              f" (H={c['theory']['H']:+.2e}) | E2 {c['E2']['benefit']:+.5f}"
              f" (per-rep {c['E2']['benefit_perrep_mean']:+.5f}±{c['E2']['benefit_perrep_se']:.5f})"
              f" | E3 {c['E3']['benefit']:+.5f} | E4 {c['E4']['benefit']:+.5f}")


if __name__ == "__main__":
    if PLOT_ONLY:
        with open(JSON_PATH) as f:
            result = json.load(f)
    else:
        raw = load_cache() if FROM_CACHE else compute_raw()
        result = assemble(raw)
    plot(result)
    summarize(result)
