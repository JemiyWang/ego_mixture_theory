"""N1d H 判据边界(粗批;主判据 = 有限 n_r 判据,2026-10-25 跑前修订;
低频列加密重测,跑前修订之二)。

配置(实施卡 §4 N1d):谱系 A(2.0/0.9);‖δ‖∈{0.1,0.25,0.5,0.75,1.0,1.25,
1.5,1.75,2.0} 九档 × 位置{低频/白色/高频} × 反对齐 = 27 格;n_r=1000/n_h=6000;
μ=0.05(E3 显式 l2=0.05);p=1000;E3 为主(s0=23,9 λ 点),E2 为对照(200 reps)。
跑前修订之二:低频列 ‖δ‖ 在 [0.25,0.75] 加密为 {0.25,0.30,0.35,0.40,0.45,0.5,
0.6,0.75}(档距 0.05–0.15 ≤ 20% 容差 ≈0.078;首版 9 档下 benefit 截零 + 插值只能
报档位端点,容差小于档距,结构性不可达);E2/E3 只补跑新档 {0.30,0.35,0.40,0.45,0.6}。

度量(N1b 修正口径):每格 benefit=R(0)−min_λ R(λ) 的符号,估计器均值/单 seed
曲线上取 min(E2 官方用 9 点子网格;另跑 41 点密网格辅助边界定位,卡允许)。
理论侧(主文 H 判据,paper_cvpr/sec/X_suppl.tex 口径;
K = 2Σ_jσ̃_r,j²(S1j+S2j),S1j=a_j²/D0j²,S2j=a_j²(b_j−a_j)/D0j³,
σ̃_r,j²=σ_r²+F_r,j,F_r,j=Σ_{k≠j}a_kβ*_k²):
  ① 有限 n_r 判据(主):有益 ⟺ K/n_r > H(⟺ R̄'(0⁺)=H−K/n_r<0);
  ② 渐近 H(仅报告)与其过零根 ‖δ‖*_H=C∞/a1;
  交叉验证:v5 公式密网格 benefit 符号(卡注明与有限判据等价)。
  理论边界闭式:有限 n_r 根 ‖δ‖*=(C∞+K/n_r)/a1,H 根=C∞/a1(a1=A∞ 对 ‖δ‖ 的斜率)。
边界实测:benefit 随 ‖δ‖ 的 +→− 变号处,相邻异号档线性插值;若上端档 benefit
恒等 0(截零),插值退化报上端档(口径注记)。
辅助定位量:每格 v5 公式 R̄'(0⁺)(密网格首步数值微分)符号 + E2 实测 λ=0.05 处
边际 benefit R(0)−R(0.05)(200 reps 配对差,均值±se)——与边界判定交叉验证。

判据(预注册):实测符号 vs 有限 n_r 判据一致率 ≥90%(E3、E2 分别,27 格);
实测 ‖δ‖* 边界 vs 有限 n_r 理论边界 ≤20%(加密低频列:E2/E3 分别);
渐近 H 根仅报告。旧的 27 格一致率结果在 JSON 保留不动(gates.agreement_*
与 boundaries 段),加密重测结果写入 gates.boundary_lowfreq_dense 与
lowfreq_dense 段。

用法:python3 run_n1d.py               # 增量计算(只补缓存缺的格)+ 落盘 + 绘图
     python3 run_n1d.py --from-cache   # 全部用缓存重组 JSON/图
     python3 run_n1d.py --plot-only    # 只从已存 JSON 重画图
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
from estimators import e2_full_ridge, e2_grams, e3_sgd_linear
from theory import mse_v5, risk_from_mse, feedthrough

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
JSON_PATH = OUT / "n1d_h_boundary.json"
PNG_PATH = OUT / "n1d_h_boundary.png"
CACHE_PATH = OUT / "n1d_raw_cache.npz"
PLOT_ONLY = "--plot-only" in sys.argv
FROM_CACHE = "--from-cache" in sys.argv

DELTA_NORMS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
SHAPES = ["lowfreq", "white", "highfreq"]
CELLS = [(dn, sh) for dn in DELTA_NORMS for sh in SHAPES]
# 跑前修订之二:低频列 [0.25,0.75] 加密;0.25/0.5/0.75 复用原批缓存
FINE_LOWFREQ = [0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.6, 0.75]
NEW_LOWFREQ = [0.30, 0.35, 0.40, 0.45, 0.6]
ALL_CELLS = CELLS + [(dn, "lowfreq") for dn in NEW_LOWFREQ]
LAMS = np.array(C.LAMBDAS)
DENSE41 = np.linspace(0.0, 1.0, 41)          # 步长 0.025 ⊇ 9 点子网格
IDX9 = [int(round(l / 0.025)) for l in LAMS]  # 9 点网格在 41 点网格中的下标
ALPHA_R, ALPHA_H = C.SPECTRA["A"]
REPS = C.REPS_CLOSED
AGREE_MIN = 0.90
BOUND_TOL = 0.20

BETA_STAR = C.make_beta_star()
LAM_R, LAM_H = C.make_spectra(ALPHA_R, ALPHA_H)


def sgn(benefit):
    return 1 if benefit > 0 else -1


# ---------------- 理论侧(全部闭式) ----------------
def theory_consts():
    a, b, mu = LAM_R, LAM_H, C.MU
    D0 = a + mu
    sig_t = C.SIG_R2 + feedthrough(a, BETA_STAR)
    K = 2.0 * np.sum(sig_t * (a**2 / D0**2 + a**2 * (b - a) / D0**3))
    Cinf = float((a * (2 * mu**2 * BETA_STAR**2 * (b - a)) / D0**3).sum())
    return K, Cinf


K_COEF, C_INF = theory_consts()


def a1_slope(shape):
    """有符号 A∞ 对 ‖δ‖ 的斜率:a1 = -2μ Σ a β* b δ̂/D0²,δ̂ 单位方向"""
    d_hat = C.make_delta(shape, 1.0, BETA_STAR, anti_align=True)
    a, b = LAM_R, LAM_H
    return float((a * (-2 * C.MU * BETA_STAR * b * d_hat) / (a + C.MU)**2).sum())


def theory_cell(delta_norm, shape):
    a1 = a1_slope(shape)
    H = a1 * delta_norm - C_INF
    S = K_COEF / C.N_R - H                     # 有限 n_r:有益 ⟺ S>0
    delta = C.make_delta(shape, delta_norm, BETA_STAR, anti_align=True)
    args = (LAM_R, LAM_H, delta, BETA_STAR, BETA_STAR + delta,
            C.N_R, C.N_H, C.SIG_R2, C.SIG_H2, C.MU)
    dense = C.grid_dense()
    curve = np.array([risk_from_mse(mse_v5(l, *args), LAM_R) for l in dense])
    ben_v5 = float(curve[0] - curve.min())
    rprime0 = float((curve[1] - curve[0]) / (dense[1] - dense[0]))  # v5 R̄'(0⁺) 数值微分
    i05 = int(round(0.05 / (dense[1] - dense[0])))
    mb05_v5 = float(curve[0] - curve[i05])     # v5 口径的 λ=0.05 边际 benefit
    return dict(H=H, A_inf=a1 * delta_norm, C_inf=C_INF, K=K_COEF,
                K_over_nr_minus_H=S,
                finite_sign=sgn(S),            # +1 预测有益
                v5_benefit=ben_v5, v5_sign=sgn(ben_v5),
                v5_lambda_star=float(dense[int(np.argmin(curve))]),
                v5_rprime0=rprime0, v5_rprime0_beneficial=bool(rprime0 < 0),
                v5_mb05=mb05_v5,
                c_delta_beta=float(delta @ BETA_STAR))


def theory_boundaries():
    out = {}
    for sh in SHAPES:
        a1 = a1_slope(sh)
        out[sh] = dict(a1=a1, root_H=C_INF / a1,
                       root_finite=(C_INF + K_COEF / C.N_R) / a1)
    return out


def theory_benefit_curve(shape, norms):
    """细 ‖δ‖ 网格上的 v5 benefit 曲线(画图用)"""
    out = []
    for dn in norms:
        delta = C.make_delta(shape, float(dn), BETA_STAR, anti_align=True)
        args = (LAM_R, LAM_H, delta, BETA_STAR, BETA_STAR + delta,
                C.N_R, C.N_H, C.SIG_R2, C.SIG_H2, C.MU)
        dense = C.grid_dense()
        curve = np.array([risk_from_mse(mse_v5(l, *args), LAM_R) for l in dense])
        out.append(float(curve[0] - curve.min()))
    return out


# ---------------- workers ----------------
def _e2_cell(cell):
    dn, sh = cell
    delta = C.make_delta(sh, dn, BETA_STAR, anti_align=True)
    rng = np.random.default_rng(C.SEED)   # 各格共享同一随机流 ⇒ 跨格配对
    risks = np.zeros((REPS, len(DENSE41)))
    for rep in range(REPS):
        rob, hum = gen_two_domain(rng, C.N_R, C.N_H, LAM_R, LAM_H,
                                  BETA_STAR, delta, C.SIG_R2, C.SIG_H2)
        grams = e2_grams(rob, hum)
        for i, lam in enumerate(DENSE41):
            b2 = e2_full_ridge(rob, hum, lam, C.MU, grams=grams)
            risks[rep, i] = risk_from_mse((b2 - BETA_STAR) ** 2, LAM_R)
    return dn, sh, risks


def _e3_task(args):
    dn, sh, lam = args
    import torch
    torch.set_num_threads(4)
    delta = C.make_delta(sh, dn, BETA_STAR, anti_align=True)
    rng = np.random.default_rng(C.S0)
    rob, hum = gen_two_domain(rng, C.N_R, C.N_H, LAM_R, LAM_H,
                              BETA_STAR, delta, C.SIG_R2, C.SIG_H2)
    b = e3_sgd_linear(rob, hum, lam, l2=C.L2_N1A,
                      steps=C.E3_STEPS, lr0=C.E3_LR, seed=C.S0)
    return dn, sh, lam, risk_from_mse((b - BETA_STAR) ** 2, LAM_R)


def compute_raw():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.environ.update(OPENBLAS_NUM_THREADS="8", OMP_NUM_THREADS="8",
                      MKL_NUM_THREADS="8")
    mp = __import__("multiprocessing").get_context("spawn")
    raw = load_cache() if CACHE_PATH.exists() else {"e2": {}, "e3": {}}
    raw.setdefault("e2", {}); raw.setdefault("e3", {})
    miss2 = [c for c in ALL_CELLS if c not in raw["e2"]]
    miss3 = [c for c in ALL_CELLS if c not in raw["e3"]]
    print(f"[cache] reuse {len(raw['e2'])}/{len(ALL_CELLS)} e2, "
          f"{len(raw['e3'])}/{len(ALL_CELLS)} e3; missing e2={miss2} e3={miss3}",
          flush=True)
    if miss2:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=9, mp_context=mp) as ex:
            futs = {ex.submit(_e2_cell, cell): cell for cell in miss2}
            for fut in as_completed(futs):
                dn, sh, risks = fut.result()
                raw["e2"][(dn, sh)] = risks
                print(f"[E2] ‖δ‖={dn} {sh} done ({time.time()-t0:.0f}s)", flush=True)
    if miss3:
        for cell in miss3:
            raw["e3"][cell] = np.zeros(len(LAMS))
        tasks = [(dn, sh, float(l)) for dn, sh in miss3 for l in LAMS]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=14, mp_context=mp) as ex:
            futs = {ex.submit(_e3_task, t): t for t in tasks}
            for fut in as_completed(futs):
                dn, sh, lam_r, risk = fut.result()
                raw["e3"][(dn, sh)][int(np.argmin(np.abs(LAMS - lam_r)))] = risk
                print(f"[E3] ‖δ‖={dn} {sh} λ={lam_r:<4.2f} ({time.time()-t0:.0f}s)",
                      flush=True)
    np.savez(CACHE_PATH,
             **{f"e2::{dn}::{sh}": raw["e2"][c] for c in raw["e2"]
                for dn, sh in [c]},
             **{f"e3::{dn}::{sh}": raw["e3"][c] for c in raw["e3"]
                for dn, sh in [c]})
    print(f"saved {CACHE_PATH}", flush=True)
    return raw


def load_cache():
    z = np.load(CACHE_PATH)
    raw = {"e2": {}, "e3": {}}
    for key in z.files:
        k, dn, sh = key.split("::")
        raw[k][(float(dn), sh)] = z[key]
    return raw


def cell_record(dn, sh, raw):
    th = theory_cell(dn, sh)
    risks = raw["e2"][(dn, sh)]
    mean9 = risks[:, IDX9].mean(0)
    se9 = risks[:, IDX9].std(0, ddof=1) / np.sqrt(REPS)
    mean41 = risks.mean(0)
    ben_e2 = float(mean9[0] - mean9.min())                 # 官方 9 点口径
    ben_e2_dense = float(mean41[0] - mean41.min())         # 密网格辅助
    mb_rep = risks[:, IDX9[0]] - risks[:, IDX9[1]]         # λ=0.05 边际 benefit(配对)
    r3 = raw["e3"][(dn, sh)]
    ben_e3 = float(r3[0] - r3.min())
    return dict(
        delta_norm=dn, shape=sh, theory=th,
        E2=dict(risk_mean=list(map(float, mean9)), risk_se=list(map(float, se9)),
                benefit=ben_e2, sign=sgn(ben_e2),
                benefit_dense41=ben_e2_dense, sign_dense41=sgn(ben_e2_dense),
                marginal_benefit_005=dict(
                    mean=float(mb_rep.mean()),
                    se=float(mb_rep.std(ddof=1) / np.sqrt(REPS)))),
        E3=dict(risk=list(map(float, r3)), benefit=ben_e3, sign=sgn(ben_e3)),
    )


def measured_boundary(bens, norms):
    """+→− 变号位置(benefit 线性插值;上端档恒等 0 时退化为上端档——口径注记)。"""
    for i in range(len(norms) - 1):
        if bens[i] > 0 and bens[i + 1] <= 0:
            return float(norms[i] + (0.0 - bens[i]) * (norms[i + 1] - norms[i])
                         / (bens[i + 1] - bens[i]))
    return None


def assemble(raw):
    tb = theory_boundaries()
    cells = []
    agree = {"E2": 0, "E3": 0}
    cross = 0          # 有限判据 vs v5 benefit 符号交叉验证
    for dn, sh in CELLS:
        c = cell_record(dn, sh, raw)
        th = c["theory"]
        agree["E2"] += int(c["E2"]["sign"] == th["finite_sign"])
        agree["E3"] += int(c["E3"]["sign"] == th["finite_sign"])
        cross += int(th["v5_sign"] == th["finite_sign"])
        cells.append(c)

    # ---- 低频列加密段(跑前修订之二) ----
    dense_rows = [cell_record(dn, "lowfreq", raw) for dn in FINE_LOWFREQ]
    dense_rows.sort(key=lambda c: c["delta_norm"])
    b_e2 = [c["E2"]["benefit_dense41"] for c in dense_rows]
    b_e2_9 = [c["E2"]["benefit"] for c in dense_rows]
    b_e3 = [c["E3"]["benefit"] for c in dense_rows]
    root_fin = tb["lowfreq"]["root_finite"]

    def boundary_verdict(m):
        if m is None:
            return dict(measured=None, verdict="FAIL(加密列内无变号)")
        rel = abs(m - root_fin) / root_fin
        return dict(measured=m, rel_err=rel,
                    verdict="PASS" if rel <= BOUND_TOL else "FAIL")

    boundary_lowfreq_dense = dict(
        theory_root_finite=root_fin, theory_root_H=tb["lowfreq"]["root_H"],
        norms=FINE_LOWFREQ,
        convention="相邻异号档 benefit 线性插值;上端档恒等 0(截零)时报上端档",
        E2_dense41=boundary_verdict(measured_boundary(b_e2, FINE_LOWFREQ)),
        E2_9pt=boundary_verdict(measured_boundary(b_e2_9, FINE_LOWFREQ)),
        E3=boundary_verdict(measured_boundary(b_e3, FINE_LOWFREQ)),
    )
    # 交叉验证:v5 R̄'(0⁺) 符号 vs 有限判据;E2 λ=0.05 边际 benefit 与 v5 同量对照
    # (mb.05 与 R̄'(0⁺) 在 λ*<0.05 的格上符号本就不同——v5 曲线在近 0 处弯曲,
    #  故 E2 实测必须与 v5 的 mb.05 比,而不是与 R̄'(0⁺) 比)
    xval = dict(
        v5_rprime0_vs_finite=int(np.sum(
            (np.array([r["theory"]["v5_rprime0"] < 0 for r in dense_rows])
             == np.array([r["theory"]["finite_sign"] > 0 for r in dense_rows])))),
        e2_marginal005_vs_finite=int(np.sum(
            (np.array([r["E2"]["marginal_benefit_005"]["mean"] > 0
                       for r in dense_rows])
             == np.array([r["theory"]["finite_sign"] > 0 for r in dense_rows])))),
        e2_mb05_vs_v5_mb05_sign_match=int(np.sum(
            np.array([r["E2"]["marginal_benefit_005"]["mean"] > 0
                      for r in dense_rows])
            == np.array([r["theory"]["v5_mb05"] > 0 for r in dense_rows]))),
        e2_mb05_max_zscore_vs_v5=float(max(
            abs(r["E2"]["marginal_benefit_005"]["mean"] - r["theory"]["v5_mb05"])
            / max(r["E2"]["marginal_benefit_005"]["se"], 1e-300)
            for r in dense_rows)),
        n=len(dense_rows),
        note="E2 的 mb.05 必须与 v5 的 mb.05 同量对照(近 0 弯曲使 mb.05 与 "
             "R̄'(0⁺) 在 λ*<0.05 的格符号不同);z-score 用 E2 配对 se",
    )

    # ---- 原 9 档边界(首版口径,保留) ----
    boundaries = {}
    for sh in SHAPES:
        row = sorted([c for c in cells if c["shape"] == sh],
                     key=lambda c: c["delta_norm"])
        norms = [c["delta_norm"] for c in row]
        b2 = [c["E2"]["benefit_dense41"] for c in row]
        b3 = [c["E3"]["benefit"] for c in row]
        rt = tb[sh]
        in_range = DELTA_NORMS[0] <= rt["root_finite"] <= DELTA_NORMS[-1]
        m_e2, m_e3 = measured_boundary(b2, norms), measured_boundary(b3, norms)

        def verdict(m):
            if not in_range:
                return "N/A(理论根在量程外)" + ("一致(量程内无变号)" if m is None
                                               else f"不一致(量程内变号于 {m:.3f})")
            if m is None:
                return "FAIL(理论根在量程内但实测无变号)"
            rel = abs(m - rt["root_finite"]) / rt["root_finite"]
            return f"{'PASS' if rel <= BOUND_TOL else 'FAIL'}(实测 {m:.3f} vs 理论 {rt['root_finite']:.3f},rel {rel:.1%})"
        boundaries[sh] = dict(
            theory_root_finite=rt["root_finite"], theory_root_H=rt["root_H"],
            in_range=bool(in_range),
            measured_E2_dense41=m_e2, measured_E3=m_e3,
            verdict_E2=verdict(m_e2), verdict_E3=verdict(m_e3),
            note="首版 9 档口径(保留);加密重测见 boundary_lowfreq_dense")

    def agree_rec(n):
        rate = n / len(CELLS)
        return dict(match=n, of=len(CELLS), rate=rate, threshold=AGREE_MIN,
                    verdict="PASS" if rate >= AGREE_MIN else "FAIL")

    result = dict(
        experiment="N1d H 判据边界(谱系 A,反对齐,27 格 + 低频列加密 8 档;"
                   "主判据=有限 n_r 判据)",
        config=dict(p=C.P, s=C.S_SMOOTH, sig_r2=C.SIG_R2, sig_h2=C.SIG_H2, mu=C.MU,
                    l2=C.L2_N1A, n_r=C.N_R, n_h=C.N_H, seed=C.SEED, reps=REPS, s0=C.S0,
                    spectrum_family="A", alpha_r=ALPHA_R, alpha_h=ALPHA_H,
                    delta_norms=DELTA_NORMS, shapes=SHAPES, anti_aligned=True,
                    fine_lowfreq_norms=FINE_LOWFREQ, lambdas=list(map(float, LAMS)),
                    e2_dense41_grid=list(map(float, DENSE41)),
                    benefit_semantics="N1b 修正口径:均值/单 seed 曲线上取 min_λ;"
                                      "E2 官方 9 点,41 点密网格辅助边界定位",
                    theory_semantics="有限 n_r 判据:有益 ⟺ K/n_r>H(K=2σ̃_r²(S1+S2),"
                                     "主文 Theorem 2,σ̃²=σ²+F_r);H 渐近仅报告"),
        theory_boundaries=tb,
        K=K_COEF, C_inf=C_INF, K_over_nr=K_COEF / C.N_R,
        cells=cells,
        lowfreq_dense=dense_rows,
        boundaries=boundaries,
        gates=dict(
            agreement_E2=agree_rec(agree["E2"]),
            agreement_E3=agree_rec(agree["E3"]),
            cross_check_finite_vs_v5=dict(match=cross, of=len(CELLS),
                                          note="卡注明两者等价;仅交叉验证,不判定"),
            boundaries_9pt_legacy=boundaries,
            boundary_lowfreq_dense=boundary_lowfreq_dense,
            crossval_lowfreq_dense=xval,
        ),
    )
    with open(JSON_PATH, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"saved {JSON_PATH}", flush=True)
    return result


def plot(result):
    fine_norms = np.linspace(0.08, 2.05, 60)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharey=False)
    dense_cells = {c["delta_norm"]: c for c in result.get("lowfreq_dense", [])}
    for k, sh in enumerate(SHAPES):
        ax = axes[k]
        th_curve = theory_benefit_curve(sh, fine_norms)
        ax.plot(fine_norms, th_curve, "-", color="C0", lw=1.6,
                label="theory v5 benefit (dense λ)")
        row = sorted([c for c in result["cells"] if c["shape"] == sh],
                     key=lambda c: c["delta_norm"])
        norms = [c["delta_norm"] for c in row]
        ax.plot(norms, [c["E2"]["benefit"] for c in row], "s", color="C1",
                ms=6, label="E2 benefit (200 reps, 9-pt)")
        ax.plot(norms, [c["E2"]["benefit_dense41"] for c in row], "x",
                color="C1", ms=6, alpha=0.6, label="E2 benefit (dense-41λ)")
        ax.plot(norms, [c["E3"]["benefit"] for c in row], "^", color="C2",
                ms=6, label="E3 benefit (s0)")
        ax.axhline(0, color="k", lw=0.8)
        bd = result["boundaries"][sh]
        x_hi = 2.1
        for root, ls, name in ((bd["theory_root_finite"], "--", "finite-n_r root"),
                               (bd["theory_root_H"], ":", "H root")):
            if root <= x_hi:
                ax.axvline(root, color="gray", ls=ls, lw=1.2,
                           label=f"{name}={root:.3f}")
            else:
                ax.annotate(f"{name}={root:.3g} (off-range)", xy=(0.03, 0.03),
                            xycoords="axes fraction", fontsize=6.5, color="gray")
        # 低频面板:加密列覆盖 + 加密实测根
        title_extra = ""
        if sh == "lowfreq" and dense_cells:
            fn = sorted(dense_cells)
            ax.plot(fn, [dense_cells[v]["E2"]["benefit_dense41"] for v in fn],
                    "D", color="C1", ms=6, mfc="none", mew=1.6,
                    label="E2 dense ‖δ‖ column (dense-41λ)")
            ax.plot(fn, [dense_cells[v]["E3"]["benefit"] for v in fn],
                    "P", color="C2", ms=6, mfc="none", mew=1.6,
                    label="E3 dense ‖δ‖ column")
            bld = result["gates"]["boundary_lowfreq_dense"]
            for est, color in (("E2_dense41", "C1"), ("E3", "C2")):
                m = bld[est]["measured"]
                if m is not None:
                    ax.axvline(m, color=color, ls="-.", lw=1.4,
                               label=f"{est} root ≈{m:.3f} ({bld[est]['verdict']})")
            title_extra = (f" | dense: E2 {bld['E2_dense41']['measured']} "
                           f"{bld['E2_dense41']['verdict']}, "
                           f"E3 {bld['E3']['measured']} {bld['E3']['verdict']}")

        def short(est):
            m = bd[f"measured_{est}_dense41" if est == "E2" else f"measured_{est}"]
            rt = bd["theory_root_finite"]
            if not bd["in_range"]:
                return (f"{est}: N/A root-out-range, "
                        + ("no flip ok" if m is None else f"flip@{m:.2f}"))
            if m is None:
                return f"{est}: FAIL no-flip"
            rel = abs(m - rt) / rt
            return f"{est}: {'PASS' if rel <= BOUND_TOL else 'FAIL'} {m:.3f}vs{rt:.3f}({rel:.0%})"

        ax.set_title(f"{sh} | 9-lvl: {short('E2')} | {short('E3')}{title_extra}",
                     fontsize=8)
        ax.set_xlim(0.0, 2.1)
        ax.set_xlabel(r"$\|\delta\|$")
        if k == 0:
            ax.set_ylabel(r"benefit $=R(0)-\min_\lambda R(\lambda)$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="best")
    g = result["gates"]
    fig.suptitle(
        f"N1d H-criterion boundary — family A, anti-aligned | agreement vs finite-n_r: "
        f"E2 {g['agreement_E2']['match']}/27 ({g['agreement_E2']['verdict']}), "
        f"E3 {g['agreement_E3']['match']}/27 ({g['agreement_E3']['verdict']}); "
        f"cross-check finite=v5 {g['cross_check_finite_vs_v5']['match']}/27 | "
        f"lowfreq dense root: E2 {g['boundary_lowfreq_dense']['E2_dense41']['verdict']}, "
        f"E3 {g['boundary_lowfreq_dense']['E3']['verdict']}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PNG_PATH, dpi=140)
    print(f"saved {PNG_PATH}", flush=True)


def summarize(result):
    g = result["gates"]
    print("\n=== N1d gates ===")
    for k in ("agreement_E2", "agreement_E3"):
        print(f"[{k}] {g[k]['match']}/27 = {g[k]['rate']:.1%} (≥90%) → {g[k]['verdict']}")
    print(f"[cross-check] 有限判据 == v5 benefit 符号: "
          f"{g['cross_check_finite_vs_v5']['match']}/27")
    bld = g["boundary_lowfreq_dense"]
    print(f"[boundary lowfreq dense] theory finite root={bld['theory_root_finite']:.3f} "
          f"(H root={bld['theory_root_H']:.3f})")
    for est in ("E2_dense41", "E2_9pt", "E3"):
        v = bld[est]
        print(f"  {est}: measured={v['measured']} → {v['verdict']}")
    x = g["crossval_lowfreq_dense"]
    print(f"[crossval dense] v5 R̄'(0⁺) sign == finite: {x['v5_rprime0_vs_finite']}/{x['n']}; "
          f"E2 mb.05 sign == finite: {x['e2_marginal005_vs_finite']}/{x['n']}; "
          f"E2 mb.05 sign == v5 mb.05: {x['e2_mb05_vs_v5_mb05_sign_match']}/{x['n']}, "
          f"max |z|={x['e2_mb05_max_zscore_vs_v5']:.1f}σ")
    print("\nlowfreq dense column:")
    for c in result["lowfreq_dense"]:
        mb = c["E2"]["marginal_benefit_005"]
        print(f"  ‖δ‖={c['delta_norm']:<5} finite={c['theory']['finite_sign']:+d} "
              f"v5R'(0+)={c['theory']['v5_rprime0']:+.1e} v5mb05={c['theory']['v5_mb05']:+.1e} | "
              f"E2 ben9={c['E2']['benefit']:+.1e} ben41={c['E2']['benefit_dense41']:+.1e} "
              f"mb.05={mb['mean']:+.1e}±{mb['se']:.1e} | E3 ben={c['E3']['benefit']:+.1e}")
    for sh in SHAPES:
        bd = result["boundaries"][sh]
        print(f"[boundary {sh} legacy 9-lvl] finite root={bd['theory_root_finite']:.3f} "
              f"| E2: {bd['verdict_E2']} | E3: {bd['verdict_E3']}")


if __name__ == "__main__":
    if PLOT_ONLY:
        with open(JSON_PATH) as f:
            result = json.load(f)
    else:
        raw = load_cache() if FROM_CACHE else compute_raw()
        result = assemble(raw)
    plot(result)
    summarize(result)
