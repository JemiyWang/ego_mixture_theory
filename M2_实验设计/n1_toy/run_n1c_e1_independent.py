"""N1c-new：E1 对角 ridge 的独立、零参数剂量律验证。

严格实现 W4_独立验证修订执行方案.md §3：匹配谱 2.0/2.0、白色对齐
δ=0.3、20261026 独立种子、11 个 n_h、400 reps、44 点 λ 网格、1000 次
bootstrap。E3 不运行。JSON 先落盘，PNG 其次。

每 rep 只生成一次数据并只计算一次两域对角 Gram/标签投影，全部 λ 共用。
每格 bootstrap 都是先对重抽 reps 的逐 λ 风险求均值，再对均值曲线取 min。
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_distribution

import common as C
from estimators import e1_diag_ridge
from gen_data import gen_two_domain
from theory import feedthrough


OUT = Path(__file__).parent / "outputs"
JSON_PATH = OUT / "n1c_e1_independent.json"
PNG_PATH = OUT / "n1c_e1_independent.png"

SEED_BASE = 20261026
REPS = 400
N_BOOT = 1000
N_HS = np.array([30, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000],
                dtype=int)
LAMBDAS = np.unique(np.concatenate([
    np.linspace(0.0, 1.0, 41), np.array([0.005, 0.01, 0.02])
]))
TRAIN_MASK = N_HS <= 4000
HELD_OUT_MASK = np.isin(N_HS, [8000, 16000, 32000])
DELTA_NORM = 0.3
REP_CHUNK_SIZE = 20
MAX_WORKERS = min(16, os.cpu_count() or 1)

BETA_STAR = C.make_beta_star()
LAM_R, LAM_H = C.make_spectra(2.0, 2.0)
DELTA = C.make_delta("white", DELTA_NORM, BETA_STAR, anti_align=False)
BETA_H = BETA_STAR + DELTA
LAM_COLUMN = LAMBDAS[:, None]
ONE_MINUS_LAM_COLUMN = 1.0 - LAM_COLUMN

assert len(LAMBDAS) == 44 and LAMBDAS[0] == 0.0 and LAMBDAS[-1] == 1.0
assert TRAIN_MASK.sum() == 8 and HELD_OUT_MASK.sum() == 3


def e1_sufficient_stats(rob, hum):
    """返回除以各域样本量后的对角 Gram 与标签投影。"""
    (x_r, y_r), (x_h, y_h) = rob, hum
    n_r, n_h = x_r.shape[0], x_h.shape[0]
    return (
        np.einsum("ij,ij->j", x_r, x_r) / n_r,
        (x_r.T @ y_r) / n_r,
        np.einsum("ij,ij->j", x_h, x_h) / n_h,
        (x_h.T @ y_h) / n_h,
    )


def e1_risk_curve(stats):
    """在预计算充分统计量上一次性求全部 λ 的 E1 机器人域期望风险。"""
    s_r, t_r, s_h, t_h = stats
    numerator = ONE_MINUS_LAM_COLUMN * t_r + LAM_COLUMN * t_h
    denominator = ONE_MINUS_LAM_COLUMN * s_r + LAM_COLUMN * s_h + C.MU
    error = numerator / denominator - BETA_STAR
    return np.sum(LAM_R * error**2, axis=1)


def validate_e1_against_reference():
    """用小矩阵逐 λ 对照现有 E1；此检查不属于实验 reps。"""
    rng = np.random.default_rng(918273)
    n_r, n_h, p = 8, 11, 13
    rob = (rng.normal(size=(n_r, p)), rng.normal(size=n_r))
    hum = (rng.normal(size=(n_h, p)), rng.normal(size=n_h))
    stats = e1_sufficient_stats(rob, hum)
    global BETA_STAR, LAM_R
    beta_saved, weights_saved = BETA_STAR.copy(), LAM_R.copy()
    try:
        BETA_STAR = rng.normal(size=p)
        LAM_R = np.geomspace(1.0, 0.01, p)
        actual = e1_risk_curve(stats)
        expected = []
        for lam in LAMBDAS:
            beta_hat = e1_diag_ridge(rob, hum, float(lam), C.MU)
            expected.append(float(np.sum(LAM_R * (beta_hat - BETA_STAR)**2)))
    finally:
        BETA_STAR = beta_saved
        LAM_R = weights_saved
    max_abs_error = float(np.max(np.abs(actual - np.asarray(expected))))
    if max_abs_error > 1e-12:
        raise RuntimeError(f"E1 vectorization mismatch: {max_abs_error}")
    return {"reference": "estimators.e1_diag_ridge", "max_abs_risk_error": max_abs_error,
            "tolerance": 1e-12, "passed": True}


def calculate_theory():
    a = LAM_R
    d0 = a + C.MU
    f_r = feedthrough(LAM_R, BETA_STAR)
    f_h = feedthrough(LAM_H, BETA_H)
    sigma_tilde_r2 = C.SIG_R2 + f_r
    sigma_tilde_h2 = C.SIG_H2 + f_h
    v_r = sigma_tilde_r2 * a / C.N_R
    u = a * DELTA
    w = C.MU * BETA_STAR
    a1 = float(np.sum(a * (-2.0 * (v_r + u * w)) / d0**2))
    a2r = float(np.sum(a * (v_r + u**2) / d0**2))
    b = float(np.sum(a * (sigma_tilde_h2 * a + 2.0 * u**2) / d0**2))
    return {
        "A1": a1, "A2r": a2r, "B": b,
        "nh_half": b / a2r,
        "g_inf": a1**2 / (4.0 * a2r),
        "a": a, "d0": d0, "v_r": v_r, "u": u, "w": w,
        "sigma_tilde_r2": sigma_tilde_r2,
        "sigma_tilde_h2": sigma_tilde_h2,
    }


THEORY = calculate_theory()


def theory_g0(n_h):
    n_h = np.asarray(n_h, dtype=float)
    return THEORY["A1"]**2 / (4.0 * (THEORY["A2r"] + THEORY["B"] / n_h))


def theory_lambda_star(n_h):
    n_h = np.asarray(n_h, dtype=float)
    return -THEORY["A1"] / (2.0 * (THEORY["A2r"] + THEORY["B"] / n_h))


def validate_theory_identity():
    """由含 v_h 的逐坐标二次风险复核零参数增益恒等式。"""
    max_relative_error = 0.0
    for n_h in N_HS:
        v_h = THEORY["sigma_tilde_h2"] * LAM_H / n_h

        def quadratic_risk(lam):
            numerator = (
                THEORY["v_r"] * (1.0 - lam)**2
                + v_h * lam**2
                + (THEORY["u"] * lam - THEORY["w"])**2
                + 2.0 * THEORY["u"]**2 * lam**2 / n_h
            )
            return float(np.sum(LAM_R * numerator / THEORY["d0"]**2))

        lam_star = float(theory_lambda_star(n_h))
        if not 0.0 < lam_star < 1.0:
            raise RuntimeError(f"theory optimum is not interior at n_h={n_h}")
        numerical = quadratic_risk(0.0) - quadratic_risk(lam_star)
        closed_form = float(theory_g0(n_h))
        relative_error = abs(numerical - closed_form) / closed_form
        max_relative_error = max(max_relative_error, relative_error)
    if max_relative_error > 1e-10:
        raise RuntimeError(f"dosage identity mismatch: {max_relative_error}")
    return {"reference": "per-coordinate quadratic risk with v_r and v_h",
            "max_relative_gain_error": float(max_relative_error),
            "tolerance": 1e-10, "passed": True}


def compute_rep_chunk(task):
    n_h, start, stop, seed = task
    rng = np.random.default_rng(seed)
    risks = np.empty((stop - start, len(LAMBDAS)), dtype=float)
    for local_rep in range(stop - start):
        rob, hum = gen_two_domain(
            rng, C.N_R, n_h, LAM_R, LAM_H, BETA_STAR, DELTA, C.SIG_R2, C.SIG_H2
        )
        stats = e1_sufficient_stats(rob, hum)
        risks[local_rep] = e1_risk_curve(stats)
        del rob, hum, stats
    return n_h, start, risks


def compute_all_risks(data_seeds):
    raw = {int(n_h): np.empty((REPS, len(LAMBDAS)), dtype=float) for n_h in N_HS}
    tasks = []
    for n_h, cell_seed in zip(N_HS, data_seeds):
        starts = list(range(0, REPS, REP_CHUNK_SIZE))
        chunk_seeds = cell_seed.spawn(len(starts))
        for start, chunk_seed in zip(starts, chunk_seeds):
            stop = min(start + REP_CHUNK_SIZE, REPS)
            tasks.append((int(n_h), start, stop, chunk_seed))

    completed_reps = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(compute_rep_chunk, task) for task in tasks]
        for future in as_completed(futures):
            n_h, start, risks = future.result()
            raw[n_h][start:start + risks.shape[0]] = risks
            completed_reps += risks.shape[0]
            print(f"[E1] completed {completed_reps}/{REPS * len(N_HS)} reps; "
                  f"latest n_h={n_h}, rows={start}:{start + risks.shape[0]}", flush=True)
    for n_h, risks in raw.items():
        if risks.shape != (REPS, len(LAMBDAS)) or not np.all(np.isfinite(risks)):
            raise RuntimeError(f"invalid risk matrix at n_h={n_h}")
    return raw


def bootstrap_cell(risks, seed):
    rng = np.random.default_rng(seed)
    mean_risk = risks.mean(axis=0)
    g = float(mean_risk[0] - mean_risk.min())
    g_boot = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        indices = rng.integers(0, REPS, size=REPS)
        resampled_mean_curve = risks[indices].mean(axis=0)
        g_boot[b] = resampled_mean_curve[0] - resampled_mean_curve.min()
    return {
        "mean_risk": mean_risk,
        "risk_se": risks.std(axis=0, ddof=1) / math.sqrt(REPS),
        "g": g,
        "lambda_star": float(LAMBDAS[int(np.argmin(mean_risk))]),
        "g_boot": g_boot,
        "g_se": float(g_boot.std(ddof=1)),
        "g_ci95": np.percentile(g_boot, [2.5, 97.5]),
    }


def hyperbola(x, a, b, c):
    return a / (b + c / x)


def exponential(x, g_inf, tau):
    return g_inf * (1.0 - np.exp(-x / tau))


def logarithmic(x, a, b):
    return a * np.log1p(x / b)


MODELS = {"hyperbola": hyperbola, "exponential": exponential, "logarithmic": logarithmic}
PARAMETER_NAMES = {
    "hyperbola": ["a", "b", "c"],
    "exponential": ["g_inf", "tau"],
    "logarithmic": ["a", "b"],
}


def initial_params(kind, x, y):
    positive_max = max(float(np.max(y)), 1e-8)
    if kind == "hyperbola":
        return [positive_max, 1.0, float(np.median(x))]
    if kind == "exponential":
        return [positive_max, 500.0]
    return [positive_max / 5.0, 100.0]


def fit_shape(kind, x, y, p0=None):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if p0 is None:
        p0 = initial_params(kind, x, y)
    try:
        params, _ = curve_fit(MODELS[kind], x, y, p0=p0, maxfev=60000)
        prediction = MODELS[kind](x, *params)
        if not np.all(np.isfinite(params)) or not np.all(np.isfinite(prediction)):
            raise RuntimeError("non-finite fit")
        ss_res = float(np.sum((y - prediction)**2))
        ss_tot = float(np.sum((y - y.mean())**2))
        train_r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
        return {"success": True, "params": params, "train_r2": float(train_r2),
                "error": None}
    except Exception as exc:
        return {"success": False, "params": None, "train_r2": None,
                "error": f"{type(exc).__name__}: {exc}"}


def weighted_held_out_rmse(prediction, observed, se):
    """w_i=1/SE_i^2 的归一化加权 RMSE。"""
    weights = 1.0 / np.asarray(se, dtype=float)**2
    residual = np.asarray(prediction, dtype=float) - np.asarray(observed, dtype=float)
    return float(np.sqrt(np.sum(weights * residual**2) / np.sum(weights)))


def finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def auxiliary_hyperbola_bootstrap(g_boot, measured_fit):
    result = {
        "role": "auxiliary_only_not_a_gate",
        "fit_n_h": N_HS.tolist(),
        "fit_form": "g=a/(b+c/n_h); half_saturation=c/b; saturation=a/b",
        "bootstrap_replicates_requested": N_BOOT,
    }
    if not measured_fit["success"]:
        result.update({"success": False, "error": measured_fit["error"]})
        return result

    a, b, c = measured_fit["params"]
    point_half = float(c / b)
    point_saturation = float(a / b)
    half_samples = np.full(N_BOOT, np.nan)
    saturation_samples = np.full(N_BOOT, np.nan)
    fit_failures = 0
    first_error = None
    for index in range(N_BOOT):
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                params, _ = curve_fit(
                    hyperbola, N_HS.astype(float), g_boot[index],
                    p0=measured_fit["params"], maxfev=60000
                )
            aa, bb, cc = params
            half_samples[index] = cc / bb
            saturation_samples[index] = aa / bb
        except Exception as exc:
            fit_failures += 1
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"

    valid = (np.isfinite(half_samples) & np.isfinite(saturation_samples)
             & (half_samples > 0.0) & (saturation_samples > 0.0))
    n_valid = int(valid.sum())
    result.update({
        "success": bool(n_valid > 0),
        "measured_half_saturation": finite_or_none(point_half),
        "measured_saturation_gain": finite_or_none(point_saturation),
        "theory_half_saturation": THEORY["nh_half"],
        "theory_saturation_gain": THEORY["g_inf"],
        "bootstrap_fit_failures": fit_failures,
        "bootstrap_nonphysical_or_nonfinite": int(N_BOOT - fit_failures - n_valid),
        "bootstrap_valid_replicates": n_valid,
        "first_error": first_error,
    })
    if n_valid:
        result.update({
            "half_saturation_se": float(half_samples[valid].std(ddof=1)) if n_valid > 1 else None,
            "half_saturation_ci95": np.percentile(half_samples[valid], [2.5, 97.5]).tolist(),
            "saturation_gain_se": float(saturation_samples[valid].std(ddof=1)) if n_valid > 1 else None,
            "saturation_gain_ci95": np.percentile(saturation_samples[valid], [2.5, 97.5]).tolist(),
        })
    return result


def assemble(raw, cells, data_seeds, bootstrap_seeds, checks, elapsed_seconds):
    g = np.array([cells[int(n_h)]["g"] for n_h in N_HS])
    g_se = np.array([cells[int(n_h)]["g_se"] for n_h in N_HS])
    g0 = theory_g0(N_HS)
    standardized_residuals = (g - g0) / g_se
    chi2_value = float(np.sum(standardized_residuals**2))
    df = len(N_HS)
    p_value = float(chi2_distribution.sf(chi2_value, df))
    compatible = bool(p_value >= 0.05)

    x_train, y_train = N_HS[TRAIN_MASK].astype(float), g[TRAIN_MASK]
    fits = {}
    for kind in MODELS:
        fit = fit_shape(kind, x_train, y_train)
        entry = {
            "success": fit["success"],
            "parameter_names": PARAMETER_NAMES[kind],
            "parameters": None if fit["params"] is None else fit["params"].tolist(),
            "train_r2": fit["train_r2"],
            "error": fit["error"],
        }
        if fit["success"]:
            prediction = MODELS[kind](N_HS[HELD_OUT_MASK].astype(float), *fit["params"])
            entry["held_out_predictions"] = prediction.tolist()
            entry["held_out_rmse"] = weighted_held_out_rmse(
                prediction, g[HELD_OUT_MASK], g_se[HELD_OUT_MASK]
            )
        else:
            entry["held_out_predictions"] = None
            entry["held_out_rmse"] = None
        fits[kind] = entry

    rmse = {kind: fits[kind]["held_out_rmse"] for kind in MODELS}
    finite_rmse = all(value is not None and np.isfinite(value) for value in rmse.values())
    hyperbola_dominant = bool(
        finite_rmse
        and rmse["hyperbola"] <= rmse["exponential"]
        and rmse["hyperbola"] <= rmse["logarithmic"]
    )

    g_boot = np.column_stack([cells[int(n_h)]["g_boot"] for n_h in N_HS])
    measured_auxiliary_fit = fit_shape("hyperbola", N_HS.astype(float), g)
    auxiliary = auxiliary_hyperbola_bootstrap(g_boot, measured_auxiliary_fit)
    auxiliary["all_11_cell_fit_parameters"] = (
        None if measured_auxiliary_fit["params"] is None else measured_auxiliary_fit["params"].tolist()
    )
    auxiliary["all_11_cell_fit_r2"] = measured_auxiliary_fit["train_r2"]

    per_n_h = []
    for index, n_h in enumerate(N_HS):
        cell = cells[int(n_h)]
        risks = raw[int(n_h)]
        per_n_h.append({
            "n_h": int(n_h),
            "reps": REPS,
            "risk_reps_shape": [REPS, len(LAMBDAS)],
            "risk_reps_flat": risks.ravel().tolist(),
            "risk_mean": cell["mean_risk"].tolist(),
            "risk_se_across_reps": cell["risk_se"].tolist(),
            "risk_at_zero": float(cell["mean_risk"][0]),
            "min_mean_risk": float(cell["mean_risk"].min()),
            "lambda_star_on_grid": cell["lambda_star"],
            "g": cell["g"],
            "g_bootstrap_se": cell["g_se"],
            "g_bootstrap_ci95": cell["g_ci95"].tolist(),
            "g_bootstrap_replicates": cell["g_boot"].tolist(),
            "theory_g0": float(g0[index]),
            "theory_lambda_star_continuous": float(theory_lambda_star(n_h)),
            "g_minus_theory_g0": float(g[index] - g0[index]),
            "standardized_residual": float(standardized_residuals[index]),
        })

    return {
        "experiment": "N1c-new E1 independent zero-parameter dosage-law validation",
        "preregistration": "W4_独立验证修订执行方案.md §3 (N1c-new)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "estimator": "E1 diagonal ridge",
            "p": C.P, "s": C.S_SMOOTH,
            "alpha_r": 2.0, "alpha_h": 2.0,
            "delta_shape": "white", "delta_alignment": "aligned",
            "delta_norm_requested": DELTA_NORM,
            "delta_norm_actual": float(np.linalg.norm(DELTA)),
            "c_delta_beta": float(DELTA @ BETA_STAR),
            "sigma_r2": C.SIG_R2, "sigma_h2": C.SIG_H2,
            "mu": C.MU, "n_r": C.N_R,
            "n_h": N_HS.tolist(), "reps_per_n_h": REPS,
            "seed_base": SEED_BASE,
            "seed_construction": "SeedSequence(seed_base).spawn(22); data children 0..10; bootstrap children 11..21",
            "data_seed_spawn_keys": {str(int(n_h)): list(seed.spawn_key)
                                     for n_h, seed in zip(N_HS, data_seeds)},
            "bootstrap_seed_spawn_keys": {str(int(n_h)): list(seed.spawn_key)
                                          for n_h, seed in zip(N_HS, bootstrap_seeds)},
            "rep_chunk_size_for_parallelism": REP_CHUNK_SIZE,
            "chunk_seeds": "each n_h data child spawns independent rep-chunk children",
            "max_workers": MAX_WORKERS,
            "lambdas": LAMBDAS.tolist(),
            "lambda_count": len(LAMBDAS),
            "lambda_grid": "union of np.linspace(0,1,41) and {0.005,0.01,0.02}",
            "paired_across_lambda_within_replicate": True,
            "training_samples_shared_across_n_h": False,
            "risk": "sum_j lambda_r,j (beta_hat_j-beta_star_j)^2",
            "bootstrap_replicates": N_BOOT,
            "bootstrap_gain_semantics": "resample reps; mean each risk curve by lambda; then take min over lambda",
            "bootstrap_ci": "percentile 2.5% and 97.5%",
            "shape_fit_train_n_h": N_HS[TRAIN_MASK].tolist(),
            "shape_held_out_n_h": N_HS[HELD_OUT_MASK].tolist(),
            "held_out_rmse_weighting": "w_i=1/g_bootstrap_SE_i^2; sqrt(sum(w_i residual_i^2)/sum(w_i))",
            "E3_run": False,
            "E2_run": False,
            "training_run": False,
        },
        "theory": {
            "parameterization": "zero fitted constants; per-coordinate feedthrough included",
            "feedthrough_definition": "F_e,j=sum_{k != j} lambda_e,k beta_e,k^2; sigma_tilde_e,j^2=sigma_e^2+F_e,j",
            "coefficient_formulas": {
                "v_r": "sigma_tilde_r,j^2*a_j/n_r",
                "v_h": "sigma_tilde_h,j^2*a_j/n_h",
                "u": "a_j*delta_j", "w": "mu*beta_star_j", "D0": "a_j+mu",
                "A1": "sum a*(-2*(v_r+u*w))/D0^2",
                "A2r": "sum a*(v_r+u^2)/D0^2",
                "B": "sum a*(sigma_tilde_h^2*a+2*u^2)/D0^2",
                "g0": "A1^2/(4*(A2r+B/n_h))",
            },
            "A1": THEORY["A1"], "A2r": THEORY["A2r"], "B": THEORY["B"],
            "half_saturation": THEORY["nh_half"],
            "saturation_gain": THEORY["g_inf"],
            "sigma_tilde_r2_min_max": [float(THEORY["sigma_tilde_r2"].min()),
                                        float(THEORY["sigma_tilde_r2"].max())],
            "sigma_tilde_h2_min_max": [float(THEORY["sigma_tilde_h2"].min()),
                                        float(THEORY["sigma_tilde_h2"].max())],
        },
        "implementation_checks": checks,
        "implementation_deviations_from_section_3": [],
        "per_n_h": per_n_h,
        "zero_parameter_goodness_of_fit": {
            "chi2": chi2_value, "df": df, "p_value": p_value,
            "alpha": 0.05,
            "rule": "compatible if and only if p>=0.05; all 11 cells retained",
            "compatible_with_zero_parameter_theory": compatible,
            "verdict": "PASS" if compatible else "FAIL",
            "conclusion": "与零参数理论相容" if compatible else "当前精度下不相容",
        },
        "shape_extrapolation": {
            "fit_scope": "only n_h<=4000",
            "held_out_scope": "only n_h in {8000,16000,32000}",
            "observed_held_out_g": g[HELD_OUT_MASK].tolist(),
            "held_out_g_bootstrap_se": g_se[HELD_OUT_MASK].tolist(),
            "fits": fits,
            "held_out_rmse": rmse,
            "hyperbola_not_greater_than_both_alternatives": hyperbola_dominant,
            "verdict": "PASS" if hyperbola_dominant else "UNRESOLVED",
            "conclusion": "形状外推占优" if hyperbola_dominant else "形状不可分辨",
        },
        "auxiliary_half_saturation_and_saturation_gain": auxiliary,
        "auxiliary_inverse_gain_plot_only": {
            "role": "1/g versus 1/n_h is shown only in the PNG; not a judgment criterion"
        },
        "runtime_seconds_before_json_write": float(elapsed_seconds),
    }


def plot_result(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_cell = result["per_n_h"]
    n_h = np.array([cell["n_h"] for cell in per_cell], dtype=float)
    g = np.array([cell["g"] for cell in per_cell])
    g_se = np.array([cell["g_bootstrap_se"] for cell in per_cell])
    g0 = np.array([cell["theory_g0"] for cell in per_cell])
    dense_n_h = np.geomspace(n_h.min(), n_h.max(), 400)
    dense_g0 = theory_g0(dense_n_h)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    ax = axes[0]
    ax.plot(dense_n_h, dense_g0, color="black", lw=1.8, label="zero-parameter $g_0$")
    ax.errorbar(n_h, g, yerr=1.96 * g_se, fmt="o", color="C0", ms=5,
                capsize=3, label=r"E1 measured $g$ ($\pm1.96$ bootstrap SE)")
    ax.set_xscale("log")
    ax.set_xlabel("$n_h$")
    ax.set_ylabel("$g=R(0)-\\min_\\lambda R(\\lambda)$")
    ax.set_title("Zero-parameter theory vs independent E1")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    train = n_h <= 4000
    held_out = ~train
    ax.scatter(n_h[train], g[train], color="C0", marker="o", label="fit range")
    ax.scatter(n_h[held_out], g[held_out], color="C0", marker="s", s=55,
               label="held-out")
    colors = {"hyperbola": "C1", "exponential": "C2", "logarithmic": "C3"}
    for kind, fit in result["shape_extrapolation"]["fits"].items():
        if fit["success"]:
            curve = MODELS[kind](dense_n_h, *fit["parameters"])
            rmse = fit["held_out_rmse"]
            ax.plot(dense_n_h, curve, color=colors[kind], lw=1.4,
                    label=f"{kind}, held-out RMSE={rmse:.3e}")
    ax.axvline(4000, color="gray", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("$n_h$")
    ax.set_ylabel("$g$")
    ax.set_title(r"Shape fits: train $\leq4000$, test $>4000$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    ax = axes[2]
    positive = g > 0
    ax.plot(1.0 / n_h[positive], 1.0 / g[positive], "o", color="C0", label="E1 measured")
    ax.plot(1.0 / n_h, 1.0 / g0, "-", color="black", label="zero-parameter theory")
    ax.set_xlabel("$1/n_h$")
    ax.set_ylabel("$1/g$")
    ax.set_title("Auxiliary inverse-gain plot only (no gate)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    primary = result["zero_parameter_goodness_of_fit"]
    secondary = result["shape_extrapolation"]
    fig.suptitle(
        f"N1c-new independent E1: $\\chi^2$/df = "
        f"{primary['chi2']:.3f}/{primary['df']}, p={primary['p_value']:.4g} "
        f"({primary['verdict']}); shape={secondary['verdict']}", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PNG_PATH, dpi=160)
    plt.close(fig)


def main():
    started = time.time()
    print("[check] validating vectorized E1 and zero-parameter theory", flush=True)
    checks = {
        "e1_vectorization": validate_e1_against_reference(),
        "theory_gain_identity": validate_theory_identity(),
    }
    root_seed = np.random.SeedSequence(SEED_BASE)
    children = root_seed.spawn(2 * len(N_HS))
    data_seeds = children[:len(N_HS)]
    bootstrap_seeds = children[len(N_HS):]

    print(f"[E1] full run: {len(N_HS)} cells x {REPS} reps; "
          f"{len(LAMBDAS)} lambdas; workers={MAX_WORKERS}", flush=True)
    raw = compute_all_risks(data_seeds)
    print("[bootstrap] resampling reps and computing mean-curve gains", flush=True)
    cells = {}
    for n_h, seed in zip(N_HS, bootstrap_seeds):
        cells[int(n_h)] = bootstrap_cell(raw[int(n_h)], seed)

    print("[analysis] zero-parameter goodness of fit and held-out shape competition", flush=True)
    result = assemble(raw, cells, data_seeds, bootstrap_seeds, checks, time.time() - started)
    OUT.mkdir(exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1, allow_nan=False)
        handle.write("\n")
    print(f"saved {JSON_PATH}", flush=True)

    plot_result(result)
    print(f"saved {PNG_PATH}", flush=True)

    primary = result["zero_parameter_goodness_of_fit"]
    auxiliary = result["auxiliary_half_saturation_and_saturation_gain"]
    print(f"zero-parameter chi2/df={primary['chi2']:.6f}/{primary['df']}, "
          f"p={primary['p_value']:.8g}: {primary['conclusion']}", flush=True)
    if auxiliary.get("success"):
        print(f"half-saturation measured={auxiliary['measured_half_saturation']:.6f}, "
              f"theory={auxiliary['theory_half_saturation']:.6f}", flush=True)
    print("held-out RMSE: " + ", ".join(
        f"{kind}={value:.8g}" for kind, value in result["shape_extrapolation"]["held_out_rmse"].items()
        if value is not None
    ), flush=True)
    print(f"shape conclusion: {result['shape_extrapolation']['conclusion']}", flush=True)


if __name__ == "__main__":
    main()
