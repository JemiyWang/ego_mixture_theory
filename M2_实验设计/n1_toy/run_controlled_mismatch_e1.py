"""W7 second-round independent E1 validation for automatically selected directions."""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import common as C
import run_controlled_mismatch as first_round
from estimators import e1_diag_ridge
from gen_data import gen_two_domain


OUT = Path(__file__).resolve().parent / "outputs"
SOURCE_JSON = OUT / "controlled_mismatch.json"
JSON_PATH = OUT / "controlled_mismatch_e1.json"
PNG_PATH = OUT / "controlled_mismatch_e1.png"

SEED_BASE = 841733
EXCLUDED_OLD_SEED_BASES = (23, 20261025, 20261026)
REPS = 400
N_BOOTSTRAP = 1000
LAMBDAS = np.array([0.0, 0.005, 0.01, 0.02, 0.04, 0.08])
CI_LEVEL = 0.95
P, N_R, N_H, MU = C.P, C.N_R, C.N_H, C.MU
BETA = C.make_beta_star(P, C.S_SMOOTH)
LAM_R, LAM_H = C.make_spectra(*C.SPECTRA["A"], P)
ZERO_DELTA = np.zeros(P)
LAM_COLUMN = LAMBDAS[:, None]
ONE_MINUS_LAM_COLUMN = 1.0 - LAM_COLUMN


def e1_risk_curve(s_r, t_r, s_h, t_h, n_r, n_h, beta, lam_r):
    numerator = ONE_MINUS_LAM_COLUMN * (t_r / n_r) + LAM_COLUMN * (t_h / n_h)
    denominator = ONE_MINUS_LAM_COLUMN * (s_r / n_r) + LAM_COLUMN * (s_h / n_h) + MU
    error = numerator / denominator - beta
    return np.sum(lam_r * error**2, axis=1)


def sufficient_stats(x_r, y_r, x_h, y_h):
    return (
        np.einsum("ij,ij->j", x_r, x_r),
        x_r.T @ y_r,
        np.einsum("ij,ij->j", x_h, x_h),
        x_h.T @ y_h,
    )


def validate_against_reference():
    rng = np.random.default_rng(572914)
    n_r, n_h, p = 8, 11, 13
    x_r = rng.normal(size=(n_r, p))
    y_r = rng.normal(size=n_r)
    x_h = rng.normal(size=(n_h, p))
    y_h = rng.normal(size=n_h)
    beta = rng.normal(size=p)
    lam_r = np.geomspace(1.0, 0.01, p)
    stats = sufficient_stats(x_r, y_r, x_h, y_h)
    actual = e1_risk_curve(*stats, n_r, n_h, beta, lam_r)
    expected = []
    for lam in LAMBDAS:
        beta_hat = e1_diag_ridge((x_r, y_r), (x_h, y_h), float(lam), MU)
        expected.append(float(np.sum(lam_r * (beta_hat - beta)**2)))
    max_error = float(np.max(np.abs(actual - np.asarray(expected))))
    if max_error > 1e-12:
        raise RuntimeError(f"E1 moment implementation differs from reference: {max_error}")
    return {"reference": "estimators.e1_diag_ridge", "max_abs_risk_error": max_error,
            "tolerance": 1e-12, "passed": True, "seed": 572914}


def compute_rep(rep, deltas):
    rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, 0, rep]))
    (x_r, y_r), (x_h, y_h0) = gen_two_domain(
        rng, N_R, N_H, LAM_R, LAM_H, BETA, ZERO_DELTA, C.SIG_R2, C.SIG_H2)
    s_r, t_r, s_h, t_h0 = sufficient_stats(x_r, y_r, x_h, y_h0)
    t_h_delta = x_h.T @ (x_h @ deltas.T)
    risks = np.empty((len(deltas), len(LAMBDAS)))
    for index in range(len(deltas)):
        risks[index] = e1_risk_curve(
            s_r, t_r, s_h, t_h0 + t_h_delta[:, index],
            N_R, N_H, BETA, LAM_R)
    return rep, risks


def quadratic_coefficients(risks):
    scale = float(LAMBDAS[-1])
    scaled_lambdas = LAMBDAS / scale
    powers = np.arange(3)
    design = scaled_lambdas[:, None] ** powers
    responses = risks.reshape(-1, len(LAMBDAS)).T
    scaled = np.linalg.lstsq(design, responses, rcond=None)[0]
    coefficients = scaled / (scale ** powers)[:, None]
    return coefficients.reshape(3, risks.shape[0], -1).transpose(1, 2, 0)


def sign(value):
    if value > 1e-12:
        return 1
    if value < -1e-12:
        return -1
    return 0


def bootstrap_derivatives(derivatives):
    rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, 1]))
    n_directions, n_reps = derivatives.shape
    samples = np.empty((n_directions, N_BOOTSTRAP))
    for index in range(N_BOOTSTRAP):
        rep_indices = rng.integers(0, n_reps, size=n_reps)
        samples[:, index] = derivatives[:, rep_indices].mean(axis=1)
    return samples


def analyze(risks, representatives):
    coefficients = quadratic_coefficients(risks)
    derivatives = coefficients[:, :, 1]
    bootstrap = bootstrap_derivatives(derivatives)
    rows = []
    for index, representative in enumerate(representatives):
        values = derivatives[index]
        mean = float(values.mean())
        lower, upper = np.percentile(bootstrap[index], [2.5, 97.5])
        h = float(representative["H"])
        h_sign = sign(h)
        measured_sign = sign(mean)
        rows.append({
            "direction_key": representative["direction_key"],
            "role": representative["role"],
            "E": representative["E"],
            "c_delta_beta_star": representative["c_delta_beta_star"],
            "A_inf": representative["A_inf"],
            "C_inf": representative["C_inf"],
            "H": h,
            "K": representative["K"],
            "ncrit": representative["ncrit"],
            "H_sign": h_sign,
            "finite_nr_H_minus_K_over_nr": h - representative["K"] / N_R,
            "quadratic_derivative_mean": mean,
            "quadratic_derivative_se": float(values.std(ddof=1) / np.sqrt(REPS)),
            "measured_sign": measured_sign,
            "mean_sign_consistent_with_controlled_H": bool(measured_sign == h_sign),
            "bootstrap": {
                "method": "paired rep-index percentile bootstrap; the same resampled indices are used for all three directions",
                "reps": N_BOOTSTRAP,
                "confidence_level": CI_LEVEL,
                "percentiles": [2.5, 97.5],
                "lower": float(lower), "upper": float(upper),
                "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                "entire_ci_consistent_with_controlled_H_sign": bool(
                    (h_sign > 0 and lower > 0.0) or (h_sign < 0 and upper < 0.0)),
            },
            "risk_mean": risks[index].mean(axis=0),
            "risk_se": risks[index].std(axis=0, ddof=1) / np.sqrt(REPS),
        })
    return {
        "per_direction": rows,
        "all_mean_signs_consistent_with_controlled_H": bool(all(
            row["mean_sign_consistent_with_controlled_H"] for row in rows)),
        "all_bootstrap_cis_consistent_with_controlled_H_sign": bool(all(
            row["bootstrap"]["entire_ci_consistent_with_controlled_H_sign"] for row in rows)),
        "raw_arrays": {
            "risk_reps": risks,
            "quadratic_coefficients_reps": coefficients,
            "quadratic_derivative_reps": derivatives,
            "bootstrap_derivative_samples": bootstrap,
            "dimensions": {
                "risk_reps": ["direction_min_intermediate_max", "fresh_rep", "lambda"],
                "quadratic_coefficients_reps": ["direction", "fresh_rep", "c0_c1_c2"],
                "quadratic_derivative_reps": ["direction", "fresh_rep"],
                "bootstrap_derivative_samples": ["direction", "bootstrap_rep"],
            },
        },
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(result):
    safe = json_safe(result)
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(safe, handle, ensure_ascii=False, indent=1, allow_nan=False)
        handle.write("\n")
    print(f"saved {JSON_PATH}", flush=True)
    return safe


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["analysis"]["per_direction"]
    fig, (ax_derivative, ax_risk) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(rows))
    means = np.array([row["quadratic_derivative_mean"] for row in rows])
    lower = np.array([row["bootstrap"]["lower"] for row in rows])
    upper = np.array([row["bootstrap"]["upper"] for row in rows])
    ax_derivative.errorbar(x, means, yerr=np.vstack([means - lower, upper - means]),
                           fmt="o", capsize=5, color="C0", label="mean with paired bootstrap 95% CI")
    for index, row in enumerate(rows):
        ax_derivative.scatter(index, row["H"], marker="x", s=70, color="C3",
                              label="controlled H" if index == 0 else None)
    ax_derivative.axhline(0.0, color="black", lw=0.8)
    ax_derivative.set_xticks(x, [row["role"] for row in rows])
    ax_derivative.set_ylabel(r"estimated $R'(0^+)$ and controlled $H$")
    ax_derivative.set_title("Per-rep six-point quadratic first coefficient")
    ax_derivative.grid(alpha=0.25)
    ax_derivative.legend(fontsize=8)

    for index, row in enumerate(rows):
        ax_risk.errorbar(LAMBDAS, row["risk_mean"], yerr=row["risk_se"], fmt="o-",
                         capsize=3, lw=1.1, ms=4, label=f"{row['role']}: {row['direction_key']}")
    ax_risk.set_xlabel(r"$\lambda$")
    ax_risk.set_ylabel("E1 Sigma_r-weighted excess risk")
    ax_risk.set_title("Fresh-rep mean risk curves (mean +/- SE)")
    ax_risk.grid(alpha=0.25)
    ax_risk.legend(fontsize=8)
    fig.suptitle(
        f"W7 independent E1 | fresh reps={REPS}, seed={SEED_BASE}, workers={result['actual_workers']}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PNG_PATH, dpi=160)
    plt.close(fig)
    print(f"saved {PNG_PATH}", flush=True)


def load_representatives():
    with open(SOURCE_JSON, encoding="utf-8") as handle:
        source = json.load(handle)
    if source.get("run_state") != "complete" or not source.get("verdict_is_final", False):
        raise RuntimeError("E1 requires the complete final first-round JSON, not a checkpoint")
    if source.get("config", {}).get("algorithm_version") != first_round.ALGORITHM_VERSION:
        raise RuntimeError("E1 requires the current analytic/constructive first-round algorithm version")
    second_round = source["verdict"]["second_round"]
    if not second_round["required"] or second_round["representative_selection"] is None:
        raise RuntimeError("first round did not trigger E1 with three certified representatives")
    selection = second_round["representative_selection"]
    keys = selection["direction_keys_in_order_min_intermediate_max"]
    roles = ("A_inf_min", "intermediate", "A_inf_max")
    representatives = []
    for key, role, record in zip(keys, roles, selection["directions"]):
        delta = np.asarray(record["delta"], dtype=float)
        z = -delta
        residuals = first_round.constraint_residuals(
            z, selection["E_target"], selection["c_midpoint_abs"])
        if residuals["max_abs_including_nonnegativity"] >= first_round.RESIDUAL_TOL:
            raise RuntimeError(f"selected direction {key} fails independent residual verification")
        a_inf = float(first_round.W @ z)
        if not np.isclose(a_inf, record["A_inf"], rtol=1e-9, atol=1e-12):
            raise RuntimeError(f"selected direction {key} has inconsistent A_inf")
        enriched = dict(record)
        enriched.update({"direction_key": key, "role": role})
        representatives.append(enriched)
    return source, selection, representatives


def main():
    started = time.time()
    resources = first_round.resource_snapshot()
    requested_workers = int(os.environ.get("W7_E1_WORKER_CAP", "24"))
    if not 1 <= requested_workers <= 24:
        raise ValueError("W7_E1_WORKER_CAP must be between 1 and 24")
    workers = min(resources["selected_worker_cap_if_parallel"], requested_workers)
    resources["requested_worker_cap"] = requested_workers
    print(json.dumps({"resource_check": resources["commands"]["uptime"]["stdout"],
                      "actual_workers": workers, "fresh_reps": REPS,
                      "seed_base": SEED_BASE}, ensure_ascii=False), flush=True)
    reference_check = validate_against_reference()
    source, selection, representatives = load_representatives()
    deltas = np.asarray([record["delta"] for record in representatives], dtype=float)
    risks = np.full((len(representatives), REPS, len(LAMBDAS)), np.nan)
    context = mp.get_context("spawn")
    completed = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [executor.submit(compute_rep, rep, deltas) for rep in range(REPS)]
        for future in as_completed(futures):
            rep, rep_risks = future.result()
            risks[:, rep, :] = rep_risks
            completed += 1
            if completed % 50 == 0 or completed == REPS:
                print(f"[controlled E1] fresh reps {completed}/{REPS}; "
                      f"elapsed={time.time() - started:.1f}s", flush=True)
    if not np.isfinite(risks).all():
        raise RuntimeError("non-finite or missing fresh-rep risk")
    analysis = analyze(risks, representatives)
    result = {
        "experiment": "W7 controlled-mismatch independent E1 near-zero derivative validation",
        "preregistration": "W7 §6; triggered only by the completed first-round automatic decision",
        "run_state": "complete",
        "source_json": str(SOURCE_JSON),
        "source_trigger": source["verdict"]["second_round"]["reason"],
        "source_final_category_at_launch": source["verdict"]["category"],
        "representative_selection": selection,
        "actual_workers": workers,
        "resources_before_recompute": resources,
        "config": {
            "estimator": "E1 diagonal ridge",
            "p": P, "s": C.S_SMOOTH, "mu": MU,
            "n_r": N_R, "n_h": N_H,
            "sigma_r2": C.SIG_R2, "sigma_h2": C.SIG_H2,
            "spectrum_family": "A", "alpha_r": C.SPECTRA["A"][0], "alpha_h": C.SPECTRA["A"][1],
            "E_target": selection["E_target"],
            "c_midpoint_abs": selection["c_midpoint_abs"],
            "fresh_reps": REPS,
            "lambdas": LAMBDAS,
            "quadratic_fit": "per fresh rep, unweighted least squares on six lambda points scaled by 0.08; original-scale c1 estimates R'(0+)",
            "bootstrap_reps": N_BOOTSTRAP,
            "ci_level": CI_LEVEL,
            "seed_base": SEED_BASE,
            "excluded_old_seed_bases": EXCLUDED_OLD_SEED_BASES,
            "data_seed_scheme": "default_rng(SeedSequence([841733,0,rep])), rep=0..399",
            "bootstrap_seed_scheme": "default_rng(SeedSequence([841733,1]))",
            "paired_direction_protocol": "one fresh zero-delta data stream per rep; T_h(delta)=T_h(0)+X_h^T(X_h delta) for each selected delta",
            "old_N1d_N1c_cache_or_seed_reused": False,
            "gpu_used": False,
            "nested_blas_threading": False,
            "nr_bracketing_or_training_run": False,
        },
        "reference_implementation_check": reference_check,
        "analysis": analysis,
        "interpretation": {
            "primary_check": "measured quadratic derivative mean sign versus controlled H sign for each selected direction",
            "bootstrap_check": "whether the entire 95% percentile CI lies on the controlled-H side of zero",
            "finite_nr_note": "H is asymptotic; H-K/n_r is reported as a finite-n_r diagnostic and is not substituted for the preregistered H comparison",
            "no_ncrit_measurement": "No n_r sign-change bracketing was requested for a paper quantitative ncrit claim, so none was run",
        },
        "implementation_deviations": [],
        "elapsed_seconds_at_json_write": time.time() - started,
    }
    safe = write_json(result)
    plot(safe)
    print(json.dumps({"category_at_launch": result["source_final_category_at_launch"],
                      "all_mean_signs_consistent_with_H": analysis["all_mean_signs_consistent_with_controlled_H"],
                      "all_CIs_consistent_with_H": analysis["all_bootstrap_cis_consistent_with_controlled_H_sign"],
                      "elapsed_seconds": time.time() - started}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
