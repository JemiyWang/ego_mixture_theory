"""N1d-new：E1 对角 ridge 的独立相变边界验证（预注册 W4 §2）。"""
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np

import common as C
from gen_data import gen_two_domain
from theory import risk_from_mse, feedthrough

OUT = Path(__file__).resolve().parent / "outputs"
JSON_PATH = OUT / "n1d_e1_independent.json"
PNG_PATH = OUT / "n1d_e1_independent.png"

SEED_BASE = 20261025
LAMBDAS = np.array([0.0, 0.005, 0.01, 0.02, 0.04, 0.08])
AUX_LAMBDA = 0.05
REPS_SCHEDULE = (400, 800, 1600)
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
MAX_RELATIVE_HALFWIDTH = 0.20
N_WORKERS = min(8, os.cpu_count() or 1)

P, N_R, N_H, MU = C.P, C.N_R, C.N_H, C.MU
SIG_R2, SIG_H2 = C.SIG_R2, C.SIG_H2
ALPHA_R, ALPHA_H = C.SPECTRA["A"]
BETA_STAR = C.make_beta_star(P, C.S_SMOOTH)
LAM_R, LAM_H = C.make_spectra(ALPHA_R, ALPHA_H, P)
DELTA_UNIT = C.make_delta("lowfreq", 1.0, BETA_STAR, P, anti_align=True)
ZERO_DELTA = np.zeros(P)

D0 = LAM_R + MU
SIGMA_TILDE_R2 = SIG_R2 + feedthrough(LAM_R, BETA_STAR)
K_COEF = 2.0 * np.sum(
    SIGMA_TILDE_R2 * (LAM_R**2 / D0**2 + LAM_R**2 * (LAM_H - LAM_R) / D0**3)
)
C_INF = float(np.sum(2.0 * MU**2 * LAM_R * BETA_STAR**2 * (LAM_H - LAM_R) / D0**3))
A1_SLOPE = float(np.sum(LAM_R * (-2.0 * MU * BETA_STAR * LAM_H * DELTA_UNIT) / D0**2))
THEORY_ROOT = (C_INF + K_COEF / N_R) / A1_SLOPE
DELTA_NORMS = np.array([0.25, 0.30, 0.35, THEORY_ROOT, 0.40, 0.45, 0.50])
N_DELTA = len(DELTA_NORMS)


def e1_risk_from_moments(s_r, t_r, s_h, t_h, lam):
    wr, wh = (1.0 - lam) / N_R, lam / N_H
    numerator = wr * t_r + wh * t_h
    denominator = wr * s_r + wh * s_h + MU
    beta_hat = numerator / denominator
    return risk_from_mse((beta_hat - BETA_STAR) ** 2, LAM_R)


def compute_rep(rep):
    seed_sequence = np.random.SeedSequence([SEED_BASE, 0, rep])
    rng = np.random.default_rng(seed_sequence)
    (x_r, y_r), (x_h, y_h0) = gen_two_domain(
        rng, N_R, N_H, LAM_R, LAM_H, BETA_STAR, ZERO_DELTA, SIG_R2, SIG_H2
    )
    s_r = np.einsum("ij,ij->j", x_r, x_r)
    t_r = x_r.T @ y_r
    s_h = np.einsum("ij,ij->j", x_h, x_h)
    t_h0 = x_h.T @ y_h0
    t_h_unit = x_h.T @ (x_h @ DELTA_UNIT)

    risks = np.empty((N_DELTA, len(LAMBDAS)))
    risk_005 = np.empty(N_DELTA)
    for delta_index, delta_norm in enumerate(DELTA_NORMS):
        t_h = t_h0 + delta_norm * t_h_unit
        for lambda_index, lam in enumerate(LAMBDAS):
            risks[delta_index, lambda_index] = e1_risk_from_moments(
                s_r, t_r, s_h, t_h, lam
            )
        risk_005[delta_index] = e1_risk_from_moments(
            s_r, t_r, s_h, t_h, AUX_LAMBDA
        )
    return rep, risks, risk_005


def compute_batch(start, stop, risks, risk_005):
    t0 = time.time()
    completed = 0
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=context) as executor:
        futures = [executor.submit(compute_rep, rep) for rep in range(start, stop)]
        for future in as_completed(futures):
            rep, rep_risks, rep_risk_005 = future.result()
            risks[:, rep, :] = rep_risks
            risk_005[:, rep] = rep_risk_005
            completed += 1
            if completed % 50 == 0 or completed == stop - start:
                print(
                    f"[E1] reps {completed}/{stop - start} in batch; "
                    f"total target={stop}; elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )


def polynomial_coefficients(risks, degree):
    scale = float(LAMBDAS[-1])
    scaled_lambdas = LAMBDAS / scale
    powers = np.arange(degree + 1)
    design = scaled_lambdas[:, None] ** powers
    responses = risks.reshape(-1, len(LAMBDAS)).T
    scaled_coefficients = np.linalg.lstsq(design, responses, rcond=None)[0]
    coefficients = scaled_coefficients / (scale ** powers)[:, None]
    return coefficients.reshape(degree + 1, N_DELTA, -1).transpose(1, 2, 0)


def unweighted_linear_root(mean_derivatives):
    x = DELTA_NORMS
    y = np.asarray(mean_derivatives)
    x_centered = x - x.mean()
    denominator = float(x_centered @ x_centered)
    slope = float((x_centered * (y - y.mean())).sum() / denominator)
    intercept = float(y.mean() - slope * x.mean())
    root = -intercept / slope if slope != 0.0 and np.isfinite(slope) else float("nan")
    return intercept, slope, root


def root_summary(mean_derivatives, bootstrap_roots):
    intercept, slope, root = unweighted_linear_root(mean_derivatives)
    root_is_finite = bool(np.isfinite(root))
    sign_bracket = bool(
        np.any(mean_derivatives < 0.0) and np.any(mean_derivatives > 0.0)
    )
    root_in_interval = bool(
        root_is_finite and DELTA_NORMS[0] <= root <= DELTA_NORMS[-1]
    )
    finite_bootstrap = np.isfinite(bootstrap_roots)
    n_valid = int(finite_bootstrap.sum())
    lower = upper = halfwidth = relative_halfwidth = float("nan")
    theory_in_ci = False
    if n_valid == N_BOOTSTRAP:
        lower, upper = np.percentile(bootstrap_roots, [2.5, 97.5])
        halfwidth = (upper - lower) / 2.0
        relative_halfwidth = halfwidth / THEORY_ROOT
        theory_in_ci = bool(lower <= THEORY_ROOT <= upper)
    ci_too_wide = bool(relative_halfwidth > MAX_RELATIVE_HALFWIDTH)
    return {
        "unweighted_linear_fit": {
            "intercept": intercept,
            "slope": slope,
            "root": root,
            "weights": [1.0] * N_DELTA,
            "theory_root_candidate_weight": 1.0,
        },
        "root": root,
        "sign_bracket_across_candidates": sign_bracket,
        "root_in_candidate_interval": root_in_interval,
        "candidate_interval": [float(DELTA_NORMS[0]), float(DELTA_NORMS[-1])],
        "bootstrap": {
            "method": "joint paired rep-index percentile bootstrap",
            "reps": N_BOOTSTRAP,
            "valid_root_reps": n_valid,
            "all_root_reps_valid": bool(n_valid == N_BOOTSTRAP),
            "confidence_level": CI_LEVEL,
            "percentiles": [2.5, 97.5],
            "lower": lower,
            "upper": upper,
            "halfwidth": halfwidth,
            "halfwidth_over_theory_root": relative_halfwidth,
            "maximum_relative_halfwidth": MAX_RELATIVE_HALFWIDTH,
            "ci_too_wide": ci_too_wide,
            "theory_root": THEORY_ROOT,
            "theory_root_in_ci": theory_in_ci,
            "root_reps": bootstrap_roots,
        },
    }


def judge_primary(primary):
    reasons = []
    if not primary["sign_bracket_across_candidates"]:
        reasons.append("derivative signs do not bracket zero")
    if not np.isfinite(primary["root"]):
        reasons.append("fitted root is not finite")
    elif not primary["root_in_candidate_interval"]:
        reasons.append("fitted root is outside the candidate interval")
    bootstrap = primary["bootstrap"]
    if not bootstrap["all_root_reps_valid"]:
        reasons.append("not all bootstrap roots are finite")
    elif bootstrap["ci_too_wide"]:
        reasons.append("95% CI relative halfwidth exceeds 20%")
    if reasons:
        verdict = "UNRESOLVED"
    elif not bootstrap["theory_root_in_ci"]:
        verdict = "FAIL"
        reasons.append("theory root is outside the 95% bootstrap CI")
    else:
        verdict = "PASS"
    return verdict, reasons


def bootstrap_roots(derivatives_quadratic, derivatives_cubic):
    rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, 1]))
    roots_quadratic = np.full(N_BOOTSTRAP, np.nan)
    roots_cubic = np.full(N_BOOTSTRAP, np.nan)
    n_reps = derivatives_quadratic.shape[1]
    for bootstrap_index in range(N_BOOTSTRAP):
        rep_indices = rng.integers(0, n_reps, size=n_reps)
        roots_quadratic[bootstrap_index] = unweighted_linear_root(
            derivatives_quadratic[:, rep_indices].mean(axis=1)
        )[2]
        roots_cubic[bootstrap_index] = unweighted_linear_root(
            derivatives_cubic[:, rep_indices].mean(axis=1)
        )[2]
    return roots_quadratic, roots_cubic


def analyze(risks, risk_005, n_reps):
    risks = risks[:, :n_reps, :]
    risk_005 = risk_005[:, :n_reps]
    if not np.isfinite(risks).all() or not np.isfinite(risk_005).all():
        raise RuntimeError("non-finite or missing risk values")
    coefficients_quadratic = polynomial_coefficients(risks, 2)
    coefficients_cubic = polynomial_coefficients(risks, 3)
    derivatives_quadratic = coefficients_quadratic[:, :, 1]
    derivatives_cubic = coefficients_cubic[:, :, 1]
    roots_quadratic, roots_cubic = bootstrap_roots(
        derivatives_quadratic, derivatives_cubic
    )
    primary = root_summary(derivatives_quadratic.mean(axis=1), roots_quadratic)
    primary["role"] = "preregistered main quadratic zero-point derivative"
    verdict, reasons = judge_primary(primary)
    primary["verdict"] = verdict
    primary["verdict_reasons"] = reasons
    cubic = root_summary(derivatives_cubic.mean(axis=1), roots_cubic)
    cubic["role"] = "sensitivity diagnostic only; no effect on verdict or extension"
    cubic["root_difference_from_primary"] = cubic["root"] - primary["root"]

    per_delta = []
    for delta_index, delta_norm in enumerate(DELTA_NORMS):
        rep_risks = risks[delta_index]
        mean_risk = rep_risks.mean(axis=0)
        b_005_reps = rep_risks[:, 0] - risk_005[delta_index]
        d2 = derivatives_quadratic[delta_index]
        d3 = derivatives_cubic[delta_index]
        per_delta.append({
            "delta_norm": float(delta_norm),
            "risk_mean": mean_risk,
            "risk_se": rep_risks.std(axis=0, ddof=1) / np.sqrt(n_reps),
            "quadratic_derivative_mean": float(d2.mean()),
            "quadratic_derivative_se": float(d2.std(ddof=1) / np.sqrt(n_reps)),
            "cubic_derivative_mean": float(d3.mean()),
            "cubic_derivative_se": float(d3.std(ddof=1) / np.sqrt(n_reps)),
            "marginal_benefit_005": {
                "definition": "R(0)-R(0.05), paired within each rep",
                "mean": float(b_005_reps.mean()),
                "se": float(b_005_reps.std(ddof=1) / np.sqrt(n_reps)),
            },
            "global_benefit": {
                "definition": "mean R(0)-min over the six preregistered lambdas of mean R(lambda)",
                "value": float(mean_risk[0] - mean_risk.min()),
                "lambda_star": float(LAMBDAS[int(mean_risk.argmin())]),
                "used_for_verdict": False,
            },
        })
    return {
        "n_reps": n_reps,
        "primary": primary,
        "cubic_sensitivity": cubic,
        "per_delta": per_delta,
        "raw_arrays": {
            "risk_reps": risks,
            "risk_005_reps": risk_005,
            "quadratic_coefficients_reps": coefficients_quadratic,
            "cubic_coefficients_reps": coefficients_cubic,
            "dimensions": {
                "risk_reps": ["delta", "rep", "lambda"],
                "risk_005_reps": ["delta", "rep"],
                "quadratic_coefficients_reps": ["delta", "rep", "c0_c1_c2"],
                "cubic_coefficients_reps": ["delta", "rep", "c0_c1_c2_c3"],
            },
        },
    }


def stage_record(analysis):
    primary = analysis["primary"]
    bootstrap = primary["bootstrap"]
    return {
        "n_reps": analysis["n_reps"],
        "root": primary["root"],
        "ci_lower": bootstrap["lower"],
        "ci_upper": bootstrap["upper"],
        "relative_halfwidth": bootstrap["halfwidth_over_theory_root"],
        "theory_root_in_ci": bootstrap["theory_root_in_ci"],
        "verdict": primary["verdict"],
        "verdict_reasons": primary["verdict_reasons"],
        "cubic_root": analysis["cubic_sensitivity"]["root"],
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
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
    safe_result = json_safe(result)
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(safe_result, handle, ensure_ascii=False, indent=1, allow_nan=False)
        handle.write("\n")
    print(f"saved {JSON_PATH}", flush=True)
    return safe_result


def format_number(value, digits=10):
    return "NA" if value is None else f"{value:.{digits}f}"


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = result["analysis"]
    primary = analysis["primary"]
    bootstrap = primary["bootstrap"]
    rows = analysis["per_delta"]
    delta_norms = np.array(result["config"]["delta_norms"])
    lambdas = np.array(result["config"]["lambdas"])
    d2_mean = np.array([row["quadratic_derivative_mean"] for row in rows])
    d2_se = np.array([row["quadratic_derivative_se"] for row in rows])
    d3_mean = np.array([row["cubic_derivative_mean"] for row in rows])

    fig, (ax_root, ax_risk, ax_aux) = plt.subplots(1, 3, figsize=(18, 5.4))
    ax_root.errorbar(
        delta_norms, d2_mean, yerr=d2_se, fmt="o", capsize=3,
        color="C0", label="quadratic derivative (mean ± SE)",
    )
    fit = primary["unweighted_linear_fit"]
    ax_root.plot(
        delta_norms, fit["intercept"] + fit["slope"] * delta_norms,
        color="C0", lw=1.5, label="unweighted linear fit",
    )
    ax_root.plot(
        delta_norms, d3_mean, "s--", ms=4, lw=1.0, color="C2",
        label="cubic sensitivity (diagnostic only)",
    )
    ax_root.axhline(0.0, color="black", lw=0.8)
    ax_root.axvline(
        THEORY_ROOT, color="black", ls=":", lw=1.4,
        label=f"theory root={THEORY_ROOT:.10f}",
    )
    if primary["root"] is not None:
        ax_root.axvline(
            primary["root"], color="C1", ls="-.", lw=1.3,
            label=f"estimated root={primary['root']:.6f}",
        )
    if bootstrap["lower"] is not None:
        ax_root.axvspan(
            bootstrap["lower"], bootstrap["upper"], color="C1", alpha=0.12,
            label="joint bootstrap 95% CI",
        )
    ax_root.set_xlabel(r"$\|\delta\|$")
    ax_root.set_ylabel(r"estimated $R'(0^+)$")
    relative_halfwidth = format_number(bootstrap["halfwidth_over_theory_root"], 3)
    ax_root.set_title(
        f"E1 phase boundary: {primary['verdict']} | "
        f"CI halfwidth/theory={relative_halfwidth}"
    )
    ax_root.grid(alpha=0.3)
    ax_root.legend(fontsize=7)

    for index, row in enumerate(rows):
        ax_risk.plot(
            lambdas, row["risk_mean"], "o-", ms=4, lw=1.1,
            color=f"C{index}", label=f"‖δ‖={row['delta_norm']:.4f}",
        )
    ax_risk.set_xlabel(r"$\lambda$")
    ax_risk.set_ylabel(r"E1 $\Sigma_r$-weighted excess risk")
    ax_risk.set_title("Mean risk curves (six shared-data λ points)")
    ax_risk.grid(alpha=0.3)
    ax_risk.legend(fontsize=7)

    b005 = np.array([row["marginal_benefit_005"]["mean"] for row in rows])
    b005_se = np.array([row["marginal_benefit_005"]["se"] for row in rows])
    global_benefit = np.array([row["global_benefit"]["value"] for row in rows])
    ax_aux.errorbar(
        delta_norms, b005, yerr=b005_se, fmt="o-", capsize=3,
        color="C0", label="B(0.05)=R(0)-R(0.05)",
    )
    ax_aux.plot(
        delta_norms, global_benefit, "s--", color="C4",
        label="global g on mean six-λ curve",
    )
    ax_aux.axhline(0.0, color="black", lw=0.8)
    ax_aux.set_xlabel(r"$\|\delta\|$")
    ax_aux.set_ylabel("benefit (auxiliary; not a verdict)")
    ax_aux.set_title("Marginal and global benefits reported separately")
    ax_aux.grid(alpha=0.3)
    ax_aux.legend(fontsize=8)

    fig.suptitle(
        f"N1d-new independent within-model E1 check | family A, low-frequency anti-aligned δ | "
        f"seed={SEED_BASE}, reps={analysis['n_reps']}, joint bootstrap={N_BOOTSTRAP}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    print(f"saved {PNG_PATH}", flush=True)


def summarize(result):
    analysis = result["analysis"]
    primary = analysis["primary"]
    bootstrap = primary["bootstrap"]
    cubic = analysis["cubic_sensitivity"]
    cubic_bootstrap = cubic["bootstrap"]
    print("\n=== N1d-new independent E1 ===")
    print(f"reps={analysis['n_reps']}; verdict={primary['verdict']}; "
          f"reasons={primary['verdict_reasons']}")
    print(f"quadratic root={format_number(primary['root'])}; "
          f"95% CI=[{format_number(bootstrap['lower'])}, "
          f"{format_number(bootstrap['upper'])}]; "
          f"halfwidth/theory={format_number(bootstrap['halfwidth_over_theory_root'], 6)}; "
          f"theory in CI={bootstrap['theory_root_in_ci']}")
    print(f"cubic diagnostic root={format_number(cubic['root'])}; "
          f"95% CI=[{format_number(cubic_bootstrap['lower'])}, "
          f"{format_number(cubic_bootstrap['upper'])}]; "
          f"root difference={cubic['root_difference_from_primary']}")


def main():
    t0 = time.time()
    max_reps = REPS_SCHEDULE[-1]
    risks = np.full((N_DELTA, max_reps, len(LAMBDAS)), np.nan)
    risk_005 = np.full((N_DELTA, max_reps), np.nan)
    history = []
    decisions = []
    completed = 0
    result = None

    for target in REPS_SCHEDULE:
        if target > completed:
            compute_batch(completed, target, risks, risk_005)
            completed = target
        analysis = analyze(risks, risk_005, target)
        history.append(stage_record(analysis))
        primary = analysis["primary"]
        extend = primary["bootstrap"]["ci_too_wide"] and target < max_reps
        if extend:
            next_target = REPS_SCHEDULE[REPS_SCHEDULE.index(target) + 1]
            decisions.append({
                "n_reps": target,
                "action": f"extend_to_{next_target}",
                "reason": "primary 95% CI halfwidth/theory root > 0.20",
            })
            run_state = "extension_required_by_preregistered_ci_width_rule"
        else:
            if target == max_reps and primary["bootstrap"]["ci_too_wide"]:
                action = "stop_at_preregistered_1600_rep_cap"
            elif primary["verdict"] == "PASS":
                action = "stop_pass"
            elif primary["verdict"] == "FAIL":
                action = "stop_fail_ci_narrow"
            else:
                action = "stop_unresolved_not_eligible_for_extension"
            decisions.append({
                "n_reps": target,
                "action": action,
                "reason": "automatic preregistered decision; no result-driven sample addition",
            })
            run_state = "complete"

        result = {
            "experiment": "N1d-new independent within-model E1 diagonal-ridge phase-boundary check",
            "preregistration": "W4_独立验证修订执行方案.md §2 (N1d-new)",
            "run_state": run_state,
            "config": {
                "estimator": "E1 diagonal ridge",
                "p": P, "s": C.S_SMOOTH, "mu": MU,
                "n_r": N_R, "n_h": N_H,
                "sigma_r2": SIG_R2, "sigma_h2": SIG_H2,
                "spectrum_family": "A", "alpha_r": ALPHA_R, "alpha_h": ALPHA_H,
                "delta_shape": "lowfreq", "anti_aligned": True,
                "delta_norms": DELTA_NORMS,
                "theory_root": THEORY_ROOT,
                "theory_root_is_candidate": True,
                "lambdas": LAMBDAS,
                "aux_lambda": AUX_LAMBDA,
                "initial_reps": REPS_SCHEDULE[0],
                "allowed_reps_schedule": REPS_SCHEDULE,
                "actual_reps": target,
                "bootstrap_reps": N_BOOTSTRAP,
                "ci_level": CI_LEVEL,
                "max_ci_halfwidth_over_theory_root": MAX_RELATIVE_HALFWIDTH,
                "seed_base": SEED_BASE,
                "data_seed_scheme": "default_rng(SeedSequence([20261025, 0, rep])); rep is zero-based",
                "bootstrap_seed_scheme": "default_rng(SeedSequence([20261025, 1]))",
                "rep_indices": list(range(target)),
                "workers": N_WORKERS,
            },
            "methods": {
                "risk": "sum_j lam_r,j (beta_hat_j-beta_star_j)^2",
                "moment_reuse": "one per-coordinate Gram and label projection per rep; all λ values reuse S_r,T_r,S_h,T_h",
                "paired_delta_construction": "one random data stream per rep; T_h(delta)=T_h(0)+norm*T_h(unit lowfreq anti-aligned delta)",
                "quadratic_derivative": "unweighted six-point least squares in z=lambda/0.08; original-scale linear coefficient c1 is R'(0+) estimate",
                "cubic_sensitivity": "same six-point data and unweighted least squares with degree 3; diagnostic only",
                "root": "unweighted OLS of mean derivative on seven delta norms, including theory candidate with weight 1; root=-d0/d1",
                "bootstrap": "1000 joint resamples of the same rep indices across all delta norms; identical indices for quadratic and cubic diagnostics; percentile 95% CI",
                "extension_rule": "extend 400->800->1600 only when primary CI halfwidth/theory root > 0.20; never triggered by cubic fit or desired verdict",
                "global_benefit": "computed on the mean six-λ risk curve; never used as a zero-derivative substitute or verdict input",
            },
            "theory": {"root": THEORY_ROOT},
            "analysis": analysis,
            "stage_history": history,
            "sample_size_decisions": decisions,
            "implementation_deviations": [],
            "elapsed_seconds_at_write": time.time() - t0,
        }
        result = write_json(result)
        if not extend:
            break

    plot(result)
    summarize(result)


if __name__ == "__main__":
    main()
