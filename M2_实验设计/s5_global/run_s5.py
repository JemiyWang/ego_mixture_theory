"""Pre-registered S5 full-ridge global validation suite."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "M1_T1定理"))
sys.path.insert(0, str(ROOT / "M2_实验设计" / "n1_toy"))
from verify_full_ridge_zero_derivative import derivative_coefficients
from verify_full_ridge_second_order import (
    base_configurations,
    risk_coefficients,
    summarize_draws,
)
from verify_full_ridge_global_optimum import (
    coefficient_derivatives,
    conditional_derivative_draws,
    actual_central_difference,
)
from verify_full_ridge_boundary import conditional_risk_and_gains
from verify_full_ridge_dosage_exchange import (
    full_optimum,
    leading_dose,
    leading_direct,
    matched_coefficients,
)
from estimators import e2_full_ridge
from gen_data import gen_two_domain

OUT = Path(__file__).with_name("outputs")
OUT.mkdir(exist_ok=True)
BASE = {name: config for name, config, _ in base_configurations()}
RNG = np.random.default_rng(20261030)


def root_bracket(ns, values, predicted):
    for i in range(len(ns) - 1):
        if values[i] == 0:
            return float(ns[i])
        if values[i] * values[i + 1] < 0:
            root = ns[i] - values[i] * (ns[i + 1] - ns[i]) / (values[i + 1] - values[i])
            return float(root)
    return None


def bootstrap_ratio(ns, draws, predicted, rng):
    estimates = []
    for _ in range(1000):
        means = [rng.choice(draw, size=len(draw), replace=True).mean() for draw in draws]
        root = root_bracket(ns, means, predicted)
        if root is not None:
            estimates.append(root / predicted)
    if not estimates:
        return {"status": "no_bootstrap_root", "compatible": False}
    lo, hi = np.percentile(estimates, [1.25, 98.75])
    return {"root_ratio_bootstrap_median": float(np.median(estimates)),
            "root_ratio_ci_97_5": [float(lo), float(hi)],
            "compatible": bool(lo >= 0.8 and hi <= 1.2),
            "bootstrap_successful_roots": len(estimates)}


def run_boundary(name, ns):
    a, b, beta, delta, mu, sigma_r2, _, _ = BASE[name]
    sigma_r, sigma_h = np.diag(a), np.diag(b)
    theory = derivative_coefficients(sigma_r, sigma_h, beta, delta, mu, sigma_r2)
    predicted = theory["K_full"] / theory["H"]
    cells, draws = [], []
    for n_r in ns:
        values = conditional_derivative_draws(RNG, 400, n_r, 0.0, sigma_r, sigma_h,
                                              beta, delta, mu, sigma_r2, 4.0, 6.0)
        mean, se = summarize_draws(values)
        prediction = theory["H"] - theory["K_full"] / n_r
        cells.append({"n_r": n_r, "derivative_prediction": prediction,
                      "derivative_mc": mean, "standard_error": se})
        draws.append(values)
    means = [cell["derivative_mc"] for cell in cells]
    root = root_bracket(ns, means, predicted)
    result = {"configuration": name, "theory": theory,
              "predicted_root": predicted, "cells": cells,
              "estimated_root": root}
    if root is None:
        result.update({"status": "no_sign_bracket", "compatible": False})
    else:
        result.update(bootstrap_ratio(ns, draws, predicted, RNG))
        result["root_ratio"] = root / predicted
        result["status"] = "compatible" if result["compatible"] else "not_compatible"
    return result


def population_optimum(sigma_r, sigma_h, beta, delta, mu, sigma_r2, sigma_h2, rho, n_r):
    objective = lambda w: float(risk_coefficients(w, sigma_r, sigma_h, beta, delta,
                                                  mu, sigma_r2, sigma_h2, rho)[0])
    opt = minimize_scalar(objective, bounds=(0, 1), method="bounded")
    w0 = float(opt.x)
    r00, r10, _ = risk_coefficients(0, sigma_r, sigma_h, beta, delta,
                                    mu, sigma_r2, sigma_h2, rho)
    _, r1w, _ = risk_coefficients(w0, sigma_r, sigma_h, beta, delta,
                                  mu, sigma_r2, sigma_h2, rho)
    return w0, r00 - float(opt.fun) + (r10 - r1w) / n_r


def run_global(name, ns):
    a, b, beta, delta, mu, sigma_r2, sigma_h2, rho = BASE[name]
    sigma_r, sigma_h = np.diag(a), np.diag(b)
    cells = []
    for n_r in ns:
        weights = np.linspace(0, 1, 21)
        baseline, means, gains, gain_ses = conditional_risk_and_gains(
            RNG, 200, n_r, weights, sigma_r, sigma_h, beta, delta,
            mu, sigma_r2, sigma_h2, rho)
        i = int(np.argmin(means))
        w0, gain_prediction = population_optimum(sigma_r, sigma_h, beta, delta,
                                                 mu, sigma_r2, sigma_h2, rho, n_r)
        cells.append({"n_r": n_r, "weights": weights.tolist(),
                      "risk_means": means.tolist(), "baseline_risk": baseline,
                      "mc_w": float(weights[i]), "mc_optimal_risk": float(means[i]),
                      "mc_gain": float(gains[i]), "mc_gain_se": float(gain_ses[i]),
                      "theory_w0": w0, "gain_prediction": gain_prediction})
    return {"configuration": name, "cells": cells}


def actual_paired_gain(rng, reps, n_r, w, sigma_r, sigma_h, beta, delta,
                       mu, sigma_r2, sigma_h2, rho):
    a, b = np.diag(sigma_r), np.diag(sigma_h)
    n_h = round(rho * n_r)
    gains = np.empty(reps)
    for rep in range(reps):
        rob, hum = gen_two_domain(rng, n_r, n_h, a, b, beta, delta, sigma_r2, sigma_h2)
        base = e2_full_ridge(rob, hum, 0.0, mu) - beta
        mixed = e2_full_ridge(rob, hum, w, mu) - beta
        gains[rep] = a @ base**2 - a @ mixed**2
    return gains


def counterexample_config():
    return (np.diag([0.06878172866100844, 3.348404027039282, 0.013486232610484008]),
            np.diag([0.5763005499719634, 1.0560136030378924, 12.597717940912908]),
            np.array([-1.5063590767700887, 2.6917054532527134, -1.0857110393770384]),
            np.array([-0.7271735734168655, -0.8534676714155205, 0.9487393526514577]),
            0.040455141206854305, 1.0, 4.0, 2.0)


def run_actual_checks(counter_w):
    checks = []
    for name, n_r in [("p2_anti_aligned", 50), ("p2_anti_aligned", 70),
                      ("p20_spectral_A", 250), ("p20_spectral_A", 700)]:
        a, b, beta, delta, mu, sigma_r2, sigma_h2, rho = BASE[name]
        sr, sh = np.diag(a), np.diag(b)
        draws = actual_central_difference(RNG, 200, n_r, 0.0, 1e-4,
                                          sr, sh, beta, delta, mu, sigma_r2, sigma_h2, rho)
        mean, se = summarize_draws(draws)
        checks.append({"configuration": name, "n_r": n_r,
                       "actual_paired_derivative": mean, "se": se})
    for name, w, config in [
            ("matched_interior", matched_coefficients(1.0)["A1"] and None, None),
            ("counterexample", counter_w, counterexample_config())]:
        if name == "matched_interior":
            c = matched_coefficients(1.0)
            sr, sh, beta, delta, mu = c["sigma"], c["sigma"], c["beta"], c["delta"], c["mu"]
            sr2, sh2, rho = c["sigma_r2"], c["sigma_h2"], 6.0
            w = -c["A1"] / (2 * c["A2"])
        else:
            sr, sh, beta, delta, mu, sr2, sh2, rho = config
        draws = actual_paired_gain(RNG, 200, 1000, w, sr, sh, beta, delta,
                                   mu, sr2, sh2, rho)
        mean, se = summarize_draws(draws)
        checks.append({"configuration": name, "n_r": 1000, "w": w,
                       "actual_paired_gain": mean, "se": se})
    return checks


def run_flat_boundary(reps_by_n=None):
    a, b, beta, delta0, mu, sigma_r2, sigma_h2, rho = BASE["p2_anti_aligned"]
    sigma_r, sigma_h = np.diag(a), np.diag(b)
    scale = brentq(lambda s: coefficient_derivatives(
        0, sigma_r, sigma_h, beta, s * delta0, mu, sigma_r2, sigma_h2, rho)[0], 0, 1)
    delta = scale * delta0
    objective = lambda w: float(risk_coefficients(w, sigma_r, sigma_h, beta, delta,
                                                  mu, sigma_r2, sigma_h2, rho)[0])
    h = 1e-5
    curvature = (objective(h) - 2 * objective(0) + objective(-h)) / h**2
    _, r1_prime = coefficient_derivatives(0, sigma_r, sigma_h, beta, delta,
                                          mu, sigma_r2, sigma_h2, rho)
    cells = []
    for n_r in [10_000, 100_000, 1_000_000]:
        w_prediction = -r1_prime / (n_r * curvature)
        gain_prediction = r1_prime**2 / (2 * curvature * n_r**2)
        if w_prediction >= 1:
            candidates = np.linspace(0, 1, 31)
        else:
            candidates = np.linspace(0.5 * w_prediction, 1.5 * w_prediction, 31)
        weights = np.concatenate([[0.0], candidates])
        reps = 400 if reps_by_n is None else reps_by_n[n_r]
        baseline, means, gains, ses = conditional_risk_and_gains(
            RNG, reps, n_r, weights, sigma_r, sigma_h, beta, delta,
            mu, sigma_r2, sigma_h2, rho)
        i = int(np.argmin(means))
        cells.append({"n_r": n_r, "scale": scale, "curvature": curvature,
                      "R1_prime": r1_prime, "w_prediction": w_prediction,
                      "gain_prediction": gain_prediction, "mc_w": float(weights[i]),
                      "mc_gain": float(gains[i]), "mc_gain_se": float(ses[i])})
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ns = np.array([cell["n_r"] for cell in cells])
    for ax, pred_key, mc_key, ylabel in [
            (axes[0], "w_prediction", "mc_w", "optimal w"),
            (axes[1], "gain_prediction", "mc_gain", "optimal gain")]:
        ax.plot(ns, [cell[pred_key] for cell in cells], "-", label="theory")
        ax.plot(ns, [cell[mc_key] for cell in cells], "o", label="MC")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n_r")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "s5_flat_boundary.png", dpi=180)
    plt.close(fig)
    return cells


def run_exchange_checks():
    rows = []
    for name, scale in [("interior", 1.0), ("hits_boundary", 0.3)]:
        c = matched_coefficients(scale)
        for rho in [1.0, 2.0, 6.0]:
            w0, g0, g1, r10, finite, kappa = full_optimum(c, rho)
            rows.append({"configuration": name, "n_r": 1000, "rho": rho,
                         "w0": float(w0), "G0": float(g0), "G1": float(g1),
                         "R1_baseline": float(r10), "finite_exchange": bool(finite),
                         "kappa": float(kappa)})
    c = matched_coefficients(1.0, n_r=40)
    w0, g0, g1, r10, finite, kappa = full_optimum(c, 6.0)
    rows.append({"configuration": "finite_example", "n_r": 40, "rho": 6.0,
                 "w0": float(w0), "G0": float(g0), "G1": float(g1),
                 "R1_baseline": float(r10), "finite_exchange": bool(finite),
                 "kappa": float(kappa)})
    return rows


def make_figures(boundaries, globals_, dose, counter):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, result in zip(axes.flat[:2], boundaries):
        cells = result["cells"]
        ns = np.array([c["n_r"] for c in cells])
        mc = np.array([c["derivative_mc"] for c in cells])
        se = np.array([c["standard_error"] for c in cells])
        theory = np.array([c["derivative_prediction"] for c in cells])
        ax.errorbar(ns, mc, yerr=1.96 * se, fmt="o", capsize=3, label="design MC")
        ax.plot(ns, theory, "-", label="theory")
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(result["predicted_root"], color="r", ls="--", label="predicted root")
        ax.set_xscale("log")
        ax.set_title(result["configuration"])
        ax.set_xlabel("n_r")
        ax.set_ylabel("R'(0+)")
        ax.legend(fontsize=7)
    for ax, result in zip(axes.flat[2:4], globals_[:2]):
        cells = result["cells"]
        ns = np.array([c["n_r"] for c in cells])
        mc = np.array([c["mc_gain"] for c in cells])
        pred = np.array([c["gain_prediction"] for c in cells])
        ax.plot(ns, mc, "o", label="MC optimal gain")
        ax.plot(ns, pred, "-", label="theory")
        ax.set_xscale("log")
        ax.set_title(result["configuration"])
        ax.set_xlabel("n_r")
        ax.set_ylabel("optimal gain")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "s5_overview.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, scale) in zip(axes, [("interior", 1.0), ("hits_boundary", 0.3)]):
        c = matched_coefficients(scale)
        nh = np.logspace(np.log10(30), 6, 200)
        gains = np.array([leading_dose(c, x)[1] for x in nh])
        ax.plot(nh, gains, label="piecewise theory")
        points = [x for x in dose if x["configuration"] == name]
        ax.plot([x["n_h"] for x in points], [x["gain_direct"] for x in points],
                "o", label="direct grid")
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("n_h")
        ax.set_ylabel("leading gain")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "s5_dosage.png", dpi=180)
    plt.close(fig)

    sr, sh, beta, delta, mu, sr2, sh2, rho = counter
    grid = np.linspace(0, 0.25, 200)
    risk = [float(risk_coefficients(w, sr, sh, beta, delta, mu, sr2, sh2, rho)[0]) for w in grid]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, risk, label="population risk")
    ax.axvline(0.068289747, color="r", ls="--", label="global optimum")
    ax.set_xlabel("w")
    ax.set_ylabel("R0(w)")
    ax.set_title("Local harm at zero, global benefit at w>0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "s5_counterexample.png", dpi=180)
    plt.close(fig)


def main():
    boundaries = [run_boundary("p2_anti_aligned", [30, 50, 70, 100]),
                  run_boundary("p20_spectral_A", [250, 450, 700, 1000])]
    (OUT / "boundary_results.json").write_text(json.dumps(boundaries, indent=2), encoding="utf-8")
    for result in boundaries:
        print(result["configuration"], result["status"],
              "root", result.get("estimated_root"), "ratio", result.get("root_ratio"))

    globals_ = [run_global("p2_aligned", [100, 1000, 4000]),
                run_global("p2_anti_aligned", [30, 100, 1000]),
                run_global("p20_spectral_A", [250, 1000, 4000])]
    (OUT / "global_results.json").write_text(json.dumps(globals_, indent=2), encoding="utf-8")

    dose = []
    for name, scale in [("interior", 1.0), ("hits_boundary", 0.3)]:
        c = matched_coefficients(scale)
        for n_h in [30, 100, 300, 1000, 1779.1732077446363, 3000, 10000, 1000000]:
            w, g = leading_dose(c, n_h)
            wd, gd = leading_direct(c, n_h)
            dose.append({"configuration": name, "n_h": n_h, "w": w,
                         "gain": g, "w_direct": wd, "gain_direct": gd,
                         "gain_error": g - gd})
    (OUT / "dosage_results.json").write_text(json.dumps(dose, indent=2), encoding="utf-8")

    counter = counterexample_config()
    counter_w = population_optimum(*counter, 1000)[0]
    counter_cells = []
    for n_r in [100, 1000, 4000]:
        sr, sh, beta, delta, mu, sr2, sh2, rho = counter
        derivative = conditional_derivative_draws(RNG, 400, n_r, 0.0, sr, sh,
                                                  beta, delta, mu, sr2, sh2, rho)
        dm, ds = summarize_draws(derivative)
        baseline, means, gains, ses = conditional_risk_and_gains(
            RNG, 400, n_r, np.array([0.0, counter_w]), sr, sh, beta, delta,
            mu, sr2, sh2, rho)
        counter_cells.append({"n_r": n_r, "derivative_mc": dm, "derivative_se": ds,
                              "optimal_gain_mc": float(gains[1]), "gain_se": float(ses[1]),
                              "w0": counter_w})
    (OUT / "counterexample_results.json").write_text(json.dumps(counter_cells, indent=2), encoding="utf-8")

    flat = run_flat_boundary()
    (OUT / "flat_boundary_results.json").write_text(json.dumps(flat, indent=2), encoding="utf-8")
    flat_high = run_flat_boundary({10_000: 10_000, 100_000: 10_000, 1_000_000: 100_000})
    (OUT / "flat_boundary_high_reps.json").write_text(json.dumps(flat_high, indent=2), encoding="utf-8")
    exchange = run_exchange_checks()
    (OUT / "exchange_results.json").write_text(json.dumps(exchange, indent=2), encoding="utf-8")

    actual = run_actual_checks(counter_w)
    (OUT / "actual_e2_checks.json").write_text(json.dumps(actual, indent=2), encoding="utf-8")
    make_figures(boundaries, globals_, dose, counter)

    summary = {"preregistration": "preregistration.md",
               "boundaries": [{"configuration": r["configuration"],
                               "status": r["status"],
                               "predicted_root": r["predicted_root"],
                               "estimated_root": r.get("estimated_root"),
                               "root_ratio": r.get("root_ratio"),
                               "root_ratio_ci_97_5": r.get("root_ratio_ci_97_5")}
                              for r in boundaries],
               "dose_max_error": max(abs(x["gain_error"]) for x in dose),
               "flat_boundary_cells": flat,
               "flat_boundary_high_reps": flat_high,
               "exchange_cells": exchange,
               "actual_e2_checks": actual,
               "counterexample_cells": counter_cells,
               "notes": "All evidence is controlled-model evidence, not external validation."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
