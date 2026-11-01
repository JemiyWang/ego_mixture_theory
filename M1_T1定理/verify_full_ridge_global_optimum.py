"""Uniform full-ridge risk derivatives and nondegenerate interior optima."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from verify_full_ridge_second_order import (
    actual_risk_draws,
    base_configurations,
    risk_coefficients,
    summarize_draws,
)
from verify_full_ridge_zero_derivative import sample_gram


def coefficient_derivatives(w, sigma_r, sigma_h, beta, delta, mu, sigma_r2, sigma_h2, rho):
    h = 1e-20
    r0, r1, _ = risk_coefficients(w + 1j * h, sigma_r, sigma_h, beta, delta,
                                  mu, sigma_r2, sigma_h2, rho)
    return float(r0.imag / h), float(r1.imag / h)


def conditional_derivative_draws(rng, reps, n_r, w, sigma_r, sigma_h, beta, delta,
                                 mu, sigma_r2, sigma_h2, rho):
    p = len(beta)
    a, b = np.diag(sigma_r), np.diag(sigma_h)
    n_h = round(rho * n_r)
    eye = np.eye(p)
    v = -mu * beta + w * sigma_h @ delta
    out = np.empty(reps)
    for rep in range(reps):
        gram_r = sample_gram(rng, n_r, a)
        gram_h = sample_gram(rng, n_h, b)
        gap = gram_h - gram_r
        resolvent = np.linalg.inv((1 - w) * gram_r + w * gram_h + mu * eye)
        resolvent_prime = -resolvent @ gap @ resolvent
        c = resolvent @ sigma_r @ resolvent
        c_prime = (resolvent_prime @ sigma_r @ resolvent
                   + resolvent @ sigma_r @ resolvent_prime)
        q = v + w * (gram_h - sigma_h) @ delta
        q_prime = gram_h @ delta
        noise = ((1 - w)**2 * sigma_r2 / n_r * gram_r
                 + w**2 * sigma_h2 / n_h * gram_h)
        noise_prime = (-2 * (1 - w) * sigma_r2 / n_r * gram_r
                       + 2 * w * sigma_h2 / n_h * gram_h)
        bias_derivative = 2 * q_prime @ c @ q + q @ c_prime @ q
        noise_derivative = np.trace(sigma_r @ (
            resolvent_prime @ noise @ resolvent
            + resolvent @ noise_prime @ resolvent
            + resolvent @ noise @ resolvent_prime))
        out[rep] = bias_derivative + noise_derivative
    return out


def conditional_risk_grid(rng, reps, n_r, weights, sigma_r, sigma_h, beta, delta,
                          mu, sigma_r2, sigma_h2, rho):
    p = len(beta)
    a, b = np.diag(sigma_r), np.diag(sigma_h)
    n_h = round(rho * n_r)
    eye = np.eye(p)
    sums = np.zeros(len(weights))
    sums2 = np.zeros(len(weights))
    for rep in range(reps):
        gram_r = sample_gram(rng, n_r, a)
        gram_h = sample_gram(rng, n_h, b)
        for i, w in enumerate(weights):
            resolvent = np.linalg.inv((1 - w) * gram_r + w * gram_h + mu * eye)
            v = -mu * beta + w * sigma_h @ delta
            q = v + w * (gram_h - sigma_h) @ delta
            noise = ((1 - w)**2 * sigma_r2 / n_r * gram_r
                     + w**2 * sigma_h2 / n_h * gram_h)
            risk = q @ resolvent @ sigma_r @ resolvent @ q + np.trace(sigma_r @ resolvent @ noise @ resolvent)
            sums[i] += risk
            sums2[i] += risk**2
    means = sums / reps
    variances = np.maximum(0.0, (sums2 - reps * means**2) / (reps - 1))
    return means, np.sqrt(variances / reps)


def actual_central_difference(rng, reps, n_r, w, step, sigma_r, sigma_h, beta, delta,
                              mu, sigma_r2, sigma_h2, rho):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "M2_实验设计" / "n1_toy"))
    from estimators import e2_full_ridge
    from gen_data import gen_two_domain

    a, b = np.diag(sigma_r), np.diag(sigma_h)
    n_h = round(rho * n_r)
    out = np.empty(reps)
    for rep in range(reps):
        rob, hum = gen_two_domain(rng, n_r, n_h, a, b, beta, delta, sigma_r2, sigma_h2)
        left = e2_full_ridge(rob, hum, w - step, mu) - beta
        right = e2_full_ridge(rob, hum, w + step, mu) - beta
        out[rep] = (a @ right**2 - a @ left**2) / (2 * step)
    return out


def main(reps, grid_reps, direct_reps, output):
    rng = np.random.default_rng(20261029)
    configs = {name: config for name, config, _ in base_configurations()}
    result = {"derivative_cells": [], "direct_derivative_checks": [], "optimum_checks": []}
    derivative_cases = [
        ("p1_cancellation", 0.5), ("p2_aligned", 0.3), ("p2_anti_aligned", 0.3),
        ("p20_spectral_A", 0.0), ("p20_spectral_A", 0.1),
        ("p20_spectral_A", 0.3), ("p20_spectral_A", 0.5), ("p20_spectral_A", 1.0),
    ]
    for name, w in derivative_cases:
        a, b, beta, delta, mu, sigma_r2, sigma_h2, rho = configs[name]
        sigma_r, sigma_h = np.diag(a), np.diag(b)
        d0, d1 = coefficient_derivatives(w, sigma_r, sigma_h, beta, delta,
                                         mu, sigma_r2, sigma_h2, rho)
        for n_r in [1000, 4000]:
            draws = conditional_derivative_draws(rng, reps, n_r, w, sigma_r, sigma_h,
                                                 beta, delta, mu, sigma_r2, sigma_h2, rho)
            mean, se = summarize_draws(draws)
            prediction = d0 + d1 / n_r
            row = {"configuration": name, "w": w, "n_r": n_r,
                   "R0_prime": d0, "R1_prime": d1, "prediction": prediction,
                   "conditional_mc_mean": mean, "standard_error": se,
                   "error_in_standard_errors": (mean - prediction) / se}
            result["derivative_cells"].append(row)
            print(f"DER {name:18s} w={w:.1f} n={n_r}: pred={prediction:+.8f} "
                  f"MC={mean:+.8f} z={row['error_in_standard_errors']:+.2f}")
    for name in ["p2_aligned", "p20_spectral_A"]:
        a, b, beta, delta, mu, sigma_r2, sigma_h2, rho = configs[name]
        sigma_r, sigma_h = np.diag(a), np.diag(b)
        d0, d1 = coefficient_derivatives(0.3, sigma_r, sigma_h, beta, delta,
                                         mu, sigma_r2, sigma_h2, rho)
        draws = actual_central_difference(rng, direct_reps, 1000, 0.3, 1e-4,
                                          sigma_r, sigma_h, beta, delta, mu,
                                          sigma_r2, sigma_h2, rho)
        mean, se = summarize_draws(draws)
        prediction = d0 + d1 / 1000
        row = {"configuration": name, "w": 0.3, "step": 1e-4,
               "prediction": prediction, "actual_mc_mean": mean,
               "standard_error": se, "error_in_standard_errors": (mean - prediction) / se}
        result["direct_derivative_checks"].append(row)
        print(f"DIRECT {name:15s} pred={prediction:+.8f} MC={mean:+.8f} "
              f"z={row['error_in_standard_errors']:+.2f}")

    for name in ["p1_cancellation", "p2_aligned"]:
        a, b, beta, delta, mu, sigma_r2, sigma_h2, rho = configs[name]
        sigma_r, sigma_h = np.diag(a), np.diag(b)
        objective = lambda w: float(risk_coefficients(w, sigma_r, sigma_h, beta, delta,
                                                      mu, sigma_r2, sigma_h2, rho)[0])
        opt = minimize_scalar(objective, bounds=(0, 1), method="bounded",
                              options={"xatol": 1e-12})
        w0 = float(opt.x)
        r0, r1, _ = risk_coefficients(w0, sigma_r, sigma_h, beta, delta,
                                      mu, sigma_r2, sigma_h2, rho)
        h = 1e-4
        curvature = (objective(w0 + h) - 2 * objective(w0) + objective(w0 - h)) / h**2
        if not (1e-6 < w0 < 1 - 1e-6 and curvature > 0):
            raise AssertionError(f"Nondegenerate interior optimum required: {name}")
        for n_r in [1000, 4000]:
            if name == "p1_cancellation":
                weights = np.array([w0])
            else:
                weights = np.linspace(w0 - 0.05, w0 + 0.05, 41)
            means, ses = conditional_risk_grid(rng, grid_reps, n_r, weights,
                                               sigma_r, sigma_h, beta, delta, mu,
                                               sigma_r2, sigma_h2, rho)
            i = int(np.argmin(means))
            prediction = float(r0 + r1 / n_r)
            row = {"configuration": name, "n_r": n_r, "w0": w0,
                   "R0_curvature": float(curvature), "R0_min": float(r0),
                   "R1_at_w0": float(r1), "optimal_risk_prediction": prediction,
                   "mc_grid_w": float(weights[i]), "mc_optimal_risk": float(means[i]),
                   "mc_standard_error_at_grid_min": float(ses[i]),
                   "grid_step": float(weights[1] - weights[0]) if len(weights) > 1 else 0.0,
                   "error_in_standard_errors": float((means[i] - prediction) / ses[i])}
            result["optimum_checks"].append(row)
            print(f"OPT {name:18s} n={n_r}: w0={w0:.6f} grid_w={weights[i]:.4f} "
                  f"pred={prediction:.9f} MC={means[i]:.9f} z={row['error_in_standard_errors']:+.2f}")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--grid-reps", type=int, default=5_000)
    parser.add_argument("--direct-reps", type=int, default=5_000)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("full_ridge_global_optimum_results.json"))
    args = parser.parse_args()
    main(args.reps, args.grid_reps, args.direct_reps, args.output)
