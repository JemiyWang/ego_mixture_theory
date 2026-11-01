"""First-order sample-size correction for the nondegenerate optimal mixing weight."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from verify_full_ridge_global_optimum import coefficient_derivatives
from verify_full_ridge_second_order import base_configurations, risk_coefficients
from verify_full_ridge_zero_derivative import sample_gram


def population_quantities(config):
    a, b, beta, delta, mu, sigma_r2, sigma_h2, rho = config
    sigma_r, sigma_h = np.diag(a), np.diag(b)
    objective = lambda w: float(risk_coefficients(w, sigma_r, sigma_h, beta, delta,
                                                  mu, sigma_r2, sigma_h2, rho)[0])
    opt = minimize_scalar(objective, bounds=(0, 1), method="bounded",
                          options={"xatol": 1e-13})
    w0 = float(opt.x)
    h = 1e-4
    left = coefficient_derivatives(w0 - h, sigma_r, sigma_h, beta, delta,
                                   mu, sigma_r2, sigma_h2, rho)[0]
    right = coefficient_derivatives(w0 + h, sigma_r, sigma_h, beta, delta,
                                    mu, sigma_r2, sigma_h2, rho)[0]
    curvature = (right - left) / (2 * h)
    r1_prime = coefficient_derivatives(w0, sigma_r, sigma_h, beta, delta,
                                       mu, sigma_r2, sigma_h2, rho)[1]
    if not (1e-6 < w0 < 1 - 1e-6 and curvature > 0):
        raise AssertionError("A nondegenerate interior population optimum is required")
    return {
        "config": config,
        "w0": w0,
        "curvature": float(curvature),
        "R1_prime": float(r1_prime),
        "shift_constant": float(-r1_prime / curvature),
    }


def conditional_derivative_grid(rng, reps, n_r, weights, config):
    a, b, beta, delta, mu, sigma_r2, sigma_h2, rho = config
    sigma_r, sigma_h = np.diag(a), np.diag(b)
    p = len(beta)
    n_h = round(rho * n_r)
    eye = np.eye(p)
    out = np.empty((reps, len(weights)))
    for rep in range(reps):
        gram_r = sample_gram(rng, n_r, a)
        gram_h = sample_gram(rng, n_h, b)
        gap = gram_h - gram_r
        zh = gram_h - sigma_h
        for i, w in enumerate(weights):
            resolvent = np.linalg.inv((1 - w) * gram_r + w * gram_h + mu * eye)
            resolvent_prime = -resolvent @ gap @ resolvent
            c = resolvent @ sigma_r @ resolvent
            c_prime = (resolvent_prime @ sigma_r @ resolvent
                       + resolvent @ sigma_r @ resolvent_prime)
            q = -mu * beta + w * sigma_h @ delta + w * zh @ delta
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
            out[rep, i] = bias_derivative + noise_derivative
    return out


def fitted_root(weights, draws):
    center = float(np.mean(weights))
    scale = float((weights[-1] - weights[0]) / 2)
    x = (weights - center) / scale
    design = np.column_stack([np.ones(len(weights)), x])
    means = draws.mean(axis=0)
    covariance = np.cov(draws, rowvar=False) / len(draws)
    coefficients = np.linalg.lstsq(design, means, rcond=None)[0]
    inverse = np.linalg.inv(design.T @ design)
    coefficient_covariance = inverse @ design.T @ covariance @ design @ inverse
    intercept, slope = coefficients
    if slope <= 0:
        raise AssertionError("The local fitted risk curvature must be positive")
    root_scaled = -intercept / slope
    gradient = np.array([-1 / slope, intercept / slope**2])
    root = center + scale * root_scaled
    standard_error = scale * np.sqrt(gradient @ coefficient_covariance @ gradient)
    fitted = design @ coefficients
    return {
        "root": float(root),
        "standard_error": float(standard_error),
        "fitted_slope": float(slope),
        "max_derivative_fit_residual": float(np.max(np.abs(means - fitted))),
    }


def coefficient_optimum(n_r, quantities):
    config = quantities["config"]
    a, b, beta, delta, mu, sigma_r2, sigma_h2, rho = config
    sigma_r, sigma_h = np.diag(a), np.diag(b)
    objective = lambda w: float(sum(risk_coefficients(
        w, sigma_r, sigma_h, beta, delta, mu, sigma_r2, sigma_h2, rho)[:2]
        * np.array([1.0, 1.0 / n_r])))
    opt = minimize_scalar(objective, bounds=(0, 1), method="bounded",
                          options={"xatol": 1e-13})
    return float(opt.x)


def main(reps, seeds, output, names=None, n_values=None):
    configs = {name: config for name, config, _ in base_configurations()}
    selected_names = names or ["p1_cancellation", "p2_aligned"]
    quantities = {name: population_quantities(configs[name]) for name in selected_names}
    result = {"model": "n_r * (w_n^* - w0) -> -R1_prime(w0) / R0_second(w0)",
              "replicates_per_seed": reps, "independent_seeds": seeds,
              "configurations": {}, "cells": []}
    for name, quantity in quantities.items():
        result["configurations"][name] = {key: value for key, value in quantity.items()
                                          if key != "config"}
    schedules = {"p1_cancellation": [500, 1000, 2000, 4000],
                 "p2_aligned": [500, 1000, 2000, 4000]}
    halfwidths = {"p1_cancellation": 0.002, "p2_aligned": 0.0025}
    for name, quantity in quantities.items():
        w0 = quantity["w0"]
        shift = quantity["shift_constant"]
        selected_n = n_values or schedules[name]
        for n_r in selected_n:
            predicted = w0 + shift / n_r
            weights = np.linspace(predicted - halfwidths[name],
                                  predicted + halfwidths[name], 7)
            algebraic = coefficient_optimum(n_r, quantity)
            for seed_index in range(seeds):
                seed = 20261031 + 1000 * seed_index + n_r
                rng = np.random.default_rng(seed)
                draws = conditional_derivative_grid(rng, reps, n_r, weights,
                                                    quantity["config"])
                fit = fitted_root(weights, draws)
                scaled_shift = n_r * (fit["root"] - w0)
                scaled_se = n_r * fit["standard_error"]
                row = {
                    "configuration": name, "n_r": n_r, "seed_index": seed_index,
                    "seed": seed,
                    "w0": w0, "shift_constant": shift,
                    "first_order_weight_prediction": predicted,
                    "coefficient_optimum": algebraic,
                    "coefficient_scaled_shift": n_r * (algebraic - w0),
                    "mc_optimal_weight": fit["root"],
                    "mc_weight_standard_error": fit["standard_error"],
                    "mc_scaled_shift": scaled_shift,
                    "mc_scaled_shift_standard_error": scaled_se,
                    "scaled_shift_error_in_standard_errors": (scaled_shift - shift) / scaled_se,
                    "grid_halfwidth": halfwidths[name],
                    "max_derivative_fit_residual": fit["max_derivative_fit_residual"],
                }
                result["cells"].append(row)
                print(f"WEIGHT {name:18s} n={n_r:4d} seed={seed_index}: "
                      f"A={shift:+.5f} MC={scaled_shift:+.5f}±{scaled_se:.5f} "
                      f"algebra={row['coefficient_scaled_shift']:+.5f} "
                      f"z={row['scaled_shift_error_in_standard_errors']:+.2f}")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--names", nargs="+", choices=["p1_cancellation", "p2_aligned"])
    parser.add_argument("--n-values", nargs="+", type=int)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("full_ridge_weight_expansion_results.json"))
    args = parser.parse_args()
    main(args.reps, args.seeds, args.output, args.names, args.n_values)
