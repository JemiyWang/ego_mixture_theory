"""Boundary optima, flat-boundary scaling, and global-shape checks."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from verify_full_ridge_global_optimum import coefficient_derivatives
from verify_full_ridge_second_order import base_configurations, risk_coefficients
from verify_full_ridge_zero_derivative import sample_gram


def conditional_risk_and_gains(rng, reps, n_r, weights, sigma_r, sigma_h, beta, delta,
                               mu, sigma_r2, sigma_h2, rho):
    p = len(beta)
    a, b = np.diag(sigma_r), np.diag(sigma_h)
    n_h = round(rho * n_r)
    eye = np.eye(p)
    sums = np.zeros(len(weights))
    gain_sums2 = np.zeros(len(weights))
    baseline_sum = 0.0
    for rep in range(reps):
        gram_r = sample_gram(rng, n_r, a)
        gram_h = sample_gram(rng, n_h, b)
        risks = np.empty(len(weights))
        for i, w in enumerate(weights):
            resolvent = np.linalg.inv((1 - w) * gram_r + w * gram_h + mu * eye)
            v = -mu * beta + w * sigma_h @ delta
            q = v + w * (gram_h - sigma_h) @ delta
            noise = ((1 - w)**2 * sigma_r2 / n_r * gram_r
                     + w**2 * sigma_h2 / n_h * gram_h)
            risks[i] = q @ resolvent @ sigma_r @ resolvent @ q + np.trace(sigma_r @ resolvent @ noise @ resolvent)
        baseline_sum += risks[0]
        sums += risks
        gain_sums2 += (risks[0] - risks)**2
    means = sums / reps
    baseline = baseline_sum / reps
    gains = baseline - means
    gain_variances = np.maximum(0.0, (gain_sums2 - reps * gains**2) / (reps - 1))
    return baseline, means, gains, np.sqrt(gain_variances / reps)


def main():
    rng = np.random.default_rng(20261030)
    configs = {name: config for name, config, _ in base_configurations()}
    output = {"flat_boundary": [], "nondegenerate_boundary": []}
    a, b, beta, delta0, mu, sigma_r2, sigma_h2, rho = configs["p2_anti_aligned"]
    sigma_r, sigma_h = np.diag(a), np.diag(b)

    def h_at_scale(scale):
        return coefficient_derivatives(0, sigma_r, sigma_h, beta, scale * delta0,
                                       mu, sigma_r2, sigma_h2, rho)[0]

    scale = brentq(h_at_scale, 0, 1)
    delta = scale * delta0
    r0 = lambda w: float(risk_coefficients(w, sigma_r, sigma_h, beta, delta,
                                           mu, sigma_r2, sigma_h2, rho)[0])
    opt = minimize_scalar(r0, bounds=(0, 1), method="bounded")
    h = 1e-5
    curvature = (r0(h) - 2 * r0(0) + r0(-h)) / h**2
    _, r1_prime = coefficient_derivatives(0, sigma_r, sigma_h, beta, delta,
                                          mu, sigma_r2, sigma_h2, rho)
    if not (opt.x < 1e-5 and curvature > 0 and r1_prime < 0):
        raise AssertionError("The selected example is not a nondegenerate flat boundary with an outward correction")
    print(f"flat boundary: scale={scale:.12f}, curvature={curvature:.9f}, R1'={r1_prime:.9f}")
    for n_r in [100_000, 1_000_000]:
        w_prediction = -r1_prime / (n_r * curvature)
        gain_prediction = r1_prime**2 / (2 * curvature * n_r**2)
        candidates = np.linspace(0.5 * w_prediction, 1.5 * w_prediction, 31)
        weights = np.concatenate([[0.0], candidates])
        baseline, means, gains, gain_ses = conditional_risk_and_gains(
            rng, 10_000, n_r, weights, sigma_r, sigma_h, beta, delta,
            mu, sigma_r2, sigma_h2, rho)
        i = int(np.argmin(means))
        row = {"scale": scale, "n_r": n_r, "curvature": curvature,
               "R1_prime": r1_prime, "w_prediction": w_prediction,
               "gain_prediction": gain_prediction, "mc_w": float(weights[i]),
               "mc_gain": float(gains[i]), "mc_gain_standard_error": float(gain_ses[i]),
               "baseline_risk": baseline, "mc_optimal_risk": float(means[i]),
               "grid_step": float(candidates[1] - candidates[0])}
        output["flat_boundary"].append(row)
        print(f"n={n_r}: w pred={w_prediction:.7f} MC={weights[i]:.7f}; "
              f"gain pred={gain_prediction:.3e} MC={gains[i]:.3e} +/- {gain_ses[i]:.1e}")

    a, b, beta, delta, mu, sigma_r2, sigma_h2, rho = configs["p2_anti_aligned"]
    sigma_r, sigma_h = np.diag(a), np.diag(b)
    weights = np.linspace(0, 1, 21)
    baseline, means, gains, gain_ses = conditional_risk_and_gains(
        rng, 10_000, 1000, weights, sigma_r, sigma_h, beta, delta,
        mu, sigma_r2, sigma_h2, rho)
    i = int(np.argmin(means))
    output["nondegenerate_boundary"].append({
        "configuration": "p2_anti_aligned", "n_r": 1000,
        "mc_w": float(weights[i]), "mc_optimal_risk": float(means[i]),
        "baseline_risk": baseline, "grid_step": 0.05})
    print(f"nondegenerate boundary: w_MC={weights[i]:.2f}, R0={baseline:.9f}, Rmin={means[i]:.9f}")
    target = Path(__file__).with_name("full_ridge_boundary_results.json")
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
