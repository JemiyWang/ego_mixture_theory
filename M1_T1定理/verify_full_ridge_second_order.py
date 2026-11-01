"""Fixed-p full-ridge second-order risk: algebraic and Monte-Carlo verification."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from verify_full_ridge_zero_derivative import (
    B_TOKEN,
    Z_TOKEN,
    derivative_coefficients,
    expand_factors,
    matprod,
    sample_gram,
)


def wishart_cov(middle, sigma):
    return sigma @ middle @ sigma + sigma * np.trace(sigma @ middle)


def risk_coefficients(lam, sigma_r, sigma_h, beta, delta, mu, sigma_r2, sigma_h2, rho):
    p = len(beta)
    d = (1 - lam) * sigma_r + lam * sigma_h + mu * np.eye(p)
    r = np.linalg.inv(d)
    s = r @ sigma_r @ r
    v = -mu * beta + lam * sigma_h @ delta
    t = lambda m: (1 - lam)**2 * wishart_cov(m, sigma_r) + lam**2 / rho * wishart_cov(m, sigma_h)
    u = t(r)
    c = r @ u @ s + s @ u @ r + r @ t(s) @ r
    r0 = v @ s @ v
    mismatch_square = lam**2 / rho * np.trace(s @ wishart_cov(delta[:, None] @ delta[None, :], sigma_h))
    mismatch_cross = -2 * lam**2 / rho * v @ (r @ wishart_cov(s, sigma_h) + s @ wishart_cov(r, sigma_h)) @ delta
    noise = (1 - lam)**2 * sigma_r2 * np.trace(s @ sigma_r) + lam**2 / rho * sigma_h2 * np.trace(s @ sigma_h)
    r1 = v @ c @ v + mismatch_square + mismatch_cross + noise
    return np.asarray(r0), np.asarray(r1), c


def brute_resolvent_correction(d, sigma_r, covariance):
    r = np.linalg.inv(d)
    zeroth = np.zeros_like(d)
    first = np.zeros_like(d)
    for seq, coeff in expand_factors([B_TOKEN, sigma_r, B_TOKEN], r, sigma_r):
        z_positions = [i for i, value in enumerate(seq)
                       if isinstance(value, str) and value == Z_TOKEN]
        if len(z_positions) == 0:
            zeroth += coeff * matprod(seq, len(d))
        elif len(z_positions) == 2:
            i, j = z_positions
            first += coeff * (matprod(seq[:i], len(d)) @ covariance(matprod(seq[i + 1:j], len(d)))
                              @ matprod(seq[j + 1:], len(d)))
    return zeroth, first


def conditional_risk_draws(rng, reps, n_r, lam, sigma_r, sigma_h, beta, delta,
                           mu, sigma_r2, sigma_h2, rho):
    p = len(beta)
    a, b = np.diag(sigma_r), np.diag(sigma_h)
    n_h = round(rho * n_r)
    eye = np.eye(p)
    v = -mu * beta + lam * sigma_h @ delta
    out = np.empty(reps)
    for rep in range(reps):
        gram_r = sample_gram(rng, n_r, a)
        gram_h = sample_gram(rng, n_h, b)
        resolvent = np.linalg.inv((1 - lam) * gram_r + lam * gram_h + mu * eye)
        q = v + lam * (gram_h - sigma_h) @ delta
        noise = ((1 - lam)**2 * sigma_r2 / n_r * gram_r
                 + lam**2 * sigma_h2 / n_h * gram_h)
        out[rep] = q @ resolvent @ sigma_r @ resolvent @ q + np.trace(sigma_r @ resolvent @ noise @ resolvent)
    return out


def actual_risk_draws(rng, reps, n_r, lam, sigma_r, sigma_h, beta, delta,
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
        estimate = e2_full_ridge(rob, hum, lam, mu)
        error = estimate - beta
        out[rep] = a @ error**2
    return out


def base_configurations():
    p1 = (np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([0.2]), 0.1, 0.0, 0.0, 1.0)
    p2 = (np.array([1.0, 0.5]), np.array([1.2, 0.7]), np.array([0.8, -0.3]),
          np.array([0.1, -0.4]), 0.2, 1.0, 4.0, 2.0)
    p20_j = np.arange(1, 21, dtype=float)
    a, b = p20_j**-2.0, p20_j**-0.9
    beta = p20_j**-1.5
    beta /= np.sqrt(np.sum(p20_j**2 * beta**2))
    delta = -p20_j**-1.0
    delta *= 0.5 / np.linalg.norm(delta)
    p20 = (a, b, beta, delta, 0.05, 1.0, 4.0, 6.0)
    cases = [("p1_cancellation", p1, [0.5])]
    cases += [("p2_aligned", p2, [0.1, 0.3, 0.8])]
    anti = list(p2)
    anti[3] = -anti[3]
    cases += [("p2_anti_aligned", tuple(anti), [0.3])]
    cases += [("p20_spectral_A", p20, [0.0, 0.1, 0.3, 0.5, 0.8, 1.0])]
    return cases


def summarize_draws(draws):
    return float(draws.mean()), float(draws.std(ddof=1) / np.sqrt(len(draws)))


def main(reps, direct_reps, output):
    rng = np.random.default_rng(20261030)
    result = {"model": "R_full(lambda;n_r) = R0(lambda) + R1(lambda)/n_r + o(1/n_r)",
              "conditional_replicates": reps, "actual_estimator_replicates": direct_reps,
              "cells": [], "direct_estimator_checks": [], "algebra_checks": []}
    max_algebra_error = 0.0
    for config_name, config, lambdas in base_configurations():
        a, b, beta, delta, mu, sigma_r2, sigma_h2, rho = config
        sigma_r, sigma_h = np.diag(a), np.diag(b)
        zero = derivative_coefficients(sigma_r, sigma_h, beta, delta, mu, sigma_r2)
        h = 1e-20
        r0_complex, r1_complex, _ = risk_coefficients(1j * h, sigma_r, sigma_h, beta, delta,
                                                     mu, sigma_r2, sigma_h2, rho)
        derivative_error = max(abs(r0_complex.imag / h - zero["H"]),
                               abs(r1_complex.imag / h + zero["K_full"]))
        max_algebra_error = max(max_algebra_error, derivative_error)
        result["algebra_checks"].append({"configuration": config_name,
                                         "zero_derivative_consistency_error": float(derivative_error)})
        if derivative_error > 1e-10:
            raise AssertionError(f"Zero-derivative mismatch in {config_name}: {derivative_error}")
        for lam in lambdas:
            r0, r1, c = risk_coefficients(lam, sigma_r, sigma_h, beta, delta,
                                          mu, sigma_r2, sigma_h2, rho)
            d = (1 - lam) * sigma_r + lam * sigma_h + mu * np.eye(len(beta))
            covariance = lambda m: ((1 - lam)**2 * wishart_cov(m, sigma_r)
                                    + lam**2 / rho * wishart_cov(m, sigma_h))
            _, brute_c = brute_resolvent_correction(d, sigma_r, covariance)
            algebra_error = float(np.max(np.abs(c - brute_c)))
            max_algebra_error = max(max_algebra_error, algebra_error)
            if algebra_error > 1e-10:
                raise AssertionError(f"Resolvent mismatch in {config_name}: {algebra_error}")
            for n_r in [1000, 4000]:
                draws = conditional_risk_draws(rng, reps, n_r, lam, sigma_r, sigma_h, beta, delta,
                                               mu, sigma_r2, sigma_h2, rho)
                mean, se = summarize_draws(draws)
                prediction = float(r0 + r1 / n_r)
                cell = {"configuration": config_name, "lambda": lam, "n_r": n_r,
                        "n_h": round(rho * n_r), "R0": float(r0), "R1": float(r1),
                        "prediction": prediction, "conditional_mc_mean": mean,
                        "conditional_mc_standard_error": se,
                        "mc_minus_prediction": mean - prediction,
                        "error_in_standard_errors": (mean - prediction) / se,
                        "scaled_second_order_remainder": n_r * (mean - prediction),
                        "resolvent_compact_vs_brute": algebra_error}
                result["cells"].append(cell)
                print(f"{config_name:18s} lam={lam:.1f} n={n_r:4d} pred={prediction:.9f} "
                      f"MC={mean:.9f} diff={mean - prediction:+.2e} se={se:.1e}")
            if lam in ([0.3] if config_name != "p20_spectral_A" else [0.3]) and direct_reps:
                draws = actual_risk_draws(rng, direct_reps, 1000, lam, sigma_r, sigma_h, beta, delta,
                                          mu, sigma_r2, sigma_h2, rho)
                mean, se = summarize_draws(draws)
                prediction = float(r0 + r1 / 1000)
                check = {"configuration": config_name, "lambda": lam, "n_r": 1000,
                         "prediction": prediction, "actual_estimator_mc_mean": mean,
                         "actual_estimator_standard_error": se,
                         "actual_minus_prediction": mean - prediction,
                         "error_in_standard_errors": (mean - prediction) / se}
                result["direct_estimator_checks"].append(check)
                print(f"DIRECT {config_name:13s} lam={lam:.1f} pred={prediction:.9f} "
                      f"MC={mean:.9f} diff={mean - prediction:+.2e} se={se:.1e}")
    result["max_algebra_error"] = float(max_algebra_error)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"max algebra error: {max_algebra_error:.3e}")
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--direct-reps", type=int, default=5_000)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("full_ridge_second_order_risk_results.json"))
    args = parser.parse_args()
    main(args.reps, args.direct_reps, args.output)
