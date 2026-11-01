"""Global R1 shape and finite-sample checks for matched spectra and zero task mismatch."""
import json
from pathlib import Path

import numpy as np

from verify_full_ridge_global_optimum import summarize_draws
from verify_full_ridge_second_order import risk_coefficients
from verify_full_ridge_weight_expansion import conditional_derivative_grid, fitted_root
from verify_full_ridge_zero_derivative import sample_gram


def matched_coefficients(a, beta, mu, sigma_r2, sigma_h2, rho):
    sigma = np.diag(a)
    eye = np.eye(len(a))
    r = np.linalg.inv(sigma + mu * eye)
    s = r @ sigma @ r
    t = lambda m: sigma @ m @ sigma + np.trace(sigma @ m) * sigma
    u = t(r)
    correction = r @ u @ s + s @ u @ r + r @ t(s) @ r
    gamma = float(mu**2 * beta @ correction @ beta)
    noise_scale = float(np.trace(s @ sigma))
    a_coeff = sigma_r2 * noise_scale + gamma
    b_coeff = (sigma_h2 / rho) * noise_scale + gamma / rho
    return a_coeff, b_coeff, gamma, correction


def check_algebra():
    rng = np.random.default_rng(20261031)
    max_quadratic_error = 0.0
    max_r0_error = 0.0
    max_coefficient_error = 0.0
    min_correction_eigenvalue = np.inf
    for _ in range(200):
        p = int(rng.integers(1, 7))
        a = 10**rng.uniform(-3, 2, p)
        beta = rng.normal(size=p)
        mu = 10**rng.uniform(-3, 1)
        sigma_r2 = 10**rng.uniform(-3, 2)
        sigma_h2 = 10**rng.uniform(-3, 2)
        rho = 10**rng.uniform(-2, 2)
        sigma = np.diag(a)
        delta = np.zeros(p)
        ac, bc, _, correction = matched_coefficients(a, beta, mu, sigma_r2, sigma_h2, rho)
        min_correction_eigenvalue = min(min_correction_eigenvalue,
                                        float(np.linalg.eigvalsh(correction).min()))
        direct_a = float(risk_coefficients(0, sigma, sigma, beta, delta, mu,
                                           sigma_r2, sigma_h2, rho)[1])
        direct_b = float(risk_coefficients(1, sigma, sigma, beta, delta, mu,
                                           sigma_r2, sigma_h2, rho)[1])
        max_coefficient_error = max(max_coefficient_error, abs(ac - direct_a), abs(bc - direct_b))
        r0_reference = float(risk_coefficients(0, sigma, sigma, beta, delta, mu,
                                               sigma_r2, sigma_h2, rho)[0])
        for w in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
            r0, r1, _ = risk_coefficients(w, sigma, sigma, beta, delta, mu,
                                           sigma_r2, sigma_h2, rho)
            max_r0_error = max(max_r0_error, abs(float(r0) - r0_reference))
            max_quadratic_error = max(max_quadratic_error,
                                      abs(float(r1) - (ac * (1 - w)**2 + bc * w**2)))
    if max(max_quadratic_error, max_r0_error, max_coefficient_error) > 1e-8:
        raise AssertionError("The matched-spectrum quadratic identities failed")
    if min_correction_eigenvalue < -1e-10:
        raise AssertionError("The Gram correction must be positive semidefinite")
    return {
        "random_configurations": 200,
        "max_quadratic_identity_error": max_quadratic_error,
        "max_flat_r0_error": max_r0_error,
        "max_coefficient_identity_error": max_coefficient_error,
        "minimum_gram_correction_eigenvalue": min_correction_eigenvalue,
    }


def conditional_gain_draws(rng, reps, n_r, weights, config):
    a, b, beta, delta, mu, sigma_r2, sigma_h2, rho = config
    sigma_r, sigma_h = np.diag(a), np.diag(b)
    p = len(beta)
    n_h = round(rho * n_r)
    eye = np.eye(p)
    out = np.empty((reps, len(weights)))
    for rep in range(reps):
        gram_r = sample_gram(rng, n_r, a)
        gram_h = sample_gram(rng, n_h, b)
        for i, w in enumerate(weights):
            resolvent = np.linalg.inv((1 - w) * gram_r + w * gram_h + mu * eye)
            q = -mu * beta + w * sigma_h @ delta + w * (gram_h - sigma_h) @ delta
            noise = ((1 - w)**2 * sigma_r2 / n_r * gram_r
                     + w**2 * sigma_h2 / n_h * gram_h)
            out[rep, i] = (q @ resolvent @ sigma_r @ resolvent @ q
                           + np.trace(sigma_r @ resolvent @ noise @ resolvent))
    return out


def check_monte_carlo(reps):
    cases = [
        ("p1", np.array([1.0]), np.array([1.0]), 0.1, 1.0, 4.0, 2.0),
        ("p2", np.array([1.0, 0.5]), np.array([0.8, -0.3]), 0.2, 1.0, 4.0, 2.0),
    ]
    rows = []
    for name, a, beta, mu, sigma_r2, sigma_h2, rho in cases:
        config = (a, a, beta, np.zeros(len(a)), mu, sigma_r2, sigma_h2, rho)
        ac, bc, _, _ = matched_coefficients(a, beta, mu, sigma_r2, sigma_h2, rho)
        w_star = ac / (ac + bc)
        gain_coefficient = ac**2 / (ac + bc)
        for n_r in [1000, 4000]:
            weights = np.linspace(w_star - 0.02, w_star + 0.02, 9)
            rng = np.random.default_rng(20261031 + n_r)
            derivative_draws = conditional_derivative_grid(rng, reps, n_r, weights, config)
            fit = fitted_root(weights, derivative_draws)
            risk_draws = conditional_gain_draws(rng, reps, n_r, [0.0, w_star], config)
            gain_draws = risk_draws[:, 0] - risk_draws[:, 1]
            gain_mean, gain_se = summarize_draws(gain_draws)
            prediction = gain_coefficient / n_r
            rows.append({
                "configuration": name, "n_r": n_r, "replicates": reps,
                "A": ac, "B": bc, "R1_optimal_weight": w_star,
                "mc_optimal_weight": fit["root"],
                "mc_weight_standard_error": fit["standard_error"],
                "gain_prediction": prediction,
                "mc_gain_at_R1_optimal_weight": gain_mean,
                "mc_gain_standard_error": gain_se,
                "gain_error_in_standard_errors": (gain_mean - prediction) / gain_se,
            })
            print(f"FLAT {name} n={n_r}: w*={w_star:.6f} MC={fit['root']:.6f} "
                  f"gain pred={prediction:.9g} MC={gain_mean:.9g}±{gain_se:.2g}")
    return rows


def main(reps):
    result = {
        "scope": "Matched spectra and delta=0; R1 global shape and first-order finite-sample consequences",
        "algebra": check_algebra(),
        "monte_carlo": check_monte_carlo(reps),
        "degenerate_cases": {
            "A=B=0": "R1 is identically zero; all weights are equivalent at first order",
            "A=0<B": "unique minimizer w=0",
            "A>0=B": "unique minimizer w=1",
            "A,B>0": "unique interior minimizer A/(A+B)",
        },
    }
    output = Path(__file__).with_name("flat_r0_r1_results.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20_000)
    args = parser.parse_args()
    main(args.reps)
