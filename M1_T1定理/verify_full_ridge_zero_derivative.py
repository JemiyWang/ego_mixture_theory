"""Fixed-p full-ridge zero-mixing derivative: algebra and Monte-Carlo checks."""
import argparse
import json
from pathlib import Path

import numpy as np

B_TOKEN = "B"
G_TOKEN = "G"
Z_TOKEN = "Z"


def matprod(factors, p):
    out = np.eye(p)
    for factor in factors:
        out = out @ factor
    return out


def wishart_cov(middle, sigma):
    return sigma @ middle @ sigma + sigma * np.trace(sigma @ middle)


def derivative_coefficients(sigma_r, sigma_h, beta, delta, mu, sigma_r2):
    p = len(beta)
    r = np.linalg.inv(sigma_r + mu * np.eye(p))
    s = r @ sigma_r @ r
    m = sigma_h - sigma_r
    t = lambda x: wishart_cov(x, sigma_r)
    u = t(r)

    c = r @ u @ s + s @ u @ r + r @ t(s) @ r

    def triple(n):
        return (
            r @ u @ s @ n @ r
            + s @ u @ r @ n @ r
            + s @ n @ r @ u @ r
            + r @ t(s) @ r @ n @ r
            + r @ t(s @ n @ r) @ r
            + s @ t(r @ n @ r) @ r
        )

    ell = triple(m) + r @ t(s) @ r + 2 * s @ u @ r
    h0 = -2 * mu * (sigma_h @ delta) @ s @ beta - 2 * mu**2 * beta @ s @ m @ r @ beta
    h1 = -2 * mu * (sigma_h @ delta) @ c @ beta - 2 * mu**2 * beta @ ell @ beta
    k0 = 2 * sigma_r2 * np.trace(sigma_r @ (r @ sigma_r @ r + r @ sigma_r @ r @ m @ r))
    return {
        "H": float(h0),
        "H1": float(h1),
        "K_population_derivative": float(k0),
        "K_full": float(k0 - h1),
    }


def expand_factors(factors, r, sigma):
    terms = [([], 1.0)]
    for factor in factors:
        if isinstance(factor, str) and factor == B_TOKEN:
            replacements = [([r], 1), ([r, Z_TOKEN, r], -1),
                            ([r, Z_TOKEN, r, Z_TOKEN, r], 1)]
        elif isinstance(factor, str) and factor == G_TOKEN:
            replacements = [([sigma], 1), ([Z_TOKEN], 1)]
        else:
            replacements = [([factor], 1)]
        terms = [(seq + rep, coeff * sign)
                 for seq, coeff in terms for rep, sign in replacements]
    return terms


def paired_covariance(prefix, middle, suffix, sigma):
    p = len(sigma)
    return (matprod(prefix, p) @ wishart_cov(matprod(middle, p), sigma)
            @ matprod(suffix, p))


def brute_moment(factors, d, sigma):
    r = np.linalg.inv(d)
    zeroth = np.zeros_like(d)
    first = np.zeros_like(d)
    for seq, coeff in expand_factors(factors, r, sigma):
        z_positions = [i for i, value in enumerate(seq)
                       if isinstance(value, str) and value == Z_TOKEN]
        if len(z_positions) == 0:
            zeroth += coeff * matprod(seq, len(sigma))
        elif len(z_positions) == 2:
            i, j = z_positions
            first += coeff * paired_covariance(seq[:i], seq[i + 1:j], seq[j + 1:], sigma)
    return zeroth, first


def brute_bias_coefficients(sigma_r, sigma_h, beta, delta, mu):
    d = sigma_r + mu * np.eye(len(beta))
    c0, c0_1 = brute_moment([B_TOKEN, sigma_r, B_TOKEN], d, sigma_r)
    ch, ch_1 = brute_moment([B_TOKEN, sigma_r, B_TOKEN, sigma_h, B_TOKEN], d, sigma_r)
    cg, cg_1 = brute_moment([B_TOKEN, sigma_r, B_TOKEN, G_TOKEN, B_TOKEN], d, sigma_r)
    h0 = -2 * mu * (sigma_h @ delta) @ c0 @ beta - 2 * mu**2 * beta @ (ch - cg) @ beta
    h1 = -2 * mu * (sigma_h @ delta) @ c0_1 @ beta - 2 * mu**2 * beta @ (ch_1 - cg_1) @ beta
    return float(h0), float(h1)


def sample_gram(rng, n, eigenvalues):
    """Bartlett decomposition of Sigma^(1/2) Wishart_p(n,I) Sigma^(1/2) / n."""
    p = len(eigenvalues)
    lower = np.zeros((p, p))
    for i in range(p):
        lower[i, i] = np.sqrt(rng.chisquare(n - i))
        if i + 1 < p:
            lower[i + 1:, i] = rng.normal(size=p - i - 1)
    wishart = lower @ lower.T / n
    root = np.diag(np.sqrt(eigenvalues))
    return root @ wishart @ root


def conditional_derivative_draws(rng, reps, n_r, sigma_r, sigma_h, beta, delta, mu, sigma_r2):
    """Exact design-level derivative draws, with the independent human Gram averaged out."""
    p = len(beta)
    eigenvalues = np.diag(sigma_r)
    if not np.allclose(sigma_r, np.diag(eigenvalues)):
        raise ValueError("The validation generator expects a diagonal target covariance.")
    eye = np.eye(p)
    out = np.empty(reps)
    for rep in range(reps):
        gram = sample_gram(rng, n_r, eigenvalues)
        resolvent = np.linalg.inv(gram + mu * eye)
        c = resolvent @ sigma_r @ resolvent
        gap = sigma_h - gram
        bias = (-2 * mu * (sigma_h @ delta) @ c @ beta
                - 2 * mu**2 * beta @ c @ gap @ resolvent @ beta)
        noise_factor = -resolvent @ gram @ resolvent - resolvent @ gap @ resolvent @ gram @ resolvent
        noise = 2 * sigma_r2 / n_r * np.trace(sigma_r @ noise_factor)
        out[rep] = bias + noise
    return out


def configurations():
    p1 = (np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([0.2]), 0.1, 1.0)
    p2 = (np.array([1.0, 0.5]), np.array([1.2, 0.7]),
          np.array([0.8, -0.3]), np.array([0.1, -0.4]), 0.2, 1.0)
    p = 20
    j = np.arange(1, p + 1, dtype=float)
    a, b = j**-2.0, j**-0.9
    beta = j**-1.5
    beta /= np.sqrt(np.sum(j**2 * beta**2))
    delta = -j**-1.0
    delta *= 0.5 / np.linalg.norm(delta)
    p20 = (a, b, beta, delta, 0.05, 1.0)
    return {
        "p1_scalar": (p1, [1000, 10000]),
        "p2_nonsingular": (p2, [1000, 10000]),
        "p20_spectral_A_anti_aligned": (p20, [1000, 4000, 16000]),
    }


def main(reps, output):
    rng = np.random.default_rng(20261028)
    result = {"model": "R_full'(0+) = H - K_full/n_r + o(1/n_r)",
              "replicates_per_cell": reps, "cells": []}
    max_algebra_error = 0.0
    for name, (config, sample_sizes) in configurations().items():
        a, b, beta, delta, mu, sigma_r2 = config
        sigma_r, sigma_h = np.diag(a), np.diag(b)
        coeff = derivative_coefficients(sigma_r, sigma_h, beta, delta, mu, sigma_r2)
        brute_h0, brute_h1 = brute_bias_coefficients(sigma_r, sigma_h, beta, delta, mu)
        algebra_error = max(abs(coeff["H"] - brute_h0), abs(coeff["H1"] - brute_h1))
        max_algebra_error = max(max_algebra_error, algebra_error)
        if algebra_error > 1e-10:
            raise AssertionError(f"Compact/brute expansion mismatch in {name}: {algebra_error}")
        for n_r in sample_sizes:
            draws = conditional_derivative_draws(rng, reps, n_r, sigma_r, sigma_h,
                                                 beta, delta, mu, sigma_r2)
            prediction = coeff["H"] - coeff["K_full"] / n_r
            se = float(draws.std(ddof=1) / np.sqrt(reps))
            cell = {
                "configuration": name, "n_r": n_r, **coeff,
                "prediction": float(prediction), "mc_mean": float(draws.mean()),
                "mc_standard_error": se,
                "mc_minus_prediction": float(draws.mean() - prediction),
                "error_in_standard_errors": float((draws.mean() - prediction) / se),
                "algebra_compact_vs_brute": float(algebra_error),
            }
            result["cells"].append(cell)
            print(f"{name:30s} n_r={n_r:5d} pred={prediction:+.9f} "
                  f"MC={draws.mean():+.9f} +/- {se:.2e} "
                  f"diff={draws.mean() - prediction:+.2e}")
    result["max_algebra_compact_vs_brute"] = float(max_algebra_error)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"max compact/brute algebra error: {max_algebra_error:.3e}")
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("full_ridge_zero_derivative_results.json"))
    args = parser.parse_args()
    main(args.reps, args.output)
