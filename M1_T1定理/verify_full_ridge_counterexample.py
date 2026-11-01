"""Verify a local-harm/global-benefit full-ridge counterexample."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from verify_full_ridge_boundary import conditional_risk_and_gains
from verify_full_ridge_global_optimum import coefficient_derivatives, conditional_derivative_draws
from verify_full_ridge_second_order import risk_coefficients, summarize_draws


def main():
    sigma_r = np.diag([0.06878172866100844, 3.348404027039282, 0.013486232610484008])
    sigma_h = np.diag([0.5763005499719634, 1.0560136030378924, 12.597717940912908])
    beta = np.array([-1.5063590767700887, 2.6917054532527134, -1.0857110393770384])
    delta = np.array([-0.7271735734168655, -0.8534676714155205, 0.9487393526514577])
    mu, sigma_r2, sigma_h2, rho = 0.040455141206854305, 1.0, 4.0, 2.0
    objective = lambda w: float(risk_coefficients(w, sigma_r, sigma_h, beta, delta,
                                                  mu, sigma_r2, sigma_h2, rho)[0])
    opt = minimize_scalar(objective, bounds=(0, 1), method="bounded",
                          options={"xatol": 1e-13})
    w0 = float(opt.x)
    h = 1e-5
    curvature = (objective(w0 + h) - 2 * objective(w0) + objective(w0 - h)) / h**2
    d0, d1 = coefficient_derivatives(0, sigma_r, sigma_h, beta, delta,
                                     mu, sigma_r2, sigma_h2, rho)
    r00, r10, _ = risk_coefficients(0, sigma_r, sigma_h, beta, delta,
                                    mu, sigma_r2, sigma_h2, rho)
    r0w, r1w, _ = risk_coefficients(w0, sigma_r, sigma_h, beta, delta,
                                    mu, sigma_r2, sigma_h2, rho)
    result = {"w0": w0, "curvature": curvature, "H": d0, "R1_prime": d1,
              "R0_at_zero": float(r00), "R0_at_w0": float(r0w), "cells": []}
    print(f"w0={w0:.9f}, curvature={curvature:.6f}, H={d0:.9f}, "
          f"R(0)={r00:.9f}, R(w0)={r0w:.9f}")
    rng = np.random.default_rng(20261030)
    for n_r in [1000, 4000]:
        derivative_draws = conditional_derivative_draws(
            rng, 20_000, n_r, 0.0, sigma_r, sigma_h, beta, delta,
            mu, sigma_r2, sigma_h2, rho)
        derivative_mean, derivative_se = summarize_draws(derivative_draws)
        baseline, means, gains, gain_ses = conditional_risk_and_gains(
            rng, 50_000, n_r, np.array([0.0, w0]), sigma_r, sigma_h, beta, delta,
            mu, sigma_r2, sigma_h2, rho)
        derivative_prediction = d0 + d1 / n_r
        gain_prediction = float(r00 - r0w + (r10 - r1w) / n_r)
        row = {"n_r": n_r, "derivative_prediction": derivative_prediction,
               "derivative_mc": derivative_mean, "derivative_se": derivative_se,
               "gain_prediction": gain_prediction, "gain_mc": float(gains[1]),
               "gain_se": float(gain_ses[1]), "baseline_risk": baseline,
               "mixture_risk": float(means[1])}
        result["cells"].append(row)
        print(f"n={n_r}: derivative pred={derivative_prediction:.6f} MC={derivative_mean:.6f}; "
              f"gain pred={gain_prediction:.6f} MC={gains[1]:.6f} +/- {gain_ses[1]:.1e}")
    target = Path(__file__).with_name("full_ridge_counterexample_verification.json")
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
