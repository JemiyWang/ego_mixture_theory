"""Exact R1 certificates and finite-sample risk at fixed population candidates.

The fixed-candidate comparison does not search for finite-sample local minima
and must not be interpreted as a validation of asymptotic branch selection.
"""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as s

from verify_flat_r0_r1 import conditional_gain_draws
from verify_full_ridge_global_optimum import coefficient_derivatives, summarize_draws
from verify_full_ridge_second_order import risk_coefficients
from verify_global_risk_shape import (
    interval_add,
    interval_div,
    interval_mul,
    interval_sub,
    scalar_interval,
)


def interval_eval(expr, x, w):
    if expr.is_Number:
        return scalar_interval(Fraction(expr))
    if expr == w:
        return x
    if expr.is_Add:
        total = scalar_interval(Fraction(0))
        for arg in expr.args:
            total = interval_add(total, interval_eval(arg, x, w))
        return total
    if expr.is_Mul:
        total = scalar_interval(Fraction(1))
        for arg in expr.args:
            total = interval_mul(total, interval_eval(arg, x, w))
        return total
    if expr.is_Pow and expr.exp.is_Integer and expr.exp >= 0:
        total = scalar_interval(Fraction(1))
        for _ in range(int(expr.exp)):
            total = interval_mul(total, interval_eval(expr.base, x, w))
        return total
    raise ValueError(f"Unsupported interval expression: {expr}")


def risk_interval(risk, interval, w):
    numerator, denominator = s.fraction(risk)
    num = interval_eval(numerator, interval, w)
    den = interval_eval(denominator, interval, w)
    return interval_div(num, den)


def matched_mismatch_example():
    w = s.symbols("w", real=True)
    a = [s.Integer(20), s.Rational(1, 2)]
    sigma = s.diag(*a)
    beta = s.Matrix([s.Rational(4, 5), 0])
    delta = s.Matrix([s.Rational(3, 10), s.Rational(-1, 5)])
    mu = s.Rational(2, 5)
    sr2, sh2, rho = s.Rational(1, 10), s.Integer(4), s.Integer(3)
    r = s.diag(*[1 / (a[j] + mu) for j in range(2)])
    sm = r * sigma * r
    v = -mu * beta + w * sigma * delta
    t = lambda n: sigma * n * sigma + s.trace(sigma * n) * sigma
    u = t(r)
    correction = r * u * sm + sm * u * r + r * t(sm) * r
    q = (1 - w)**2 + w**2 / rho
    risk = (q * (v.T * correction * v)[0]
            + w**2 / rho * (sm * t(delta * delta.T)).trace()
            - 2 * w**2 / rho * (v.T * (r * t(sm) + sm * t(r)) * delta)[0]
            + (1 - w)**2 * sr2 * (sm * sigma).trace()
            + w**2 / rho * sh2 * (sm * sigma).trace())
    return w, s.expand(risk), (np.array(a, dtype=float), np.array(a, dtype=float),
           np.array(beta, dtype=float).reshape(-1), np.array(delta, dtype=float).reshape(-1),
           float(mu), float(sr2), float(sh2), float(rho))


def general_zero_mismatch_example():
    w = s.symbols("w", real=True)
    a = [s.Rational(1, 5), s.Rational(1, 5)]
    b = [s.Rational(1, 25), s.Integer(4)]
    sr = s.diag(*a)
    sh = s.diag(*b)
    beta = s.Matrix([s.Rational(-2, 5), s.Rational(-7, 10)])
    mu = s.Rational(11, 20)
    sr2, sh2, rho = s.Rational(1, 50), s.Rational(1, 10), s.Integer(1)
    d = s.diag(*[(1 - w) * a[j] + w * b[j] + mu for j in range(2)])
    r = d.inv()
    sm = r * sr * r
    v = -mu * beta
    tr = lambda n: sr * n * sr + s.trace(sr * n) * sr
    th = lambda n: sh * n * sh + s.trace(sh * n) * sh
    u = (1 - w)**2 * tr(r) + w**2 / rho * th(r)
    ts = (1 - w)**2 * tr(sm) + w**2 / rho * th(sm)
    correction = r * u * sm + sm * u * r + r * ts * r
    risk = ((v.T * correction * v)[0]
            + (1 - w)**2 * sr2 * (sm * sr).trace()
            + w**2 / rho * sh2 * (sm * sh).trace())
    return w, s.factor(risk), (np.array(a, dtype=float), np.array(b, dtype=float),
           np.array(beta, dtype=float).reshape(-1), np.zeros(2), float(mu),
           float(sr2), float(sh2), float(rho))


def certify_risk_shape(name, w, risk, config, global_index):
    derivative_numerator = s.factor(s.diff(risk, w))
    numerator, _ = s.fraction(derivative_numerator)
    isolating = s.intervals(numerator, eps=s.Rational(1, 10**8))
    inside = [(interval, multiplicity) for interval, multiplicity in isolating
              if interval[0] < 1 and interval[1] > 0]
    if len(inside) != 3 or any(multiplicity != 1 for _, multiplicity in inside):
        raise AssertionError(f"{name} must have three simple interior roots")
    exact_roots = [root for root in s.real_roots(numerator) if 0 < root < 1]
    candidates = [("boundary", None)] + [("stationary", interval) for interval, _ in inside] + [("boundary", None)]
    classifications = ["local_minimum", "local_maximum", "local_minimum"]
    rows = []
    stationary_index = 0
    for kind, interval in candidates:
        if interval is None:
            endpoint = 0 if not rows else 1
            bounds = risk_interval(risk, (endpoint, endpoint), w)
            root = s.Integer(endpoint)
        else:
            bounds = risk_interval(risk, interval, w)
            root = exact_roots[stationary_index]
            stationary_index += 1
        rows.append({
            "kind": kind,
            "classification": "boundary" if kind == "boundary" else classifications[stationary_index - 1],
            "root": str(root), "root_numeric": float(s.N(root, 30)),
            "isolating_interval": None if interval is None else [str(interval[0]), str(interval[1])],
            "risk_interval_exact": [str(bounds[0]), str(bounds[1])],
            "risk_interval": [float(bounds[0]), float(bounds[1])],
        })
    target = rows[global_index]
    for i, row in enumerate(rows):
        if i != global_index and not target["risk_interval"][1] < row["risk_interval"][0]:
            raise AssertionError(f"{name} global-risk ordering failed")
    a, b, beta, delta, mu, sr2, sh2, rho = config
    max_numeric_error = 0.0
    for x in np.linspace(0, 1, 21):
        numeric = float(risk_coefficients(float(x), np.diag(a), np.diag(b), beta,
                                          delta, mu, sr2, sh2, rho)[1])
        symbolic = float(risk.subs(w, x))
        max_numeric_error = max(max_numeric_error, abs(numeric - symbolic))
    if max_numeric_error > 1e-10:
        raise AssertionError(f"{name} symbolic and independent numerical risks disagree")
    return {
        "risk": str(risk),
        "derivative_numerator_degree": int(s.degree(numerator, w)),
        "all_root_intervals": [[str(interval[0]), str(interval[1])] for interval, _ in isolating],
        "interior_root_count": len(inside),
        "candidates": rows,
        "unique_global_minimizer": target["root"],
        "unique_global_minimizer_numeric": target["root_numeric"],
        "max_symbolic_vs_numeric_error": max_numeric_error,
        "conclusion": "R1 has two local minima and one local maximum in (0,1)",
    }


def fixed_candidate_risk_check(reps):
    """Compare two fixed population minimizers without reoptimizing either."""
    t = 1.0608592591022064
    a = np.array([1.0, 0.5])
    b = np.array([1.0, 1 / 16])
    mu = 0.007
    u1 = np.sqrt(26) / 1000
    u2 = np.sqrt(3) / 500
    c = np.array([0.02, 0.95])
    beta = np.array([u1 * c[0] / mu, np.sqrt(t) * u2 * c[1] / mu])
    delta = np.array([u1 / b[0], np.sqrt(t) * u2 / b[1]])
    sr2, sh2, rho = 1.0, 4.0, 2.0
    config = (a, b, beta, delta, mu, sr2, sh2, rho)
    w1 = 0.29550604935924824
    w2 = 0.883351093497898
    sr, sh = np.diag(a), np.diag(b)
    r1_first = float(risk_coefficients(w1, sr, sh, beta, delta, mu, sr2, sh2, rho)[1])
    r1_second = float(risk_coefficients(w2, sr, sh, beta, delta, mu, sr2, sh2, rho)[1])
    if not r1_first < r1_second:
        raise AssertionError("The first fixed candidate must have smaller R1")
    n_r = 1000
    rng = np.random.default_rng(20261101)
    draws = conditional_gain_draws(rng, reps, n_r, [w1, w2], config)
    fixed_gap_draws = draws[:, 1] - draws[:, 0]
    mean, se = summarize_draws(fixed_gap_draws)
    prediction = (r1_second - r1_first) / n_r
    _, r1_prime = coefficient_derivatives(w1, sr, sh, beta, delta,
                                          mu, sr2, sh2, rho)
    h = 1e-4
    objective = lambda x: float(risk_coefficients(x, sr, sh, beta, delta,
                                                   mu, sr2, sh2, rho)[0])
    curvature = (objective(w1 + h) - 2 * objective(w1) + objective(w1 - h)) / h**2
    return {
        "weights": [w1, w2],
        "n_r": n_r,
        "replicates": reps,
        "seed": 20261101,
        "R1_first_candidate": r1_first,
        "R1_second_candidate": r1_second,
        "predicted_fixed_candidate_gap": prediction,
        "conditional_mc_fixed_candidate_gap": mean,
        "standard_error": se,
        "error_in_standard_errors": (mean - prediction) / se,
        "R0_first_candidate_curvature": float(curvature),
        "R1_first_candidate_derivative": r1_prime,
        "local_weight_correction_coefficient": float(-r1_prime / curvature),
        "formal_first_order_displacement_at_n_1000": float(-r1_prime / curvature / n_r),
        "finite_sample_local_minima_sought": False,
        "scope": "Risk ordering at two fixed population weights; no branch-persistence or optimized-branch-gap validation.",
        "interpretation": "The displacement is outside [0,1]; n_r=1000 is not a small-displacement check of the asymptotic branch-selection theorem.",
    }


def main(reps):
    matched = certify_risk_shape("matched", *matched_mismatch_example(), global_index=1)
    general = certify_risk_shape("general", *general_zero_mismatch_example(), global_index=3)
    result = {
        "matched_spectra_general_mismatch_R1": matched,
        "general_spectra_zero_mismatch_R1": general,
        "fixed_population_candidate_risk_comparison": fixed_candidate_risk_check(reps),
    }
    output = Path(__file__).with_name("r1_rn_global_shapes_results.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "matched_global": matched["unique_global_minimizer_numeric"],
        "general_global": general["unique_global_minimizer_numeric"],
        "fixed_candidate_gap": result["fixed_population_candidate_risk_comparison"]["conditional_mc_fixed_candidate_gap"],
    }, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20_000)
    args = parser.parse_args()
    main(args.reps)
