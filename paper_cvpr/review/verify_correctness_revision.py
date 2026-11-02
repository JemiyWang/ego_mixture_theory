"""Targeted algebra checks for the 2026-11-01 correctness revision.

High-precision directional checks are diagnostics, not a proof of uniform
expected remainders. The proof is in sec/X_suppl.tex.
"""
import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "M1_T1定理"))
from verify_flat_r0_r1 import check_algebra, matched_coefficients
from verify_full_ridge_second_order import risk_coefficients


def interpolation_checks():
    mp.mp.dps = 65
    sr = mp.diag([1, mp.mpf("0.5")])
    sh = mp.diag([1, mp.mpf("0.0625")])
    zr = mp.matrix([["0.3", "0.2"], ["0.2", "-0.1"]])
    zh = mp.matrix([["-0.1", "0.1"], ["0.1", "0.2"]])
    beta = mp.matrix(["0.8", "-0.3"])
    delta = mp.matrix(["0.1", "-0.4"])
    mu = mp.mpf("0.2")
    trace = lambda a: sum(a[i, i] for i in range(a.rows))

    def direct(w, t, kind):
        gr, gh = sr + t * zr, sh + t * zh
        b = ((1 - w) * gr + w * gh + mu * mp.eye(2)) ** -1
        if kind == "bias":
            q = -mu * beta + w * gh * delta
            return (q.T * b * sr * b * q)[0]
        noise = (1 - w) ** 2 * gr + 2 * w ** 2 * gh
        return trace(sr * b * noise * b)

    def taylor(w, t, kind):
        r = ((1 - w) * sr + w * sh + mu * mp.eye(2)) ** -1
        z = (1 - w) * zr + w * zh
        s = r * sr * r
        if kind == "bias":
            v = -mu * beta + w * sh * delta
            xi = w * zh * delta
            h1 = -r * z * s - s * z * r
            h2 = r * z * r * z * s + s * z * r * z * r + r * z * s * z * r
            return ((v.T * s * v)[0]
                    + t * (2 * xi.T * s * v + v.T * h1 * v)[0]
                    + t ** 2 * (v.T * h2 * v + 2 * v.T * h1 * xi + xi.T * s * xi)[0])
        w0 = (1 - w) ** 2 * sr + 2 * w ** 2 * sh
        w1 = (1 - w) ** 2 * zr + 2 * w ** 2 * zh
        b1 = -r * z * r
        return trace(sr * r * w0 * r) + t * trace(sr * (b1 * w0 * r + r * w1 * r + r * w0 * b1))

    weights = [mp.mpf(x) for x in ["0", "0.1", "0.3", "0.7", "1"]]
    max_coefficient_error = mp.mpf(0)
    for kind, order in [("bias", 2), ("noise", 1)]:
        for w in weights:
            for j in range(order + 1):
                lhs = mp.diff(lambda t: direct(w, t, kind), 0, j)
                rhs = mp.diff(lambda t: taylor(w, t, kind), 0, j)
                max_coefficient_error = max(max_coefficient_error, abs(lhs - rhs))
    assert max_coefficient_error < mp.mpf("1e-50")

    rows = []
    for kind, remainder_order in [("bias", 3), ("noise", 2)]:
        for k in range(3):
            values = []
            for divisor in [8, 16, 32, 64, 128]:
                eps = mp.mpf(1) / divisor
                error = max(abs(mp.diff(
                    lambda w: direct(w, eps, kind) - taylor(w, eps, kind),
                    w, k)) for w in weights)
                values.append({"epsilon": float(eps), "max_sampled_error": float(error),
                               "scaled_error": float(error / eps ** remainder_order)})
            empirical_order = np.log2(values[-2]["max_sampled_error"] / values[-1]["max_sampled_error"])
            assert abs(empirical_order - remainder_order) < 0.08
            rows.append({"part": kind, "w_derivative_order": k,
                         "expected_directional_order": remainder_order,
                         "observed_last_halving_order": float(empirical_order),
                         "values": values})
    return {"coefficient_error": float(max_coefficient_error),
            "scope": "Two-coordinate noncommuting perturbations, five weights including endpoints; numerical diagnostics only.",
            "remainder_checks": rows}


def zero_risk_check():
    rng = np.random.default_rng(20261101)
    max_risk = 0.0
    count = 0
    for p in [1, 2, 5]:
        a = np.arange(1, p + 1, dtype=float) ** -1
        beta = delta = np.zeros(p)
        ac, bc, _, _ = matched_coefficients(a, beta, .2, 0., 0., 2.)
        assert ac == bc == 0
        for n in [2, 10, 100]:
            xr = rng.normal(size=(n, p)) * np.sqrt(a)
            xh = rng.normal(size=(2 * n, p)) * np.sqrt(a)
            for w in [0., .2, .5, 1.]:
                gram = (1 - w) * xr.T @ xr / n + w * xh.T @ xh / (2 * n)
                fitted = np.linalg.solve(gram + .2 * np.eye(p), np.zeros(p))
                actual_risk = a @ fitted ** 2
                r0, r1, _ = risk_coefficients(w, np.diag(a), np.diag(a), beta,
                                              delta, .2, 0., 0., 2.)
                assert actual_risk == r0 == r1 == 0
                max_risk = max(max_risk, float(actual_risk))
                count += 1
    return {"design_weight_cells": count, "max_actual_risk": max_risk}


if __name__ == "__main__":
    result = {"interpolation": interpolation_checks(),
              "matched_R1_existing_algebra_check": check_algebra(),
              "zero_coefficients": zero_risk_check()}
    output = Path(__file__).with_name("correctness_revision_checks_2026-11-01.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "taylor_coefficient_error": result["interpolation"]["coefficient_error"],
        "directional_remainder_orders": [
            [r["part"], r["w_derivative_order"], r["observed_last_halving_order"]]
            for r in result["interpolation"]["remainder_checks"]],
        "matched_R1_algebra": result["matched_R1_existing_algebra_check"],
        "zero_coefficients": result["zero_coefficients"],
    }, indent=2))
