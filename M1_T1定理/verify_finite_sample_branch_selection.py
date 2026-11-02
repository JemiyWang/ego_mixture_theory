"""Finite-sample double-valley and branch-selection experiment for the full-ridge model."""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import scipy
from scipy.optimize import brentq

from verify_full_ridge_second_order import risk_coefficients


ROOT_SEED = 20261102
N_VALUES = [1000, 3000, 10000, 30000, 100000, 300000, 1000000,
            3000000, 10000000, 30000000, 100000000]
FIND_BATCHES = 32
FIND_BATCH_SIZE = 4096
EVAL_BATCHES = 32
EVAL_BATCH_SIZE = 8192
BOOTSTRAP_REPS = 1000
GRID_FINE_SIZE = 4001
GRID_COARSE_SIZE = 2001
MAX_MULTIPLIER = 8
WEIGHT_CHUNK = 128
Z95 = 1.959963984540054
T_INITIAL = 1.0608592591022064
W1_INITIAL = 0.29550604935924824
W2_INITIAL = 0.883351093497898


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def interval(mean, standard_error):
    return [finite(mean - Z95 * standard_error), finite(mean + Z95 * standard_error)]


def batch_mean_se(values):
    values = np.asarray(values, dtype=float)
    count = values.shape[0]
    mean = np.mean(values, axis=0)
    if count < 2:
        return mean, np.zeros_like(mean)
    se = np.std(values, axis=0, ddof=1) / np.sqrt(count)
    return mean, se


def runner_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def protocol_document(code_hash):
    return {
        "protocol_version": 1,
        "experiment": "finite_sample_branch_selection_2026-11-02",
        "runner_sha256": code_hash,
        "configuration": {
            "p": 2,
            "sigma_r": [1.0, 0.5],
            "sigma_h": [1.0, 0.0625],
            "mu": 0.007,
            "rho": 2.0,
            "sigma_r2": 1.0,
            "sigma_h2": 4.0,
            "u1": "sqrt(26)/1000",
            "u2": "sqrt(3)/500",
            "c": [0.02, 0.95],
            "t_initial": T_INITIAL,
            "w1_initial": W1_INITIAL,
            "w2_initial": W2_INITIAL,
            "population_digits": 80,
        },
        "n_r_values": N_VALUES,
        "n_h_rule": "n_h = 2 n_r",
        "seed_rule": "SeedSequence([20261102, n_r, stage, batch_id]); stages 0=find, 1=independent evaluation, 2=bootstrap",
        "root_seed": ROOT_SEED,
        "initial_budget": {
            "find_batches": FIND_BATCHES,
            "find_batch_size": FIND_BATCH_SIZE,
            "find_replicates": FIND_BATCHES * FIND_BATCH_SIZE,
            "evaluation_batches": EVAL_BATCHES,
            "evaluation_batch_size": EVAL_BATCH_SIZE,
            "evaluation_replicates": EVAL_BATCHES * EVAL_BATCH_SIZE,
            "bootstrap_replicates": BOOTSTRAP_REPS,
        },
        "maximum_budget_multiplier": MAX_MULTIPLIER,
        "grid": {
            "coarse_points": GRID_COARSE_SIZE,
            "fine_points": GRID_FINE_SIZE,
            "interval": [0.0, 1.0],
            "root_interpolation": "piecewise linear with bracketed Brent refinement",
            "grid_position_tolerance": 0.001,
            "near_zero_standard_errors": 3.0,
            "local_refinement_points": 17,
            "weight_chunk": WEIGHT_CHUNK,
        },
        "intervals": {
            "confidence": 0.95,
            "normal_quantile": Z95,
            "scope": "pointwise per sample size",
            "bootstrap": "percentile over reoptimized batch-mean curves; positions conditional on two minima",
        },
        "criteria": {
            "stable_double_valley": "two internal minima and at least one internal maximum; positive lower 95% curvature bounds; coarse/fine agreement; at least 95% bootstrap support; no unresolved local refinement",
            "direction_resolved": "independent paired 95% interval, conservatively expanded by the quadratic root-position error, excludes zero",
            "risk_relative_tolerance": 0.10,
            "risk_support": "entire position-expanded 95% interval for n_r Delta_n lies within [0.9 C_Delta, 1.1 C_Delta]",
            "risk_unsupported": "position-expanded 95% interval for n_r Delta_n has no intersection with the tolerance band",
            "risk_unresolved": "all other two-valley cases",
            "bootstrap_indeterminate_band": [0.05, 0.95],
        },
        "escalation": {
            "multipliers": [1, 2, 4, 8],
            "reasons": ["inconsistent or unresolved grid shape", "bootstrap root structure in the indeterminate band",
                        "paired branch direction unresolved", "first-order risk-difference accuracy unresolved"],
            "rule": "double both find-root and independent-evaluation budgets; retain every attempt; stop when resolved or at the 8x cap",
        },
        "risk_definition": "exact conditional-label-noise full-ridge risk averaged over independent exact Bartlett Gram pairs; not R0+R1/n",
        "position_rule": "roots of the Monte Carlo average conditional derivative; never an average of per-dataset argmins",
    }


def freeze_protocol(output, code_hash):
    path = output / "protocol.json"
    document = protocol_document(code_hash)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != document:
            existing_settings = dict(existing)
            new_settings = dict(document)
            existing_settings.pop("runner_sha256")
            new_settings.pop("runner_sha256")
            preflight_only = not (output / "checkpoints").exists() and not (output / "results.json").exists()
            if existing_settings != new_settings or not preflight_only:
                raise RuntimeError("protocol.json differs from the current runner/protocol; use a new output directory")
            amendment_path = output / "protocol_amendments.json"
            amendments = json.loads(amendment_path.read_text(encoding="utf-8")) if amendment_path.exists() else []
            amendments.append({"utc": utc_now(), "previous_runner_sha256": existing["runner_sha256"],
                               "new_runner_sha256": code_hash,
                               "reason": "preflight implementation-check correction before any formal sampling"})
            json_dump(amendment_path, amendments)
            json_dump(path, document)
    else:
        json_dump(path, document)
    return document


def population_risk_mp(w, t):
    w, t = mp.mpf(w), mp.mpf(t)
    a = [mp.mpf(1), mp.mpf(1) / 2]
    b = [mp.mpf(1), mp.mpf(1) / 16]
    c = [mp.mpf("0.02"), mp.mpf("0.95")]
    u = [mp.sqrt(26) / 1000, mp.sqrt(3) / 500]
    mu = mp.mpf("0.007")
    scales = [mp.mpf(1), mp.sqrt(t)]
    return sum(a[j] * (scales[j] * u[j])**2 * (w - c[j])**2 /
               ((1 - w) * a[j] + w * b[j] + mu)**2 for j in range(2))


def poly_mul(left, right):
    out = [mp.mpf(0)] * (len(left) + len(right) - 1)
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            out[i + j] += x * y
    return out


def poly_power(base, exponent):
    out = [mp.mpf(1)]
    for _ in range(exponent):
        out = poly_mul(out, base)
    return out


def population_check():
    mp.mp.dps = 80
    derivative = lambda w, t: mp.diff(lambda x: population_risk_mp(x, t), w)
    equations = lambda w1, w2, t: (derivative(w1, t), derivative(w2, t),
                                   population_risk_mp(w1, t) - population_risk_mp(w2, t))
    w1, w2, t = mp.findroot(equations, (W1_INITIAL, W2_INITIAL, T_INITIAL),
                            tol=mp.mpf("1e-70"), maxsteps=100)
    if not (0 < w1 < w2 < 1):
        raise AssertionError("Population tie solution violates 0 < w1 < w2 < 1")
    a = [mp.mpf(1), mp.mpf(1) / 2]
    b = [mp.mpf(1), mp.mpf(1) / 16]
    c = [mp.mpf("0.02"), mp.mpf("0.95")]
    u = [mp.sqrt(26) / 1000, mp.sqrt(3) / 500]
    mu = mp.mpf("0.007")
    k = [a[0] * u[0]**2, a[1] * t * u[1]**2]
    d = [[a[j] + mu, b[j] - a[j]] for j in range(2)]
    ell = []
    for j in range(2):
        e_at_c = d[j][0] + c[j] * d[j][1]
        ell.append([-2 * k[j] * c[j] * e_at_c, 2 * k[j] * e_at_c])
    polynomial = poly_mul(ell[0], poly_power(d[1], 3))
    second_term = poly_mul(ell[1], poly_power(d[0], 3))
    polynomial = [x + y for x, y in zip(polynomial, second_term)]
    all_roots = mp.polyroots(polynomial[::-1], maxsteps=200, extraprec=100)
    real_roots = []
    for root in all_roots:
        if abs(mp.im(root)) < mp.mpf("1e-60") and 0 < mp.re(root) < 1:
            real_roots.append(mp.findroot(lambda x: derivative(x, t), mp.re(root)))
    real_roots = sorted(real_roots)
    if len(real_roots) != 3:
        raise AssertionError(f"Expected three interior population roots, found {len(real_roots)}")
    if min(abs(real_roots[0] - w1), abs(real_roots[2] - w2)) > mp.mpf("1e-60"):
        raise AssertionError("Joint tie solution is inconsistent with stationary roots")
    stationary = []
    for root in real_roots:
        second = mp.diff(lambda x: derivative(x, t), root)
        stationary.append({
            "w": mp.nstr(root, 70),
            "risk": mp.nstr(population_risk_mp(root, t), 70),
            "first_derivative_residual": mp.nstr(derivative(root, t), 12),
            "second_derivative": mp.nstr(second, 60),
            "classification": "local_minimum" if second > 0 else "local_maximum" if second < 0 else "degenerate",
        })
    endpoint_risks = [population_risk_mp(0, t), population_risk_mp(1, t)]
    minimum_risk = population_risk_mp(w1, t)
    beta = [u[0] * c[0] / mu, mp.sqrt(t) * u[1] * c[1] / mu]
    ellipsoid = beta[0]**2 + 4 * beta[1]**2
    model = {
        "a": np.array([1.0, 0.5]),
        "b": np.array([1.0, 1 / 16]),
        "mu": 0.007,
        "rho": 2.0,
        "sr2": 1.0,
        "sh2": 4.0,
        "t": float(t),
        "beta": np.array([float(beta[0]), float(beta[1])]),
        "delta": np.array([float(u[0]), float(mp.sqrt(t) * u[1] / b[1])]),
    }
    sr, sh = np.diag(model["a"]), np.diag(model["b"])
    w1_float, w2_float = float(w1), float(w2)
    r0_1, r1_1, _ = risk_coefficients(w1_float, sr, sh, model["beta"], model["delta"],
                                      model["mu"], model["sr2"], model["sh2"], model["rho"])
    r0_2, r1_2, _ = risk_coefficients(w2_float, sr, sh, model["beta"], model["delta"],
                                      model["mu"], model["sr2"], model["sh2"], model["rho"])
    c_delta = float(r1_2 - r1_1)
    shifts = []
    for w in [w1_float, w2_float]:
        h = 1e-25
        r0_complex, r1_complex, _ = risk_coefficients(w + 1j * h, sr, sh, model["beta"], model["delta"],
                                                      model["mu"], model["sr2"], model["sh2"], model["rho"])
        r0_prime = float(np.imag(r0_complex) / h)
        r1_prime = float(np.imag(r1_complex) / h)
        curvature = float(mp.diff(lambda x: derivative(x, t), mp.mpf(w)))
        shifts.append(float(-r1_prime / curvature))
    float_tie_residual = float(r0_2 - r0_1)
    required = 1e-4 * abs(c_delta) / max(N_VALUES)
    predictions = []
    for n in N_VALUES:
        predictions.append({
            "n_r": n,
            "weights": [w1_float + shifts[0] / n, w2_float + shifts[1] / n],
            "available": [0 <= w1_float + shifts[0] / n <= 1,
                          0 <= w2_float + shifts[1] / n <= 1],
        })
    document = {
        "precision_digits": mp.mp.dps,
        "t": mp.nstr(t, 70),
        "w1": mp.nstr(w1, 70),
        "w2": mp.nstr(w2, 70),
        "joint_stationary_residuals": [mp.nstr(derivative(w1, t), 12), mp.nstr(derivative(w2, t), 12)],
        "equal_risk_residual_high_precision": mp.nstr(population_risk_mp(w1, t) - population_risk_mp(w2, t), 12),
        "float_equal_risk_residual": float_tie_residual,
        "float_residual_required_below": required,
        "float_precision_check_passed": abs(float_tie_residual) < required,
        "all_stationary_points": stationary,
        "all_polynomial_roots": [mp.nstr(root, 50) for root in all_roots],
        "endpoint_risks": [mp.nstr(value, 70) for value in endpoint_risks],
        "endpoint_minus_minimum": [mp.nstr(value - minimum_risk, 60) for value in endpoint_risks],
        "intermediate_maximum_w": mp.nstr(real_roots[1], 70),
        "two_equal_global_minima": (stationary[0]["classification"] == "local_minimum" and
                                    stationary[2]["classification"] == "local_minimum" and
                                    stationary[1]["classification"] == "local_maximum" and
                                    all(value > minimum_risk for value in endpoint_risks) and
                                    population_risk_mp(real_roots[1], t) > minimum_risk),
        "ellipsoid_beta1_squared_plus_4_beta2_squared": mp.nstr(ellipsoid, 60),
        "ellipsoid_constraint_passed": bool(ellipsoid <= 1),
        "R1_w1": float(r1_1),
        "R1_w2": float(r1_2),
        "C_Delta": c_delta,
        "position_shift_q": shifts,
        "position_predictions": predictions,
        "model_float": {key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in model.items()},
    }
    if not document["float_precision_check_passed"]:
        raise AssertionError("Float population tie residual exceeds the protocol threshold")
    if not document["ellipsoid_constraint_passed"] or not document["two_equal_global_minima"]:
        raise AssertionError("Population assumptions or equal-global-minimum check failed")
    if not c_delta > 0:
        raise AssertionError("The first branch must have smaller R1")
    return model, document


def independent_coefficients(w, model):
    a, b = model["a"], model["b"]
    sr, sh = np.diag(a), np.diag(b)
    beta, delta, mu, rho = model["beta"], model["delta"], model["mu"], model["rho"]
    p = len(a)
    resolvent = np.linalg.inv((1 - w) * sr + w * sh + mu * np.eye(p))
    s_matrix = resolvent @ sr @ resolvent
    v = -mu * beta + w * sh @ delta

    def wishart_second(middle, sigma):
        return sigma @ middle @ sigma + sigma * np.trace(sigma @ middle)

    def combined(middle):
        return ((1 - w)**2 * wishart_second(middle, sr) +
                w**2 / rho * wishart_second(middle, sh))

    u_matrix = combined(resolvent)
    correction = (resolvent @ u_matrix @ s_matrix + s_matrix @ u_matrix @ resolvent +
                  resolvent @ combined(s_matrix) @ resolvent)
    r0 = v @ s_matrix @ v
    mismatch_square = w**2 / rho * np.trace(s_matrix @ wishart_second(np.outer(delta, delta), sh))
    mismatch_cross = -2 * w**2 / rho * v @ (
        resolvent @ wishart_second(s_matrix, sh) + s_matrix @ wishart_second(resolvent, sh)) @ delta
    noise = ((1 - w)**2 * model["sr2"] * np.trace(s_matrix @ sr) +
             w**2 / rho * model["sh2"] * np.trace(s_matrix @ sh))
    r1 = v @ correction @ v + mismatch_square + mismatch_cross + noise
    return float(np.real(r0)), float(np.real(r1))


def sample_gram_pair(rng, reps, n_r, model):
    n_h = int(model["rho"] * n_r)

    def sample(n, eigenvalues):
        l00 = np.sqrt(rng.chisquare(n, size=reps))
        l10 = rng.standard_normal(reps)
        l11 = np.sqrt(rng.chisquare(n - 1, size=reps))
        g00 = l00 * l00 / n
        g01 = l00 * l10 / n
        g11 = (l10 * l10 + l11 * l11) / n
        return (eigenvalues[0] * g00,
                math.sqrt(eigenvalues[0] * eigenvalues[1]) * g01,
                eigenvalues[1] * g11)

    return sample(n_r, model["a"]), sample(n_h, model["b"])


def negative_akb(a, k, b):
    a00, a01, a11 = a
    k00, k01, k11 = k
    b00, b01, b11 = b
    t00 = k00 * b00 + k01 * b01
    t01 = k00 * b01 + k01 * b11
    t10 = k01 * b00 + k11 * b01
    t11 = k01 * b01 + k11 * b11
    return (-(a00 * t00 + a01 * t10),
            -(a00 * t01 + a01 * t11),
            -(a01 * t01 + a11 * t11))


def conditional_metrics(grams_r, grams_h, weights, n_r, model, second=True):
    r00, r01, r11 = [np.asarray(x)[:, None] for x in grams_r]
    h00, h01, h11 = [np.asarray(x)[:, None] for x in grams_h]
    w = np.asarray(weights)[None, :]
    n_h = int(model["rho"] * n_r)
    mu, beta, delta = model["mu"], model["beta"], model["delta"]
    a0, a1 = model["a"]
    m00 = (1 - w) * r00 + w * h00 + mu
    m01 = (1 - w) * r01 + w * h01
    m11 = (1 - w) * r11 + w * h11 + mu
    determinant = m00 * m11 - m01 * m01
    b00, b01, b11 = m11 / determinant, -m01 / determinant, m00 / determinant
    b_matrix = (b00, b01, b11)
    k_matrix = (h00 - r00, h01 - r01, h11 - r11)
    bp00, bp01, bp11 = negative_akb(b_matrix, k_matrix, b_matrix)
    q0 = -mu * beta[0] + w * (h00 * delta[0] + h01 * delta[1])
    q1 = -mu * beta[1] + w * (h01 * delta[0] + h11 * delta[1])
    qp0 = h00 * delta[0] + h01 * delta[1]
    qp1 = h01 * delta[0] + h11 * delta[1]
    c00 = a0 * b00**2 + a1 * b01**2
    c01 = b01 * (a0 * b00 + a1 * b11)
    c11 = a0 * b01**2 + a1 * b11**2
    cp00 = 2 * a0 * b00 * bp00 + 2 * a1 * b01 * bp01
    cp01 = bp01 * (a0 * b00 + a1 * b11) + b01 * (a0 * bp00 + a1 * bp11)
    cp11 = 2 * a0 * b01 * bp01 + 2 * a1 * b11 * bp11
    cr = (1 - w)**2 * model["sr2"] / n_r
    ch = w**2 * model["sh2"] / n_h
    crp = -2 * (1 - w) * model["sr2"] / n_r
    chp = 2 * w * model["sh2"] / n_h
    w00, w01, w11 = cr * r00 + ch * h00, cr * r01 + ch * h01, cr * r11 + ch * h11
    wp00 = crp * r00 + chp * h00
    wp01 = crp * r01 + chp * h01
    wp11 = crp * r11 + chp * h11

    def quadratic(x00, x01, x11, y0, y1):
        return x00 * y0**2 + 2 * x01 * y0 * y1 + x11 * y1**2

    def trace_product(x00, x01, x11, y00, y01, y11):
        return x00 * y00 + 2 * x01 * y01 + x11 * y11

    risk_bias = quadratic(c00, c01, c11, q0, q1)
    risk_noise = trace_product(c00, c01, c11, w00, w01, w11)
    risk = risk_bias + risk_noise
    cq0 = c00 * q0 + c01 * q1
    cq1 = c01 * q0 + c11 * q1
    first = (2 * (qp0 * cq0 + qp1 * cq1) + quadratic(cp00, cp01, cp11, q0, q1) +
             trace_product(cp00, cp01, cp11, w00, w01, w11) +
             trace_product(c00, c01, c11, wp00, wp01, wp11))
    if not second:
        return risk, first
    bpp00, bpp01, bpp11 = negative_akb((bp00, bp01, bp11), k_matrix, b_matrix)
    bpp00, bpp01, bpp11 = 2 * bpp00, 2 * bpp01, 2 * bpp11
    cpp00 = (2 * a0 * (bp00**2 + b00 * bpp00) +
             2 * a1 * (bp01**2 + b01 * bpp01))
    cpp01 = (bpp01 * (a0 * b00 + a1 * b11) +
             2 * bp01 * (a0 * bp00 + a1 * bp11) +
             b01 * (a0 * bpp00 + a1 * bpp11))
    cpp11 = (2 * a0 * (bp01**2 + b01 * bpp01) +
             2 * a1 * (bp11**2 + b11 * bpp11))
    wpp00 = 2 * model["sr2"] / n_r * r00 + 2 * model["sh2"] / n_h * h00
    wpp01 = 2 * model["sr2"] / n_r * r01 + 2 * model["sh2"] / n_h * h01
    wpp11 = 2 * model["sr2"] / n_r * r11 + 2 * model["sh2"] / n_h * h11
    second_value = (2 * quadratic(c00, c01, c11, qp0, qp1) +
                    4 * (qp0 * (cp00 * q0 + cp01 * q1) + qp1 * (cp01 * q0 + cp11 * q1)) +
                    quadratic(cpp00, cpp01, cpp11, q0, q1) +
                    trace_product(cpp00, cpp01, cpp11, w00, w01, w11) +
                    2 * trace_product(cp00, cp01, cp11, wp00, wp01, wp11) +
                    trace_product(c00, c01, c11, wpp00, wpp01, wpp11))
    return risk, first, second_value


def scalar_conditional_metrics(gram_r, gram_h, w, n_r, model):
    sr, sh = np.diag(model["a"]), np.diag(model["b"])
    beta, delta, mu = model["beta"], model["delta"], model["mu"]
    n_h = int(model["rho"] * n_r)
    gap = gram_h - gram_r
    resolvent = np.linalg.inv((1 - w) * gram_r + w * gram_h + mu * np.eye(2))
    resolvent_p = -resolvent @ gap @ resolvent
    resolvent_pp = 2 * resolvent @ gap @ resolvent @ gap @ resolvent
    q = -mu * beta + w * gram_h @ delta
    qp = gram_h @ delta
    noise = ((1 - w)**2 * model["sr2"] / n_r * gram_r +
             w**2 * model["sh2"] / n_h * gram_h)
    noise_p = (-2 * (1 - w) * model["sr2"] / n_r * gram_r +
               2 * w * model["sh2"] / n_h * gram_h)
    noise_pp = (2 * model["sr2"] / n_r * gram_r + 2 * model["sh2"] / n_h * gram_h)
    c = resolvent @ sr @ resolvent
    cp = resolvent_p @ sr @ resolvent + resolvent @ sr @ resolvent_p
    cpp = (resolvent_pp @ sr @ resolvent + 2 * resolvent_p @ sr @ resolvent_p +
           resolvent @ sr @ resolvent_pp)
    risk = q @ c @ q + np.trace(c @ noise)
    first = 2 * qp @ c @ q + q @ cp @ q + np.trace(cp @ noise + c @ noise_p)
    second = (2 * qp @ c @ qp + 4 * qp @ cp @ q + q @ cpp @ q +
              np.trace(cpp @ noise + 2 * cp @ noise_p + c @ noise_pp))
    return risk, first, second


def implementation_checks(model):
    checks = {}
    sr, sh = np.diag(model["a"]), np.diag(model["b"])
    coefficient_rows = []
    max_r0_error = max_r1_error = 0.0
    for w in [0.0, 0.1, float(model.get("w1", W1_INITIAL)), 0.5,
              float(model.get("w2", W2_INITIAL)), 0.9, 1.0]:
        reference = risk_coefficients(w, sr, sh, model["beta"], model["delta"], model["mu"],
                                      model["sr2"], model["sh2"], model["rho"])[:2]
        independent = independent_coefficients(w, model)
        errors = [abs(reference[i] - independent[i]) for i in range(2)]
        max_r0_error = max(max_r0_error, errors[0])
        max_r1_error = max(max_r1_error, errors[1])
        coefficient_rows.append({"w": w, "R0_absolute_error": errors[0], "R1_absolute_error": errors[1]})
    checks["risk_coefficients"] = {
        "rows": coefficient_rows,
        "max_R0_absolute_error": max_r0_error,
        "max_R1_absolute_error": max_r1_error,
        "passed": max_r0_error < 1e-12 and max_r1_error < 1e-10,
    }
    rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, 17, 10, 0]))
    grams_r, grams_h = sample_gram_pair(rng, 8, 137, model)
    weights = np.array([0.1, 0.2955, 0.6, 0.8833])
    vector = conditional_metrics(grams_r, grams_h, weights, 137, model, second=True)
    scalar = np.empty((3, 8, len(weights)))
    for rep in range(8):
        gr = np.array([[grams_r[0][rep], grams_r[1][rep]], [grams_r[1][rep], grams_r[2][rep]]])
        gh = np.array([[grams_h[0][rep], grams_h[1][rep]], [grams_h[1][rep], grams_h[2][rep]]])
        for j, w in enumerate(weights):
            values = scalar_conditional_metrics(gr, gh, w, 137, model)
            for k in range(3):
                scalar[k, rep, j] = values[k]
    errors = [float(np.max(np.abs(vector[k] - scalar[k]))) for k in range(3)]
    checks["vectorized_vs_scalar"] = {
        "replicates": 8, "weights": weights.tolist(),
        "max_absolute_errors_risk_first_second": errors,
        "passed": all(errors[k] < 1e-10 + 1e-8 * float(np.max(np.abs(scalar[k]))) for k in range(3)),
    }
    derivative_rows = []
    max_complex_first = max_complex_second = 0.0
    max_difference_first = max_difference_second = 0.0
    max_small_step_first = max_small_step_second = 0.0
    differences_converge = True
    for w in [0.2, 0.5, 0.88]:
        analytic = conditional_metrics(grams_r, grams_h, [w], 137, model, second=True)
        h_complex = 1e-25
        complex_values = conditional_metrics(grams_r, grams_h, [w + 1j * h_complex], 137, model, second=False)
        complex_first = np.imag(complex_values[0]) / h_complex
        complex_second = np.imag(complex_values[1]) / h_complex
        error_first = float(np.max(np.abs(complex_first - analytic[1])))
        error_second = float(np.max(np.abs(complex_second - analytic[2])))
        max_complex_first = max(max_complex_first, error_first)
        max_complex_second = max(max_complex_second, error_second)
        difference_rows = []
        for step in [1e-3, 1e-4, 1e-5]:
            left = conditional_metrics(grams_r, grams_h, [w - step], 137, model, second=True)
            right = conditional_metrics(grams_r, grams_h, [w + step], 137, model, second=True)
            difference_first = (right[0] - left[0]) / (2 * step)
            difference_second = (right[1] - left[1]) / (2 * step)
            e1 = float(np.max(np.abs(difference_first - analytic[1])))
            e2 = float(np.max(np.abs(difference_second - analytic[2])))
            max_difference_first = max(max_difference_first, e1)
            max_difference_second = max(max_difference_second, e2)
            difference_rows.append({"step": step, "max_first_absolute_error": e1,
                                    "max_second_absolute_error": e2})
        first_errors = [row["max_first_absolute_error"] for row in difference_rows]
        second_errors = [row["max_second_absolute_error"] for row in difference_rows]
        differences_converge = differences_converge and first_errors[2] < first_errors[1] < first_errors[0] and second_errors[2] < second_errors[1] < second_errors[0]
        max_small_step_first = max(max_small_step_first, first_errors[-1])
        max_small_step_second = max(max_small_step_second, second_errors[-1])
        derivative_rows.append({"w": w, "complex_step_first_max_absolute_error": error_first,
                                "complex_step_second_max_absolute_error": error_second,
                                "finite_differences": difference_rows})
    derivative_tolerance_first = 1e-8
    derivative_tolerance_second = 1e-6
    checks["analytic_derivatives"] = {
        "rows": derivative_rows,
        "max_complex_step_first_absolute_error": max_complex_first,
        "max_complex_step_second_absolute_error": max_complex_second,
        "max_finite_difference_first_absolute_error_all_steps": max_difference_first,
        "max_finite_difference_second_absolute_error_all_steps": max_difference_second,
        "max_smallest_step_first_absolute_error": max_small_step_first,
        "max_smallest_step_second_absolute_error": max_small_step_second,
        "finite_difference_errors_decrease_with_step": differences_converge,
        "passed": (max_complex_first < 1e-8 and max_complex_second < 1e-6 and
                   max_small_step_first < derivative_tolerance_first and
                   max_small_step_second < derivative_tolerance_second and differences_converge),
    }
    wishart_rows = []
    wishart_passed = True
    for domain, eigenvalues in [("target", model["a"]), ("human", model["b"])]:
        reps, n = 262144, 23
        local_rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, 23, 11, 0 if domain == "target" else 1]))
        local_model = dict(model)
        local_model["rho"] = 1.0
        if domain == "target":
            grams, _ = sample_gram_pair(local_rng, reps, n, local_model)
        else:
            local_model["a"] = model["b"]
            grams, _ = sample_gram_pair(local_rng, reps, n, local_model)
        sample_matrix = np.column_stack(grams)
        mean = sample_matrix.mean(axis=0)
        expected_mean = np.array([eigenvalues[0], 0.0, eigenvalues[1]])
        mean_se = sample_matrix.std(axis=0, ddof=1) / math.sqrt(reps)
        mean_z = np.abs((mean - expected_mean) / mean_se)
        covariance = np.cov(sample_matrix, rowvar=False)
        sigma = np.diag(eigenvalues)
        expected_covariance = np.empty((3, 3))
        indices = [(0, 0), (0, 1), (1, 1)]
        for i, (j, k) in enumerate(indices):
            for l, (m, q) in enumerate(indices):
                expected_covariance[i, l] = (sigma[j, m] * sigma[k, q] + sigma[j, q] * sigma[k, m]) / n
        covariance_scale = np.sqrt(np.outer(np.diag(expected_covariance), np.diag(expected_covariance)))
        covariance_scaled_error = float(np.max(np.abs(covariance - expected_covariance) / covariance_scale))
        nonzero = expected_covariance != 0
        covariance_max_relative = float(np.max(np.abs(covariance[nonzero] - expected_covariance[nonzero]) /
                                               np.abs(expected_covariance[nonzero])))
        passed = bool(np.max(mean_z) < 5 and covariance_scaled_error < 0.05)
        wishart_passed = wishart_passed and passed
        wishart_rows.append({"domain": domain, "n": n, "replicates": reps,
                             "sample_mean": mean.tolist(), "expected_mean": expected_mean.tolist(),
                             "max_absolute_mean_z": float(np.max(mean_z)),
                             "sample_covariance": covariance.tolist(),
                             "expected_covariance": expected_covariance.tolist(),
                             "max_covariance_relative_error_nonzero_entries": covariance_max_relative,
                             "max_covariance_variance_scaled_absolute_error": covariance_scaled_error,
                             "passed": passed})
    checks["wishart_moments"] = {"rows": wishart_rows, "passed": wishart_passed}
    checks["all_passed"] = all(value.get("passed", True) for value in checks.values())
    return checks


def evaluate_find_batches(n_r, model, start_batch, end_batch, grid):
    batch_risk = np.empty((end_batch - start_batch, len(grid)))
    batch_derivative = np.empty_like(batch_risk)
    for local_batch, batch_id in enumerate(range(start_batch, end_batch)):
        rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, n_r, 0, batch_id]))
        grams_r, grams_h = sample_gram_pair(rng, FIND_BATCH_SIZE, n_r, model)
        for start in range(0, len(grid), WEIGHT_CHUNK):
            stop = min(start + WEIGHT_CHUNK, len(grid))
            risk, first = conditional_metrics(grams_r, grams_h, grid[start:stop], n_r, model, second=False)
            batch_risk[local_batch, start:stop] = np.mean(risk, axis=0)
            batch_derivative[local_batch, start:stop] = np.mean(first, axis=0)
    return batch_risk, batch_derivative


def evaluate_stage0_at_weights(n_r, model, batch_count, weights):
    weights = np.asarray(weights, dtype=float)
    output = np.empty((batch_count, len(weights), 3))
    for batch_id in range(batch_count):
        rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, n_r, 0, batch_id]))
        grams_r, grams_h = sample_gram_pair(rng, FIND_BATCH_SIZE, n_r, model)
        for start in range(0, len(weights), WEIGHT_CHUNK):
            stop = min(start + WEIGHT_CHUNK, len(weights))
            values = conditional_metrics(grams_r, grams_h, weights[start:stop], n_r, model, second=True)
            for k in range(3):
                output[batch_id, start:stop, k] = np.mean(values[k], axis=0)
    return output


def evaluate_independent(n_r, model, batch_count, weights):
    weights = np.asarray(weights, dtype=float)
    output = np.empty((batch_count, len(weights), 3))
    for batch_id in range(batch_count):
        rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, n_r, 1, batch_id]))
        grams_r, grams_h = sample_gram_pair(rng, EVAL_BATCH_SIZE, n_r, model)
        values = conditional_metrics(grams_r, grams_h, weights, n_r, model, second=True)
        for k in range(3):
            output[batch_id, :, k] = np.mean(values[k], axis=0)
    return output


def find_sign_change_roots(grid, derivative):
    roots = []
    for i in range(len(grid) - 1):
        left, right = derivative[i], derivative[i + 1]
        if left == 0:
            roots.append((float(grid[i]), [float(grid[i]), float(grid[i])]))
        if left * right < 0:
            function = lambda x: float(np.interp(x, grid, derivative))
            root = brentq(function, grid[i], grid[i + 1], xtol=1e-14, rtol=1e-12)
            roots.append((float(root), [float(grid[i]), float(grid[i + 1])]))
    if derivative[-1] == 0:
        roots.append((float(grid[-1]), [float(grid[-1]), float(grid[-1])]))
    roots.sort()
    unique = []
    for root, bracket in roots:
        if unique and abs(root - unique[-1][0]) < 1e-10:
            continue
        unique.append((root, bracket))
    classified = []
    for root, bracket in unique:
        index = min(len(grid) - 2, max(0, np.searchsorted(grid, root) - 1))
        left_indices = np.flatnonzero(grid < root)
        right_indices = np.flatnonzero(grid > root)
        left_sign = np.sign(derivative[left_indices[-1]]) if len(left_indices) else 0
        right_sign = np.sign(derivative[right_indices[0]]) if len(right_indices) else 0
        kind = "local_minimum" if left_sign < 0 < right_sign else "local_maximum" if left_sign > 0 > right_sign else "degenerate"
        classified.append({"w": root, "kind": kind, "bracket": bracket,
                           "left_derivative_sign": int(left_sign), "right_derivative_sign": int(right_sign),
                           "grid_index": int(index)})
    return classified


def compare_grid_roots(coarse, fine):
    if len(coarse) != len(fine):
        return {"consistent": False, "reason": "root count changed", "maximum_position_difference": None}
    differences = []
    for left, right in zip(coarse, fine):
        if left["kind"] != right["kind"]:
            return {"consistent": False, "reason": "root classification changed",
                    "maximum_position_difference": None}
        differences.append(abs(left["w"] - right["w"]))
    maximum = max(differences, default=0.0)
    return {"consistent": maximum <= 0.001, "reason": None if maximum <= 0.001 else "root positions changed",
            "maximum_position_difference": maximum}


def near_zero_regions(grid, derivative, standard_error, roots):
    regions = []
    for i in range(1, len(grid) - 1):
        if not (abs(derivative[i]) <= abs(derivative[i - 1]) and
                abs(derivative[i]) <= abs(derivative[i + 1])):
            continue
        if derivative[i - 1] * derivative[i] < 0 or derivative[i] * derivative[i + 1] < 0:
            continue
        if standard_error[i] == 0 or abs(derivative[i]) > 3 * standard_error[i]:
            continue
        if any(abs(grid[i] - root["w"]) <= 2 * (grid[1] - grid[0]) for root in roots):
            continue
        regions.append({"center_index": int(i), "center_w": float(grid[i]),
                        "left_w": float(grid[i - 1]), "right_w": float(grid[i + 1]),
                        "derivative": float(derivative[i]), "standard_error": float(standard_error[i])})
    return regions


def bootstrap_analysis(n_r, multiplier, grid, batch_risk, batch_derivative):
    batch_count = batch_risk.shape[0]
    rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, n_r, 2, multiplier]))
    root_counts = np.empty(BOOTSTRAP_REPS, dtype=np.int32)
    minimum_counts = np.empty(BOOTSTRAP_REPS, dtype=np.int32)
    maximum_counts = np.empty(BOOTSTRAP_REPS, dtype=np.int32)
    positions = np.full((BOOTSTRAP_REPS, 2), np.nan)
    deltas = np.full(BOOTSTRAP_REPS, np.nan)
    stable = np.zeros(BOOTSTRAP_REPS, dtype=bool)
    for rep in range(BOOTSTRAP_REPS):
        indices = rng.integers(0, batch_count, size=batch_count)
        derivative = batch_derivative[indices].mean(axis=0)
        roots = find_sign_change_roots(grid, derivative)
        minima = [root for root in roots if root["kind"] == "local_minimum" and 0 < root["w"] < 1]
        maxima = [root for root in roots if root["kind"] == "local_maximum" and 0 < root["w"] < 1]
        root_counts[rep], minimum_counts[rep], maximum_counts[rep] = len(roots), len(minima), len(maxima)
        stable[rep] = len(minima) == 2 and len(maxima) >= 1
        if len(minima) == 2:
            positions[rep] = [minima[0]["w"], minima[1]["w"]]
            risk = batch_risk[indices].mean(axis=0)
            deltas[rep] = np.interp(positions[rep, 1], grid, risk) - np.interp(positions[rep, 0], grid, risk)
    valid_positions = np.isfinite(positions[:, 0])
    position_intervals = []
    for j in range(2):
        if np.any(valid_positions):
            position_intervals.append(np.nanpercentile(positions[:, j], [2.5, 97.5]).tolist())
        else:
            position_intervals.append([None, None])
    valid_deltas = deltas[np.isfinite(deltas)]
    delta_interval = np.percentile(valid_deltas, [2.5, 97.5]).tolist() if len(valid_deltas) else [None, None]
    summary = {
        "replicates": BOOTSTRAP_REPS,
        "batch_count": batch_count,
        "double_valley_support_rate": float(np.mean(stable)),
        "two_minima_rate": float(np.mean(minimum_counts == 2)),
        "position_intervals_conditional_on_two_minima": position_intervals,
        "valid_position_replicates": int(np.sum(valid_positions)),
        "reoptimized_delta_percentile_interval": delta_interval,
        "valid_delta_replicates": int(len(valid_deltas)),
        "root_count_frequencies": {str(int(value)): int(np.sum(root_counts == value)) for value in np.unique(root_counts)},
        "minimum_count_frequencies": {str(int(value)): int(np.sum(minimum_counts == value)) for value in np.unique(minimum_counts)},
    }
    arrays = {"bootstrap_root_counts": root_counts, "bootstrap_minimum_counts": minimum_counts,
              "bootstrap_maximum_counts": maximum_counts, "bootstrap_positions": positions,
              "bootstrap_deltas": deltas, "bootstrap_stable": stable}
    return summary, arrays


def paired_summary(batch_values, left_index, right_index):
    differences = batch_values[:, right_index, 0] - batch_values[:, left_index, 0]
    mean, se = batch_mean_se(differences)
    return {"mean": finite(mean), "standard_error": finite(se), "confidence_interval": interval(mean, se)}


def candidate_rows(weights, batch_metrics, classifications=None):
    mean, se = batch_mean_se(batch_metrics)
    rows = []
    for i, w in enumerate(weights):
        row = {"w": float(w), "kind": classifications[i] if classifications is not None else "candidate"}
        for k, name in enumerate(["risk", "derivative", "curvature"]):
            row[name] = finite(mean[i, k])
            row[f"{name}_standard_error"] = finite(se[i, k])
            row[f"{name}_confidence_interval"] = interval(mean[i, k], se[i, k])
        rows.append(row)
    return rows


def escalation_reasons(result):
    reasons = []
    shape = result["shape"]
    if not shape["stable_double_valley"]:
        if (not shape["grid_comparison"]["consistent"] or shape["unresolved_near_zero_regions"] or
                shape["refinement_changed_root_count"]):
            reasons.append("grid shape inconsistent or unresolved")
    rate = result["bootstrap"]["double_valley_support_rate"]
    if 0.05 < rate < 0.95:
        reasons.append("bootstrap root structure remains indeterminate")
    if result.get("branch_difference") is not None:
        if not result["branch_difference"]["direction_resolved"]:
            reasons.append("independent paired branch direction unresolved")
        if result["branch_difference"]["first_order_accuracy"] == "unresolved":
            reasons.append("first-order risk-difference accuracy unresolved")
    return list(dict.fromkeys(reasons))


def run_attempt(n_r, model, output, population, multiplier, previous_curves=None):
    started = time.time()
    grid = np.linspace(0.0, 1.0, GRID_FINE_SIZE)
    target_batches = FIND_BATCHES * multiplier
    if previous_curves is None:
        batch_risk, batch_derivative = evaluate_find_batches(n_r, model, 0, target_batches, grid)
    else:
        old_risk, old_derivative = previous_curves
        new_risk, new_derivative = evaluate_find_batches(n_r, model, old_risk.shape[0], target_batches, grid)
        batch_risk = np.concatenate([old_risk, new_risk], axis=0)
        batch_derivative = np.concatenate([old_derivative, new_derivative], axis=0)
    mean_derivative, se_derivative = batch_mean_se(batch_derivative)
    coarse_roots = find_sign_change_roots(grid[::2], mean_derivative[::2])
    fine_roots = find_sign_change_roots(grid, mean_derivative)
    grid_comparison = compare_grid_roots(coarse_roots, fine_roots)
    near_regions = near_zero_regions(grid, mean_derivative, se_derivative, fine_roots)
    refinement_weights = []
    for region in near_regions:
        refinement_weights.extend(np.linspace(region["left_w"], region["right_w"], 17).tolist())
    direct_weights = [root["w"] for root in fine_roots] + refinement_weights
    direct_metrics = evaluate_stage0_at_weights(n_r, model, target_batches, direct_weights) if direct_weights else np.empty((target_batches, 0, 3))
    direct_mean = direct_metrics.mean(axis=0) if direct_weights else np.empty((0, 3))
    refined_roots = []
    offset = len(fine_roots)
    for region_index, region in enumerate(near_regions):
        local_grid = np.linspace(region["left_w"], region["right_w"], 17)
        local_derivative = direct_mean[offset + 17 * region_index:offset + 17 * (region_index + 1), 1]
        local_roots = find_sign_change_roots(local_grid, local_derivative)
        for root in local_roots:
            if all(abs(root["w"] - old["w"]) > 1e-7 for old in fine_roots):
                root = dict(root)
                root["source"] = "direct_local_refinement"
                refined_roots.append(root)
    extra_metrics = evaluate_stage0_at_weights(n_r, model, target_batches, [root["w"] for root in refined_roots]) if refined_roots else np.empty((target_batches, 0, 3))
    all_roots = fine_roots + refined_roots
    all_roots.sort(key=lambda row: row["w"])
    root_batch_metrics = np.concatenate([direct_metrics[:, :len(fine_roots)], extra_metrics], axis=1)
    if refined_roots:
        order = np.argsort([root["w"] for root in fine_roots + refined_roots])
        root_batch_metrics = root_batch_metrics[:, order]
    root_rows = candidate_rows([root["w"] for root in all_roots], root_batch_metrics,
                               [root["kind"] for root in all_roots])
    sign_curvature_consistent = True
    for row, root in zip(root_rows, all_roots):
        row["sign_based_kind"] = root["kind"]
        row["bracket"] = root["bracket"]
        row["source"] = root.get("source", "fine_grid")
        curvature_kind = "local_minimum" if row["curvature"] > 0 else "local_maximum" if row["curvature"] < 0 else "degenerate"
        row["curvature_based_kind"] = curvature_kind
        if curvature_kind != root["kind"]:
            sign_curvature_consistent = False
        row["kind"] = curvature_kind
    minima = [row for row in root_rows if row["kind"] == "local_minimum" and 0 < row["w"] < 1]
    maxima = [row for row in root_rows if row["kind"] == "local_maximum" and 0 < row["w"] < 1]
    bootstrap, bootstrap_arrays = bootstrap_analysis(n_r, multiplier, grid, batch_risk, batch_derivative)
    positive_curvature = len(minima) == 2 and all(row["curvature_confidence_interval"][0] > 0 for row in minima)
    stable_double = (len(minima) == 2 and len(maxima) >= 1 and positive_curvature and
                     grid_comparison["consistent"] and not refined_roots and
                     bootstrap["double_valley_support_rate"] >= 0.95 and sign_curvature_consistent)
    candidate_weights = [0.0] + [row["w"] for row in root_rows] + [1.0]
    candidate_kinds = ["boundary"] + [row["kind"] for row in root_rows] + ["boundary"]
    evaluation_batch_count = EVAL_BATCHES * multiplier
    evaluation_metrics = evaluate_independent(n_r, model, evaluation_batch_count, candidate_weights)
    evaluation_rows = candidate_rows(candidate_weights, evaluation_metrics, candidate_kinds)
    for index, row in enumerate(evaluation_rows):
        row["candidate_index"] = index
    evaluation_minima = [row for row in evaluation_rows if row["kind"] == "local_minimum" and 0 < row["w"] < 1]
    global_row = min(evaluation_rows, key=lambda row: row["risk"])
    global_comparisons = []
    for row in evaluation_rows:
        if row["candidate_index"] == global_row["candidate_index"]:
            continue
        if row["kind"] == "local_maximum":
            continue
        comparison = paired_summary(evaluation_metrics, global_row["candidate_index"], row["candidate_index"])
        comparison["other_w"] = row["w"]
        comparison["other_kind"] = row["kind"]
        global_comparisons.append(comparison)
    global_conditionally_resolved = all(row["confidence_interval"][0] > 0 for row in global_comparisons)
    branch_difference = None
    position_rows = []
    w1_float, w2_float = float(population["w1"]), float(population["w2"])
    if len(minima) == 2:
        lower_stage0, upper_stage0 = minima
        lower_eval = next(row for row in evaluation_minima if abs(row["w"] - lower_stage0["w"]) < 1e-12)
        upper_eval = next(row for row in evaluation_minima if abs(row["w"] - upper_stage0["w"]) < 1e-12)
        difference = paired_summary(evaluation_metrics, lower_eval["candidate_index"], upper_eval["candidate_index"])
        position_intervals = bootstrap["position_intervals_conditional_on_two_minima"]
        position_error_terms = []
        for branch_index, row in enumerate([lower_eval, upper_eval]):
            bounds = position_intervals[branch_index]
            if bounds[0] is None:
                radius = math.inf
            else:
                radius = max(abs(bounds[0] - row["w"]), abs(bounds[1] - row["w"]))
            error = 0.5 * abs(row["curvature"]) * radius**2
            position_error_terms.append(error)
        position_error = sum(position_error_terms)
        conditional_interval = difference["confidence_interval"]
        expanded_interval = [conditional_interval[0] - position_error,
                             conditional_interval[1] + position_error]
        scaled_interval = [n_r * expanded_interval[0], n_r * expanded_interval[1]]
        c_delta = population["C_Delta"]
        tolerance_band = [0.9 * c_delta, 1.1 * c_delta]
        if scaled_interval[0] >= tolerance_band[0] and scaled_interval[1] <= tolerance_band[1]:
            accuracy = "support"
        elif scaled_interval[1] < tolerance_band[0] or scaled_interval[0] > tolerance_band[1]:
            accuracy = "unsupported"
        else:
            accuracy = "unresolved"
        direction_resolved = expanded_interval[0] > 0 or expanded_interval[1] < 0
        difference.update({
            "definition": "R_n(upper actual minimum) - R_n(lower actual minimum), independently evaluated at locked stage-1 positions",
            "lower_w": lower_eval["w"], "upper_w": upper_eval["w"],
            "position_error_quadratic_terms": position_error_terms,
            "position_error_expansion": finite(position_error),
            "position_expanded_confidence_interval": [finite(x) for x in expanded_interval],
            "scaled_mean_n_r_Delta": n_r * difference["mean"],
            "scaled_position_expanded_confidence_interval": [finite(x) for x in scaled_interval],
            "C_Delta": c_delta,
            "relative_bias": (n_r * difference["mean"] - c_delta) / abs(c_delta),
            "first_order_tolerance_band_scaled": tolerance_band,
            "first_order_accuracy": accuracy,
            "direction_resolved": bool(direction_resolved),
            "selected_branch": "lower" if direction_resolved and difference["mean"] > 0 else "upper" if direction_resolved else None,
            "reoptimized_bootstrap_interval": bootstrap["reoptimized_delta_percentile_interval"],
        })
        branch_difference = difference
        for branch_index, (population_w, row) in enumerate(zip([w1_float, w2_float], [lower_eval, upper_eval])):
            q = population["position_shift_q"][branch_index]
            predicted = population_w + q / n_r
            position_rows.append({
                "branch": branch_index + 1, "population_w": population_w, "actual_w": row["w"],
                "absolute_position_error": abs(row["w"] - population_w),
                "scaled_shift": n_r * (row["w"] - population_w), "q": q,
                "first_order_prediction": predicted,
                "prediction_available": 0 <= predicted <= 1,
                "prediction_absolute_error": abs(row["w"] - predicted) if 0 <= predicted <= 1 else None,
                "bootstrap_position_interval": position_intervals[branch_index],
                "independent_curvature": row["curvature"],
                "independent_curvature_confidence_interval": row["curvature_confidence_interval"],
                "independent_derivative": row["derivative"],
                "independent_derivative_confidence_interval": row["derivative_confidence_interval"],
            })
    failure_reasons = []
    if not stable_double:
        failure_reasons.append("two stable local minima not identified at the current numerical resolution")
    if any(not row["prediction_available"] for row in position_rows):
        failure_reasons.append("at least one first-order position prediction is outside [0,1] and was not clipped")
    if branch_difference is not None:
        if branch_difference["first_order_accuracy"] == "unsupported":
            failure_reasons.append("two branches exist but the first-order risk difference is outside the 10% tolerance")
        elif branch_difference["first_order_accuracy"] == "unresolved":
            failure_reasons.append("first-order risk-difference accuracy is unresolved by the interval")
        if not branch_difference["direction_resolved"]:
            failure_reasons.append("branch difference is masked by Monte Carlo noise or position uncertainty")
    if refined_roots or not grid_comparison["consistent"] or near_regions:
        failure_reasons.append("near-bifurcation roots or grid shape require caution")
    result = {
        "n_r": n_r, "n_h": int(model["rho"] * n_r), "budget_multiplier": multiplier,
        "find_batches": target_batches, "find_batch_size": FIND_BATCH_SIZE,
        "find_replicates": target_batches * FIND_BATCH_SIZE,
        "evaluation_batches": evaluation_batch_count, "evaluation_batch_size": EVAL_BATCH_SIZE,
        "evaluation_replicates": evaluation_batch_count * EVAL_BATCH_SIZE,
        "started_or_completed_utc": utc_now(), "elapsed_seconds": None,
        "shape": {
            "stationary_count": len(root_rows), "local_minimum_count": len(minima), "local_maximum_count": len(maxima),
            "grid_comparison": grid_comparison, "near_zero_regions_checked": near_regions,
            "unresolved_near_zero_regions": [],
            "refined_roots": refined_roots, "refinement_changed_root_count": bool(refined_roots),
            "sign_curvature_consistent": sign_curvature_consistent,
            "positive_curvature_evidence": positive_curvature,
            "stable_double_valley": bool(stable_double),
        },
        "roots": root_rows, "bootstrap": bootstrap,
        "independent_candidates": evaluation_rows,
        "conditional_global_candidate": {"w": global_row["w"], "kind": global_row["kind"],
                                          "risk": global_row["risk"],
                                          "all_relevant_paired_intervals_exclude_tie": global_conditionally_resolved,
                                          "comparisons_other_minus_global": global_comparisons},
        "branch_difference": branch_difference, "position_analysis": position_rows,
        "failure_or_caution_reasons": failure_reasons,
        "code_sha256": runner_sha256(),
    }
    result["escalation_reasons"] = escalation_reasons(result)
    result["completed"] = not result["escalation_reasons"] or multiplier >= MAX_MULTIPLIER
    result["elapsed_seconds"] = time.time() - started
    arrays = {"weights": grid, "batch_risk": batch_risk, "batch_derivative": batch_derivative,
              "stage0_candidate_weights": np.array([root["w"] for root in all_roots]),
              "stage0_candidate_batch_metrics": root_batch_metrics,
              "evaluation_weights": np.array(candidate_weights), "evaluation_batch_metrics": evaluation_metrics,
              **bootstrap_arrays}
    return result, arrays, (batch_risk, batch_derivative)


def valid_checkpoint(result, code_hash):
    return result.get("code_sha256") == code_hash


def run_sample_size(n_r, model, output, population, resume=False):
    sample_output = output / "checkpoints" / f"n_{n_r}"
    sample_output.mkdir(parents=True, exist_ok=True)
    code_hash = runner_sha256()
    latest_result = None
    previous_curves = None
    for multiplier in [1, 2, 4, 8]:
        result_path = sample_output / f"attempt_{multiplier}.json"
        arrays_path = sample_output / f"curves_{multiplier}.npz"
        if resume and result_path.exists() and arrays_path.exists():
            candidate = json.loads(result_path.read_text(encoding="utf-8"))
            if valid_checkpoint(candidate, code_hash):
                latest_result = candidate
                with np.load(arrays_path) as data:
                    previous_curves = (data["batch_risk"], data["batch_derivative"])
                if candidate["completed"]:
                    latest_result["attempt_history"] = [json.loads((sample_output / f"attempt_{m}.json").read_text(encoding="utf-8"))
                                                        for m in [1, 2, 4, 8] if m <= multiplier and (sample_output / f"attempt_{m}.json").exists()]
                    return latest_result
                continue
        result, arrays, previous_curves = run_attempt(n_r, model, output, population, multiplier, previous_curves)
        np.savez_compressed(arrays_path.with_suffix(""), **arrays)
        json_dump(result_path, result)
        latest_result = result
        if result["completed"]:
            break
    history = []
    for multiplier in [1, 2, 4, 8]:
        path = sample_output / f"attempt_{multiplier}.json"
        if path.exists():
            history.append(json.loads(path.read_text(encoding="utf-8")))
    latest_result["attempt_history"] = history
    return latest_result


def summary_row(result):
    row = {"n_r": result["n_r"], "n_h": result["n_h"], "budget_multiplier": result["budget_multiplier"],
           "find_replicates": result["find_replicates"], "evaluation_replicates": result["evaluation_replicates"],
           "stationary_count": result["shape"]["stationary_count"],
           "local_minimum_count": result["shape"]["local_minimum_count"],
           "local_maximum_count": result["shape"]["local_maximum_count"],
           "stable_double_valley": result["shape"]["stable_double_valley"],
           "bootstrap_double_valley_rate": result["bootstrap"]["double_valley_support_rate"]}
    minima = [root for root in result["roots"] if root["kind"] == "local_minimum" and 0 < root["w"] < 1]
    for i in range(2):
        row[f"w{i + 1}"] = minima[i]["w"] if len(minima) == 2 else None
        row[f"curvature{i + 1}"] = minima[i]["curvature"] if len(minima) == 2 else None
        row[f"position_scaled_shift{i + 1}"] = result["position_analysis"][i]["scaled_shift"] if len(result["position_analysis"]) == 2 else None
        row[f"position_q{i + 1}"] = result["position_analysis"][i]["q"] if len(result["position_analysis"]) == 2 else None
        row[f"position_prediction_available{i + 1}"] = result["position_analysis"][i]["prediction_available"] if len(result["position_analysis"]) == 2 else None
    difference = result["branch_difference"]
    for key, csv_key in [("mean", "Delta_n"), ("standard_error", "Delta_standard_error"),
                         ("relative_bias", "relative_bias"), ("scaled_mean_n_r_Delta", "n_r_Delta"),
                         ("first_order_accuracy", "first_order_accuracy"),
                         ("direction_resolved", "direction_resolved"), ("selected_branch", "selected_branch")]:
        row[csv_key] = difference.get(key) if difference else None
    row["Delta_ci_low"] = difference["position_expanded_confidence_interval"][0] if difference else None
    row["Delta_ci_high"] = difference["position_expanded_confidence_interval"][1] if difference else None
    row["global_candidate_w"] = result["conditional_global_candidate"]["w"]
    row["failure_or_caution_reasons"] = "; ".join(result["failure_or_caution_reasons"])
    row["escalation_reasons"] = "; ".join(result["escalation_reasons"])
    row["completed"] = result["completed"]
    return row


def write_summary_csv(results, path):
    rows = [summary_row(result) for result in results]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def consolidate_outputs(results, output, population, implementation):
    results.sort(key=lambda row: row["n_r"])
    json_dump(output / "results.json", {"experiment": "finite_sample_branch_selection_2026-11-02",
                                        "generated_utc": utc_now(), "population": population,
                                        "implementation_checks_passed": implementation["all_passed"],
                                        "results": results})
    write_summary_csv(results, output / "summary.csv")
    combined = {}
    for result in results:
        n_r, multiplier = result["n_r"], result["budget_multiplier"]
        path = output / "checkpoints" / f"n_{n_r}" / f"curves_{multiplier}.npz"
        with np.load(path) as data:
            for key in data.files:
                combined[f"n_{n_r}_{key}"] = data[key]
    np.savez_compressed(output / "batch_curves.npz", **combined)
    generate_figures(results, output, population)
    generate_chinese_summary(results, output, population)


def population_risk_float(w, model):
    w = np.asarray(w)
    values = np.zeros_like(w, dtype=float)
    scales = [1.0, math.sqrt(model["t"])]
    u = [math.sqrt(26) / 1000, math.sqrt(3) / 500]
    c = [0.02, 0.95]
    for j in range(2):
        d = (1 - w) * model["a"][j] + w * model["b"][j] + model["mu"]
        values += model["a"][j] * (scales[j] * u[j])**2 * (w - c[j])**2 / d**2
    return values


def save_figure(fig, output, name):
    fig.savefig(output / "figures" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / "figures" / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_figures(results, output, population):
    (output / "figures").mkdir(parents=True, exist_ok=True)
    model = {key: np.array(value) if key in ["a", "b", "beta", "delta"] else value
             for key, value in population["model_float"].items()}
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True)
    for axis, result in zip(axes.flat, results):
        n_r, multiplier = result["n_r"], result["budget_multiplier"]
        with np.load(output / "checkpoints" / f"n_{n_r}" / f"curves_{multiplier}.npz") as data:
            grid = data["weights"]
            risk, se = batch_mean_se(data["batch_risk"])
        r0 = population_risk_float(grid, model)
        r1 = np.array([independent_coefficients(w, model)[1] for w in grid])
        axis.plot(grid, r0, color="black", linewidth=1.2, label="Population $R_0$")
        axis.plot(grid, risk, color="tab:blue", linewidth=1.1, label="Actual MC $R_n$")
        axis.fill_between(grid, risk - Z95 * se, risk + Z95 * se, color="tab:blue", alpha=0.2,
                          label="Pointwise 95% CI")
        axis.plot(grid, r0 + r1 / n_r, color="tab:orange", linestyle="--", linewidth=1.0,
                  label="$R_0+R_1/n_r$")
        for root in result["roots"]:
            if root["kind"] == "local_minimum":
                axis.axvline(root["w"], color="tab:blue", alpha=0.45, linewidth=0.8)
        for w in [float(population["w1"]), float(population["w2"])]:
            axis.axvline(w, color="black", linestyle=":", alpha=0.65, linewidth=0.8)
        axis.set_title(f"$n_r={n_r:,}$; $n_h={2*n_r:,}$")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(results):]:
        axis.axis("off")
    for axis in axes[-1]:
        axis.set_xlabel("Mixing weight $w$")
    for axis in axes[:, 0]:
        axis.set_ylabel("Target excess risk")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("Actual finite-sample risk curves (all sample sizes)")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    save_figure(fig, output, "risk_curves_all")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ns = np.array([result["n_r"] for result in results], dtype=float)
    for branch, color in [(0, "tab:blue"), (1, "tab:orange")]:
        actual = np.full(len(results), np.nan)
        lower = np.full(len(results), np.nan)
        upper = np.full(len(results), np.nan)
        predicted = np.full(len(results), np.nan)
        for i, result in enumerate(results):
            if len(result["position_analysis"]) == 2:
                row = result["position_analysis"][branch]
                actual[i] = row["actual_w"]
                bounds = row["bootstrap_position_interval"]
                if bounds[0] is not None:
                    lower[i], upper[i] = bounds
                if row["prediction_available"]:
                    predicted[i] = row["first_order_prediction"]
        ax.plot(ns, actual, "o", color=color, label=f"Actual minimum {branch + 1}")
        valid = np.isfinite(lower)
        if np.any(valid):
            ax.fill_between(ns[valid], lower[valid], upper[valid], color=color, alpha=0.15)
        ax.plot(ns, predicted, "--", color=color, alpha=0.8, label=f"First-order prediction {branch + 1}")
        ax.axhline(float(population[f"w{branch + 1}"]), color=color, linestyle=":", linewidth=1.4,
                   label=f"Population minimum {branch + 1}")
    ax.set_xscale("log")
    ax.set_xlabel("Target sample size $n_r$ (log scale)")
    ax.set_ylabel("Mixing weight $w$")
    ax.set_title("Finite-sample valley trajectories")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output, "valley_trajectories")

    available = [result for result in results if result["branch_difference"] is not None]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    if available:
        x = np.array([row["n_r"] for row in available], dtype=float)
        delta = np.array([row["branch_difference"]["mean"] for row in available])
        ci = np.array([row["branch_difference"]["position_expanded_confidence_interval"] for row in available])
        c_delta = population["C_Delta"]
        axes[0].errorbar(x, delta, yerr=np.vstack([delta - ci[:, 0], ci[:, 1] - delta]), fmt="o",
                         capsize=3, label="Actual $\widehat{\Delta}_n$")
        axes[0].plot(x, c_delta / x, "--", label="$C_\Delta/n_r$")
        axes[0].set_ylabel("Upper-minus-lower branch risk")
        axes[0].set_yscale("symlog", linthresh=max(1e-16, float(np.min(np.abs(delta))) / 10))
        scaled = x * delta
        scaled_ci = x[:, None] * ci
        axes[1].errorbar(x, scaled, yerr=np.vstack([scaled - scaled_ci[:, 0], scaled_ci[:, 1] - scaled]),
                         fmt="o", capsize=3, color="tab:blue", label="Actual $n_r\widehat{\Delta}_n$")
        axes[1].axhline(c_delta, color="black", linestyle="--", label="$C_\Delta$")
        axes[1].axhspan(0.9 * c_delta, 1.1 * c_delta, color="gray", alpha=0.15, label="10% tolerance")
        axes[1].set_ylabel("Scaled branch risk difference")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Target sample size $n_r$ (log scale)")
    axes[0].set_title("Actual branch selection versus first-order prediction")
    fig.tight_layout()
    save_figure(fig, output, "branch_risk_difference")

    fig, ax = plt.subplots(figsize=(22, 7))
    ax.axis("off")
    columns = ["$n_r$", "Find / eval. reps", "Stat. / min.", "$w_1$", "$w_2$", "Curvatures",
               "$\widehat{\Delta}_n$ 95% CI", "Rel. bias", "Boot. double", "Status"]
    table_rows = []
    for result in results:
        row = summary_row(result)
        ci = "—" if row["Delta_ci_low"] is None else f"[{row['Delta_ci_low']:.2e}, {row['Delta_ci_high']:.2e}]"
        curvatures = "—" if row["curvature1"] is None else f"{row['curvature1']:.2e}, {row['curvature2']:.2e}"
        status = "stable double" if row["stable_double_valley"] else "not stable / unresolved"
        if row["first_order_accuracy"]:
            status += "; " + row["first_order_accuracy"]
        table_rows.append([f"{result['n_r']:,}", f"{result['find_replicates']:,} / {result['evaluation_replicates']:,}",
                           f"{row['stationary_count']} / {row['local_minimum_count']}",
                           "—" if row["w1"] is None else f"{row['w1']:.6f}",
                           "—" if row["w2"] is None else f"{row['w2']:.6f}", curvatures, ci,
                           "—" if row["relative_bias"] is None else f"{100 * row['relative_bias']:+.2f}%",
                           f"{row['bootstrap_double_valley_rate']:.3f}", status])
    table = ax.table(cellText=table_rows, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.55)
    ax.set_title("Finite-sample branch-selection diagnostics (pointwise intervals)", pad=18)
    save_figure(fig, output, "diagnostic_table")


def generate_chinese_summary(results, output, population):
    lines = [
        "# 有限样本双谷与分支选择实验结果摘要", "",
        f"生成时间：{utc_now()}。", "",
        "## 实验对象与结论口径", "",
        "本实验直接对独立 Bartlett/Wishart Gram 对上的完整 full-ridge 条件风险取平均，并在平均导数曲线上搜索全部驻点。它不是对 $R_0+R_1/n_r$ 找最优，也没有平均单次数据集的最优权重。所有区间均为逐档点态区间。", "",
        f"高精度总体等高参数为 $t_*={population['t']}$；两处总体极小值为 $w_1={population['w1']}$、$w_2={population['w2']}$。$C_\\Delta={population['C_Delta']:.12g}$，因此渐近一阶预测为低权重分支最终胜出。任务椭球约束和总体双全局极小值检查均已通过。", "",
        "## 逐档结果", "",
        "| n_r | 找根/评估重复数 | 驻点/极小值数 | 双谷 bootstrap 支持率 | 两支位置 | n_r Delta_n | 一阶准确性 | 方向 |",
        "|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for result in results:
        row = summary_row(result)
        positions = "未识别两支" if row["w1"] is None else f"{row['w1']:.8g}, {row['w2']:.8g}"
        scaled = "—" if row["n_r_Delta"] is None else f"{row['n_r_Delta']:.8g}"
        accuracy = row["first_order_accuracy"] or "不适用"
        direction = row["selected_branch"] or "未分辨"
        lines.append(f"| {result['n_r']} | {result['find_replicates']}/{result['evaluation_replicates']} | "
                     f"{row['stationary_count']}/{row['local_minimum_count']} | {row['bootstrap_double_valley_rate']:.3f} | "
                     f"{positions} | {scaled} | {accuracy} | {direction} |")
    stable_ns = [result["n_r"] for result in results if result["shape"]["stable_double_valley"]]
    supported_ns = [result["n_r"] for result in results if result["branch_difference"] and result["branch_difference"]["first_order_accuracy"] == "support"]
    unsupported_ns = [result["n_r"] for result in results if result["branch_difference"] and result["branch_difference"]["first_order_accuracy"] == "unsupported"]
    lines.extend(["", "## 主要观察", "",
                  f"- 满足预设数值稳定性判据的已测样本量：{stable_ns if stable_ns else '无'}。这是数值稳定性判据，不是数学认证。",
                  f"- 一阶风险差区间整体落入 10% 容差带的已测样本量：{supported_ns if supported_ns else '无'}。",
                  f"- 一阶风险差区间与 10% 容差带完全不相交的已测样本量：{unsupported_ns if unsupported_ns else '无'}。这些结论只针对已测档位，不能外推为连续失败区间。"])
    for result in results:
        if result["position_analysis"]:
            for row in result["position_analysis"]:
                if not row["prediction_available"]:
                    lines.append(f"- n_r={result['n_r']} 的第 {row['branch']} 支一阶位置预测越出 [0,1]，未裁剪，记为位置近似不可用。")
        if result["escalation_reasons"] and result["budget_multiplier"] >= MAX_MULTIPLIER:
            lines.append(f"- n_r={result['n_r']} 达到 8 倍预算上限仍未完全分辨：{'; '.join(result['escalation_reasons'])}。")
    lines.extend(["", "## 不确定性与局限", "",
                  "独立第二阶段风险差区间条件于第一阶段锁定的位置；表中方向与一阶准确性使用二次位置误差扩张后的区间。另保存了第一阶段批次曲线重新优化得到的 bootstrap 风险差区间。",
                  "本实验只覆盖该二维高斯 full-ridge 模型内的一组固定参数，不构成一般有限样本形状分类，也不包含真实机器人任务。未识别到谷底不能解释为严格证明其不存在。", "",
                  "## 文件", "",
                  "- `population_check.json`：高精度总体条件与一阶理论预测。",
                  "- `implementation_checks.json`：向量化、导数、Wishart 矩及风险系数核查。",
                  "- `results.json`、`summary.csv`：全部档位及失败/未分辨状态。",
                  "- `batch_curves.npz`：批次风险、导数、bootstrap 和独立评估原始统计。",
                  "- `figures/`：风险曲线、谷底轨迹、分支风险差、诊断表的 PDF 与 PNG。",
                  "- `checkpoints/`：逐档、逐追加预算检查点。"])
    (output / "result_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def smoke_checks(model, output):
    rows = []
    grid = np.linspace(0, 1, GRID_FINE_SIZE)
    for n_r in [1000, 100000, 100000000]:
        started = time.time()
        rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, n_r, 0, 0]))
        grams_r, grams_h = sample_gram_pair(rng, 2048, n_r, model)
        risk = np.empty((2048, len(grid)))
        derivative = np.empty_like(risk)
        for start in range(0, len(grid), WEIGHT_CHUNK):
            stop = min(start + WEIGHT_CHUNK, len(grid))
            values = conditional_metrics(grams_r, grams_h, grid[start:stop], n_r, model, second=False)
            risk[:, start:stop], derivative[:, start:stop] = values
        elapsed = time.time() - started
        roots = find_sign_change_roots(grid, derivative.mean(axis=0))
        rows.append({"n_r": n_r, "gram_pairs": 2048, "grid_points": len(grid),
                     "elapsed_seconds": elapsed, "gram_weight_evaluations_per_second": 2048 * len(grid) / elapsed,
                     "root_count_for_implementation_diagnostic_only": len(roots),
                     "formal_conclusion_allowed": False})
    json_dump(output / "smoke_checks.json", {"purpose": "implementation and performance check only; not formal evidence",
                                             "rows": rows})
    return rows


def environment_document(code_hash):
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                          text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_sha = "unavailable"
    return (f"generated_utc={utc_now()}\nhostname={platform.node()}\nplatform={platform.platform()}\n"
            f"python={sys.version}\npython_executable={sys.executable}\nnumpy={np.__version__}\n"
            f"scipy={scipy.__version__}\nmatplotlib={matplotlib.__version__}\nmpmath={mp.__version__}\n"
            f"runner_sha256={code_hash}\ngit_sha={git_sha}\ncpu_count={os.cpu_count()}\n"
            f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}\n"
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}\n")


def load_population_model(population):
    model = {key: np.array(value) if key in ["a", "b", "beta", "delta"] else value
             for key, value in population["model_float"].items()}
    model["w1"], model["w2"] = float(population["w1"]), float(population["w2"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("finite_sample_branch_selection_2026-11-02"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    code_hash = runner_sha256()
    if args.report_only:
        population = json.loads((output / "population_check.json").read_text(encoding="utf-8"))
        implementation = json.loads((output / "implementation_checks.json").read_text(encoding="utf-8"))
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))["results"]
        consolidate_outputs(results, output, population, implementation)
        return
    freeze_protocol(output, code_hash)
    (output / "environment.txt").write_text(environment_document(code_hash), encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(output / "run.log", mode="a", encoding="utf-8"), logging.StreamHandler()])
    logging.info("Starting population and implementation checks; runner %s", code_hash)
    model, population = population_check()
    model["w1"], model["w2"] = float(population["w1"]), float(population["w2"])
    json_dump(output / "population_check.json", population)
    try:
        implementation = implementation_checks(model)
    except Exception:
        logging.exception("Implementation checks failed")
        raise
    json_dump(output / "implementation_checks.json", implementation)
    if not implementation["all_passed"]:
        raise AssertionError("One or more implementation checks failed; inspect implementation_checks.json")
    logging.info("Population and implementation checks passed")
    if args.check_only:
        return
    if args.smoke:
        rows = smoke_checks(model, output)
        for row in rows:
            logging.info("Smoke n_r=%d: %.3fs, %.3g evaluations/s", row["n_r"], row["elapsed_seconds"],
                         row["gram_weight_evaluations_per_second"])
        return
    if args.workers < 1 or args.workers > 32:
        raise ValueError("--workers must be between 1 and 32")
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_sample_size, n_r, model, output, population, args.resume): n_r
                   for n_r in N_VALUES}
        for future in as_completed(futures):
            n_r = futures[future]
            try:
                result = future.result()
            except Exception:
                logging.exception("Formal run failed for n_r=%d", n_r)
                raise
            results.append(result)
            results.sort(key=lambda row: row["n_r"])
            json_dump(output / "results.partial.json", {"generated_utc": utc_now(), "results": results})
            logging.info("Completed n_r=%d multiplier=%d stationary=%d minima=%d bootstrap=%.3f elapsed=%.1fs",
                         n_r, result["budget_multiplier"], result["shape"]["stationary_count"],
                         result["shape"]["local_minimum_count"], result["bootstrap"]["double_valley_support_rate"],
                         result["elapsed_seconds"])
    consolidate_outputs(results, output, population, implementation)
    logging.info("All formal outputs completed in %.1f seconds", time.time() - started)


if __name__ == "__main__":
    main()
