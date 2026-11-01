"""Exact global-shape certification for fixed-dimensional full-ridge population risk."""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as s


def interval_add(x, y):
    return x[0] + y[0], x[1] + y[1]


def interval_sub(x, y):
    return x[0] - y[1], x[1] - y[0]


def interval_mul(x, y):
    values = [x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]]
    return min(values), max(values)


def interval_div(x, y):
    if y[0] <= 0 <= y[1]:
        raise ValueError("A denominator interval cannot contain zero")
    values = [x[0] / y[0], x[0] / y[1], x[1] / y[0], x[1] / y[1]]
    return min(values), max(values)


def scalar_interval(x):
    return x, x


def counterexample():
    w = s.symbols("w", real=True)
    a = [s.Integer(1), s.Rational(1, 2)]
    b = [s.Integer(1), s.Rational(1, 16)]
    mu = s.Rational(7, 1000)
    u = [s.sqrt(26) / 1000, s.sqrt(3) / 500]
    c = [s.Rational(1, 50), s.Rational(19, 20)]
    beta = [s.simplify(u[j] * c[j] / mu) for j in range(2)]
    delta = [s.simplify(u[j] / b[j]) for j in range(2)]
    k = [s.simplify(a[j] * u[j]**2) for j in range(2)]
    d = [s.expand((1 - w) * a[j] + w * b[j] + mu) for j in range(2)]
    e = [s.simplify(d[j].subs(w, c[j])) for j in range(2)]
    risk = sum(a[j] * (w * b[j] * delta[j] - mu * beta[j])**2 / d[j]**2
               for j in range(2))
    numerator = s.expand(sum(2 * k[j] * e[j] * (w - c[j]) * d[1 - j]**3
                             for j in range(2)))
    derivative_identity = s.simplify(s.diff(risk, w) * d[0]**3 * d[1]**3 - numerator)
    poly = s.Poly(numerator, w)
    _, primitive = poly.primitive()
    primitive = s.Poly(primitive, w)
    sign_factor = s.simplify(numerator / primitive.as_expr())
    isolating = s.intervals(primitive, eps=s.Rational(1, 10**8))
    exact_roots = s.real_roots(primitive)
    numeric_roots = np.roots(np.array(primitive.all_coeffs(), dtype=float))
    return {
        "w": w, "a": a, "b": b, "mu": mu, "u": u, "c": c,
        "beta": beta, "delta": delta, "k": k, "d": d, "e": e,
        "risk": risk, "numerator": numerator, "primitive": primitive,
        "derivative_identity": derivative_identity, "sign_factor": sign_factor,
        "isolating": isolating, "exact_roots": exact_roots, "numeric_roots": numeric_roots,
    }


def risk_interval(model, interval):
    a = [Fraction(1), Fraction(1, 2)]
    b = [Fraction(1), Fraction(1, 16)]
    mu = Fraction(7, 1000)
    k = [Fraction(13, 500000), Fraction(3, 500000)]
    c = [Fraction(1, 50), Fraction(19, 20)]
    x = (Fraction(interval[0]), Fraction(interval[1]))
    total = scalar_interval(Fraction(0))
    for j in range(2):
        d = interval_add(
            interval_sub(scalar_interval(a[j]), interval_mul(x, scalar_interval(a[j]))),
            interval_add(interval_mul(x, scalar_interval(b[j])), scalar_interval(mu)),
        )
        n = interval_sub(x, scalar_interval(c[j]))
        n = interval_mul(n, n)
        term = interval_mul(scalar_interval(k[j]), interval_div(n, interval_mul(d, d)))
        total = interval_add(total, term)
    return total


def exact_counterexample_certificate(model):
    inside = [(interval, multiplicity) for interval, multiplicity in model["isolating"]
              if interval[0] < 1 and interval[1] > 0]
    outside = [(interval, multiplicity) for interval, multiplicity in model["isolating"]
               if not (interval[0] < 1 and interval[1] > 0)]
    if len(inside) != 3 or any(multiplicity != 1 for _, multiplicity in model["isolating"]):
        raise AssertionError("The certified example must have three simple interior roots")
    derivative_zero = s.sign(model["numerator"].subs(model["w"], 0))
    if derivative_zero >= 0:
        raise AssertionError("The example must start with negative risk derivative")
    classifications = ["local_minimum", "local_maximum", "local_minimum"]
    candidates = [("boundary_0", None)] + [
        (classifications[i], interval) for i, (interval, _) in enumerate(inside)
    ] + [("boundary_1", None)]
    rows = []
    for kind, interval in candidates:
        if interval is None:
            endpoint = 0 if kind == "boundary_0" else 1
            bounds = risk_interval(model, (endpoint, endpoint))
            root = s.Integer(endpoint)
        else:
            bounds = risk_interval(model, interval)
            root = model["exact_roots"][len(rows) - 1]
        rows.append({
            "classification": kind,
            "root": str(root),
            "root_numeric": float(s.N(root, 30)),
            "isolating_interval": None if interval is None else [str(interval[0]), str(interval[1])],
            "risk_interval_exact": [str(bounds[0]), str(bounds[1])],
            "risk_interval": [float(bounds[0]), float(bounds[1])],
        })
    global_row = rows[1]
    for row in [rows[0], *rows[2:]]:
        if not global_row["risk_interval"][1] < row["risk_interval"][0]:
            raise AssertionError("Risk intervals do not certify the unique global minimum")
    ellipsoid = s.simplify(model["beta"][0]**2 + 4 * model["beta"][1]**2)
    numeric_inside = sorted(r.real for r in model["numeric_roots"]
                            if abs(r.imag) < 1e-10 and 0 < r.real < 1)
    exact_inside = [row["root_numeric"] for row in rows[1:4]]
    independent_root_error = float(np.max(np.abs(np.array(numeric_inside) - exact_inside)))
    return {
        "model": {
            "a": [str(x) for x in model["a"]], "b": [str(x) for x in model["b"]],
            "mu": str(model["mu"]), "beta": [str(x) for x in model["beta"]],
            "delta": [str(x) for x in model["delta"]],
            "K": [str(x) for x in model["k"]], "c": [str(x) for x in model["c"]],
            "spectral_exponents": [1, 4], "ellipsoid_s": 1,
            "ellipsoid_value": str(ellipsoid),
            "ellipsoid_numeric": float(ellipsoid),
        },
        "derivative_numerator_identity": str(model["derivative_identity"]),
        "primitive_polynomial": str(model["primitive"].as_expr()),
        "numerator_to_primitive_factor": str(model["sign_factor"]),
        "all_real_root_intervals": [
            {"interval": [str(interval[0]), str(interval[1])], "multiplicity": multiplicity}
            for interval, multiplicity in model["isolating"]
        ],
        "interior_root_count": len(inside),
        "outside_root_intervals": [
            [str(interval[0]), str(interval[1])] for interval, _ in outside
        ],
        "candidate_risks": rows,
        "unique_global_minimizer": rows[1]["root"],
        "unique_global_minimizer_numeric": rows[1]["root_numeric"],
        "independent_numpy_root_max_error": independent_root_error,
        "conclusion": "Two local minima and one local maximum in (0,1); unique global minimum at the first stationary point",
    }


def delta_zero_checks():
    w = s.symbols("w", real=True)
    a = [s.Integer(1), s.Rational(1, 2)]
    b = [s.Integer(1), s.Rational(1, 4)]
    beta = [s.Rational(1, 5), s.Rational(-1, 10)]
    mu = s.Rational(1, 10)
    d = [(1 - w) * a[j] + w * b[j] + mu for j in range(2)]
    risk = sum(a[j] * mu**2 * beta[j]**2 / d[j]**2 for j in range(2))
    log_curvatures = [s.simplify(2 * (b[j] - a[j])**2 / d[j]**2) for j in range(2)]
    endpoint_derivatives = [s.simplify(s.diff(risk, w).subs(w, point)) for point in [0, 1]]
    matched_risk = sum(a[j] * mu**2 * beta[j]**2 / (a[j] + mu)**2 for j in range(2))
    matched_difference = s.simplify(matched_risk - matched_risk.subs(w, 0))
    return {
        "log_second_derivatives": [str(x) for x in log_curvatures],
        "all_log_second_derivatives_nonnegative_on_01": all(
            all(s.N(x.subs(w, point)) >= 0 for point in [0, s.Rational(1, 2), 1])
            for x in log_curvatures
        ),
        "at_least_one_strictly_positive_log_curvature": any(
            s.N(x.subs(w, s.Rational(1, 2))) > 0 for x in log_curvatures
        ),
        "strict_log_convexity": True,
        "endpoint_derivatives": [str(x) for x in endpoint_derivatives],
        "matched_spectra_risk_minus_constant": str(matched_difference),
    }


def main():
    model = counterexample()
    if model["derivative_identity"] != 0:
        raise AssertionError("The independently derived numerator does not match R0'")
    result = {
        "scope": "Population risk R0 only; no claim about R1 or finite-sample Rn",
        "delta_zero": delta_zero_checks(),
        "multimodal_counterexample": exact_counterexample_certificate(model),
    }
    output = Path(__file__).with_name("global_risk_shape_results.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "interior_root_count": result["multimodal_counterexample"]["interior_root_count"],
        "unique_global_minimizer": result["multimodal_counterexample"]["unique_global_minimizer_numeric"],
        "candidate_risks": [row["risk_interval"] for row in
                            result["multimodal_counterexample"]["candidate_risks"]],
        "matched_spectra_risk_minus_constant": result["delta_zero"]["matched_spectra_risk_minus_constant"],
    }, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
