"""Exact bifurcation and global-minimum checks for a two-coordinate risk family."""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as s
from scipy.optimize import brentq

from verify_global_risk_shape import (
    interval_add,
    interval_div,
    interval_mul,
    interval_sub,
    scalar_interval,
)


def family_model():
    w, t = s.symbols("w t", real=True)
    a = [s.Integer(1), s.Rational(1, 2)]
    b = [s.Integer(1), s.Rational(1, 16)]
    mu = s.Rational(7, 1000)
    c = [s.Rational(1, 50), s.Rational(19, 20)]
    k = [s.Rational(13, 500000), s.Rational(3, 500000) * t]
    d = [(1 - w) * a[j] + w * b[j] + mu for j in range(2)]
    e = [d[j].subs(w, c[j]) for j in range(2)]
    poly = s.expand(sum(2 * k[j] * e[j] * (w - c[j]) * d[1 - j]**3
                        for j in range(2)))
    discriminant = s.factor(s.discriminant(poly, w))
    return {"w": w, "t": t, "a": a, "b": b, "mu": mu, "c": c,
            "k": k, "d": d, "poly": poly, "discriminant": discriminant}


def risk_interval(scale, interval):
    a = [Fraction(1), Fraction(1, 2)]
    b = [Fraction(1), Fraction(1, 16)]
    mu = Fraction(7, 1000)
    c = [Fraction(1, 50), Fraction(19, 20)]
    k1 = Fraction(13, 500000)
    k2_base = Fraction(3, 500000)
    if isinstance(scale, Fraction):
        scale = scalar_interval(scale)
    x = (Fraction(interval[0]), Fraction(interval[1]))
    total = scalar_interval(Fraction(0))
    for j in range(2):
        d = interval_add(
            interval_sub(scalar_interval(a[j]), interval_mul(x, scalar_interval(a[j]))),
            interval_add(interval_mul(x, scalar_interval(b[j])), scalar_interval(mu)),
        )
        n = interval_sub(x, scalar_interval(c[j]))
        n = interval_mul(n, n)
        k = scalar_interval(k1) if j == 0 else interval_mul(scale, scalar_interval(k2_base))
        term = interval_mul(k, interval_div(n, interval_mul(d, d)))
        total = interval_add(total, term)
    return total


def root_intervals(model, scale):
    poly = model["poly"].subs(model["t"], scale)
    return s.intervals(poly, eps=s.Rational(1, 10**10))


def candidate_risks(model, scale):
    roots = root_intervals(model, scale)
    inside = [(interval, multiplicity) for interval, multiplicity in roots
              if interval[0] < 1 and interval[1] > 0]
    rows = [{"kind": "boundary", "risk": risk_interval(scale, (0, 0))}]
    for interval, _ in inside:
        rows.append({"kind": "stationary", "interval": interval,
                     "risk": risk_interval(scale, interval)})
    rows.append({"kind": "boundary", "risk": risk_interval(scale, (1, 1))})
    return rows


def unique_global(rows, index):
    target = rows[index]["risk"]
    return all(target[1] < row["risk"][0] for i, row in enumerate(rows) if i != index)


def numeric_branches(scale):
    model = family_model()
    poly = model["poly"].subs(model["t"], scale)
    coefficients = np.array(s.Poly(poly, model["w"]).all_coeffs(), dtype=float)
    roots = np.roots(coefficients)
    weights = np.array(sorted(r.real for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < 1))
    a = np.array([1.0, 0.5])
    b = np.array([1.0, 1 / 16])
    mu = 0.007
    c = np.array([0.02, 0.95])
    k = np.array([0.000026, 0.000006 * scale])
    d = (1 - weights[:, None]) * a + weights[:, None] * b + mu
    risks = np.sum(k * (weights[:, None] - c)**2 / d**2, axis=1)
    return weights, risks


def root_count_certificate(model):
    critical = [root for root in s.real_roots(model["discriminant"]) if root > 0]
    samples = [s.Rational(1, 2), s.Integer(1), s.Rational(5, 4)]
    sample_rows = []
    for scale in samples:
        count = s.count_roots(model["poly"].subs(model["t"], scale), 0, 1)
        sample_rows.append({"scale": str(scale), "interior_root_count": int(count)})
    critical_rows = []
    for root in critical:
        numeric_scale = float(root)
        weights, risks = numeric_branches(numeric_scale)
        critical_rows.append({
            "scale_exact": str(root), "scale_numeric": numeric_scale,
            "numeric_stationary_weights": weights.tolist(),
            "numeric_stationary_risks": risks.tolist(),
        })
    if len(critical) != 2:
        raise AssertionError("The family must have two positive discriminant roots")
    if [row["interior_root_count"] for row in sample_rows] != [1, 3, 1]:
        raise AssertionError("Unexpected root counts in the three parameter regions")
    endpoint_polynomials = [s.factor(model["poly"].subs(model["w"], point))
                            for point in [0, 1]]
    endpoint_roots = [s.real_roots(poly) for poly in endpoint_polynomials]
    if any(any(root >= 0 for root in roots) for roots in endpoint_roots):
        raise AssertionError("An endpoint crossing would require an additional parameter partition")
    leading = s.factor(s.LC(model["poly"], model["w"]))
    if leading == 0:
        raise AssertionError("The quartic must not degenerate in this family")
    return {
        "discriminant": str(model["discriminant"]),
        "leading_coefficient": str(leading),
        "endpoint_polynomials": [str(poly) for poly in endpoint_polynomials],
        "endpoint_roots": [[str(root) for root in roots] for roots in endpoint_roots],
        "positive_critical_parameters": critical_rows,
        "region_samples": sample_rows,
        "classification": "1 simple interior root; birth of a double root; 3 simple roots; merger to a double root; 1 simple root",
    }


def global_tie_certificate(model):
    left = Fraction(106085925, 100000000)
    right = Fraction(106085926, 100000000)
    bracket = (left, right)
    left_rows = candidate_risks(model, left)
    right_rows = candidate_risks(model, right)
    if len(left_rows) != 5 or len(right_rows) != 5:
        raise AssertionError("The tie bracket must contain three interior roots")
    left_difference = interval_sub(left_rows[1]["risk"], left_rows[3]["risk"])
    right_difference = interval_sub(right_rows[1]["risk"], right_rows[3]["risk"])
    if not left_difference[1] < 0 < right_difference[0]:
        raise AssertionError("The branch-risk difference does not change sign")
    left_roots = root_intervals(model, left)
    right_roots = root_intervals(model, right)
    left_inside = [interval for interval, _ in left_roots if interval[0] < 1]
    right_inside = [interval for interval, _ in right_roots if interval[0] < 1]
    branches = []
    for i in range(3):
        interval = (min(left_inside[i][0], right_inside[i][0]),
                    max(left_inside[i][1], right_inside[i][1]))
        branches.append(risk_interval(bracket, interval))
    lower_endpoint = risk_interval(bracket, (0, 0))
    upper_endpoint = risk_interval(bracket, (1, 1))
    for branch_index in [0, 2]:
        if not branches[branch_index][1] < branches[1][0]:
            raise AssertionError("The middle stationary point remains above both minima")
        if not branches[branch_index][1] < lower_endpoint[0]:
            raise AssertionError("The left boundary remains above both minima")
        if not branches[branch_index][1] < upper_endpoint[0]:
            raise AssertionError("The right boundary remains above both minima")
    weights, risks = numeric_branches(1.0608592591022064)
    return {
        "bracket_exact": [str(left), str(right)],
        "left_branch_minus_right_at_left": [str(x) for x in left_difference],
        "left_branch_minus_right_at_right": [str(x) for x in right_difference],
        "branch_risk_intervals_over_bracket": [[float(x) for x in branch] for branch in branches],
        "endpoint_risk_intervals_over_bracket": [[float(x) for x in lower_endpoint],
                                                  [float(x) for x in upper_endpoint]],
        "existence_proof": "Three simple continuous root branches; branch-risk difference changes sign; all other candidates stay strictly above throughout the bracket",
        "estimated_tie_parameter": 1.0608592591022064,
        "estimated_tie_scale_of_base_K2": True,
        "independent_numeric_weights": weights.tolist(),
        "independent_numeric_risks": risks.tolist(),
        "conclusion": "There exists a parameter in the bracket with exactly two global minima at the first and third stationary points",
    }


def main():
    model = family_model()
    root_counts = root_count_certificate(model)
    below = candidate_risks(model, Fraction(1))
    above = candidate_risks(model, Fraction(11, 10))
    if not unique_global(below, 1) or not unique_global(above, 3):
        raise AssertionError("The global minimizer does not switch branches")
    result = {
        "scope": "One-parameter two-coordinate full-ridge population risk family",
        "root_bifurcations": root_counts,
        "global_branch_switch": {
            "below_tie_unique_global": "first stationary point at scale 1",
            "above_tie_unique_global": "third stationary point at scale 11/10",
        },
        "double_global_minimum": global_tie_certificate(model),
    }
    output = Path(__file__).with_name("global_risk_shape_p2_family_results.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "critical_parameters": root_counts["positive_critical_parameters"],
        "root_counts": root_counts["region_samples"],
        "double_global_minimum": result["double_global_minimum"]["estimated_tie_parameter"],
    }, indent=2, default=str))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
