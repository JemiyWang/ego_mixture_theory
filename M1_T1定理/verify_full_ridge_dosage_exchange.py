"""Matched-spectrum dose laws and leading full-ridge exchange rates."""
import json
from pathlib import Path

import numpy as np

from verify_full_ridge_boundary import conditional_risk_and_gains
from verify_full_ridge_global_optimum import coefficient_derivatives
from verify_full_ridge_second_order import risk_coefficients


def matched_coefficients(scale, n_r=1000):
    sigma = np.diag([1.0, 0.5])
    beta = np.array([0.8, -0.3])
    delta = scale * np.array([0.1, -0.4])
    mu, sigma_r2, sigma_h2 = 0.2, 1.0, 4.0
    r0 = lambda w: float(risk_coefficients(w, sigma, sigma, beta, delta,
                                           mu, sigma_r2, sigma_h2, 6.0)[0])
    a0 = r0(0)
    a1 = coefficient_derivatives(0, sigma, sigma, beta, delta,
                                 mu, sigma_r2, sigma_h2, 6.0)[0]
    a2 = r0(1) - a0 - a1
    resolvent = np.linalg.inv(sigma + mu * np.eye(2))
    s = resolvent @ sigma @ resolvent
    noise = sigma_r2 * np.trace(s @ sigma)
    a1_leading = a1 - 2 * noise / n_r
    a2r_leading = a2 + noise / n_r
    b_leading = sigma_h2 * np.trace(s @ sigma)
    return {"sigma": sigma, "beta": beta, "delta": delta, "mu": mu,
            "sigma_r2": sigma_r2, "sigma_h2": sigma_h2, "n_r": n_r,
            "A0": a0, "A1": a1, "A2": a2, "C": noise,
            "A1_leading": a1_leading, "A2r_leading": a2r_leading,
            "B_leading": b_leading}


def leading_dose(c, n_h):
    a1, a2 = c["A1_leading"], c["A2r_leading"] + c["B_leading"] / n_h
    if a1 >= 0:
        return 0.0, 0.0
    w_unconstrained = -a1 / (2 * a2)
    if w_unconstrained < 1:
        return w_unconstrained, a1**2 / (4 * a2)
    return 1.0, -a1 - a2


def leading_direct(c, n_h):
    a0 = c["A0"] + c["C"] / c["n_r"]
    a1 = c["A1_leading"]
    a2 = c["A2r_leading"] + c["B_leading"] / n_h
    grid = np.linspace(0, 1, 10001)
    risks = a0 + a1 * grid + a2 * grid**2
    i = int(np.argmin(risks))
    return grid[i], a0 - risks[i]


def full_optimum(c, rho):
    if c["A1"] >= 0:
        w0, g0 = 0.0, 0.0
    else:
        w0 = min(1.0, -c["A1"] / (2 * c["A2"]))
        g0 = c["A1"]**2 / (4 * c["A2"]) if w0 < 1 else -c["A1"] - c["A2"]
    args = (c["sigma"], c["sigma"], c["beta"], c["delta"], c["mu"],
            c["sigma_r2"], c["sigma_h2"], rho)
    _, r10, _ = risk_coefficients(0, *args)
    _, r1w, _ = risk_coefficients(w0, *args)
    g1 = r10 - r1w
    scaled_gain = c["n_r"] * g0 + g1
    finite = scaled_gain > 0 and r10 > scaled_gain
    kappa0 = (scaled_gain / (rho * (r10 - scaled_gain))) if finite else (
        float("inf") if scaled_gain > 0 else 0.0)
    return w0, g0, g1, r10, finite, kappa0


def main():
    output = {"leading_dose": [], "full_optimum": [], "mc_gain": None}
    for name, scale in [("interior", 1.0), ("hits_boundary", 0.3)]:
        c = matched_coefficients(scale)
        threshold = (c["B_leading"] / (-c["A1_leading"] / 2 - c["A2r_leading"])
                     if -c["A1_leading"] / 2 - c["A2r_leading"] > 0 else None)
        print(f"{name}: boundary threshold={threshold}")
        for n_h in [30, 100, 300, 1000, 1779.1732077446363, 3000, 10000, 1000000]:
            w, g = leading_dose(c, n_h)
            wd, gd = leading_direct(c, n_h)
            output["leading_dose"].append({
                "configuration": name, "n_h": n_h, "w_formula": w,
                "gain_formula": g, "w_direct": wd, "gain_direct": gd,
                "gain_error": g - gd, "boundary_threshold": threshold})
        for rho in [1.0, 2.0, 6.0]:
            w0, g0, g1, r10, finite, kappa0 = full_optimum(c, rho)
            row = {"configuration": name, "rho": rho, "w0": w0,
                   "G0": g0, "G1": g1, "R1_baseline": r10,
                   "finite_exchange": finite, "kappa0": kappa0}
            output["full_optimum"].append(row)
            print(f"{name} rho={rho}: G0={g0:.8f}, G1={g1:.8f}, "
                  f"finite={finite}, kappa0={kappa0:.8g}")
    c_small = matched_coefficients(1.0, n_r=40)
    w0, g0, g1, r10, finite, kappa0 = full_optimum(c_small, 6.0)
    output["finite_example"] = {"configuration": "interior", "n_r": 40,
                                "rho": 6.0, "G0": g0, "G1": g1,
                                "R1_baseline": r10, "finite_exchange": finite,
                                "kappa": kappa0}
    print(f"finite example n_r=40: finite={finite}, kappa={kappa0:.8g}")
    c = matched_coefficients(1.0)
    w0, g0, g1, r10, finite, kappa0 = full_optimum(c, 6.0)
    baseline, means, gains, ses = conditional_risk_and_gains(
        np.random.default_rng(20261030), 50_000, 1000, np.array([0.0, w0]),
        c["sigma"], c["sigma"], c["beta"], c["delta"], c["mu"],
        c["sigma_r2"], c["sigma_h2"], 6.0)
    output["mc_gain"] = {"configuration": "interior", "n_r": 1000,
                         "n_h": 6000, "w0": w0,
                         "gain_prediction": g0 + g1 / 1000,
                         "gain_mc": float(gains[1]), "gain_se": float(ses[1])}
    print(f"MC gain: pred={g0 + g1 / 1000:.9f}, MC={gains[1]:.9f} +/- {ses[1]:.1e}")
    target = Path(__file__).with_name("full_ridge_dosage_exchange_results.json")
    def json_default(value):
        if isinstance(value, np.ndarray):
            return value.item() if value.ndim == 0 else value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value))
    target.write_text(json.dumps(output, indent=2, default=json_default), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
