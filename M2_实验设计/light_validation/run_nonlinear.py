"""Lightweight multi-seed nonlinear-arm validation against full ridge."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "M2_实验设计" / "n1_toy"))
from estimators import e2_full_ridge, e4_mlp, mc_risk_model
from gen_data import gen_two_domain

OUT = Path(__file__).with_name("outputs")
OUT.mkdir(exist_ok=True)
WEIGHTS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
L2_GRID = [0.0, 1e-4, 1e-3]
STEPS = 500
SEEDS = [11, 23, 37, 53, 71]


def configs():
    p2 = (np.array([1.0, 0.5]), np.array([1.2, 0.7]), np.array([0.8, -0.3]), 0.2)
    p20_j = np.arange(1, 21, dtype=float)
    a, b = p20_j**-2.0, p20_j**-0.9
    beta = p20_j**-1.5
    beta /= np.sqrt(np.sum(p20_j**2 * beta**2))
    delta = -p20_j**-1.0
    delta *= 0.5 / np.linalg.norm(delta)
    return {
        "p2_aligned": (p2[0], p2[1], p2[2], np.array([0.1, -0.4]), 1000, 2000, p2[3]),
        "p2_anti_n50": (p2[0], p2[1], p2[2], -np.array([0.1, -0.4]), 50, 300, p2[3]),
        "p2_anti_n70": (p2[0], p2[1], p2[2], -np.array([0.1, -0.4]), 70, 420, p2[3]),
        "p20_anti_n250": (a, b, beta, delta, 250, 1500, 0.05),
        "p20_anti_n700": (a, b, beta, delta, 700, 4200, 0.05),
    }


def mlp_risk(model, a, beta, rng):
    X = rng.normal(size=(20000, len(a))) * np.sqrt(a)
    return mc_risk_model(model, X, X @ beta)


def main():
    runs_path = OUT / "nonlinear_runs.json"
    calibration_path = OUT / "nonlinear_calibration.json"
    results = json.loads(runs_path.read_text()) if runs_path.exists() else []
    calibrations = json.loads(calibration_path.read_text()) if calibration_path.exists() else []
    completed = {row["configuration"] for row in calibrations}
    for name, (a, b, beta, delta, n_r, n_h, mu) in configs().items():
        if name in completed:
            print(name, "already complete; skipping", flush=True)
            continue
        baseline_scores = []
        for l2 in L2_GRID:
            risks = []
            for seed in SEEDS[:2]:
                rng = np.random.default_rng(seed + n_r)
                rob, hum = gen_two_domain(rng, n_r, n_h, a, b, beta, delta, 1.0, 4.0)
                model = e4_mlp(rob, hum, 0.0, l2, width=32, steps=STEPS, batch=256, seed=seed)
                risks.append(mlp_risk(model, a, beta, rng))
            baseline_scores.append(float(np.mean(risks)))
        full_baselines = []
        for seed in SEEDS[:2]:
            rng = np.random.default_rng(seed + n_r)
            rob, hum = gen_two_domain(rng, n_r, n_h, a, b, beta, delta, 1.0, 4.0)
            estimate = e2_full_ridge(rob, hum, 0.0, mu)
            full_baselines.append(float(a @ (estimate - beta)**2))
        full_baseline = float(np.mean(full_baselines))
        best_l2 = L2_GRID[int(np.argmin(np.abs(np.array(baseline_scores) - full_baseline)))]
        calibrations.append({"configuration": name, "full_baseline": full_baseline,
                             "mlp_baselines": dict(zip(L2_GRID, baseline_scores)),
                             "selected_l2": best_l2})
        print(name, "baseline", full_baseline, "MLP", baseline_scores, "l2", best_l2, flush=True)
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed + n_r)
            rob, hum = gen_two_domain(rng, n_r, n_h, a, b, beta, delta, 1.0, 4.0)
            for w in WEIGHTS:
                model = e4_mlp(rob, hum, w, best_l2, width=32, steps=STEPS,
                               batch=256, seed=seed)
                risk = mlp_risk(model, a, beta, rng)
                full = e2_full_ridge(rob, hum, w, mu)
                full_risk = float(a @ (full - beta)**2)
                results.append({"configuration": name, "seed": seed, "w": w,
                                "l2": best_l2, "mlp_risk": risk,
                                "full_risk": full_risk})
            print(name, "seed", seed, "done", flush=True)
        runs_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        calibration_path.write_text(json.dumps(calibrations, indent=2), encoding="utf-8")
    (OUT / "nonlinear_runs.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "nonlinear_calibration.json").write_text(json.dumps(calibrations, indent=2), encoding="utf-8")
    summary = []
    for name in configs():
        for w in WEIGHTS:
            rows = [r for r in results if r["configuration"] == name and r["w"] == w]
            mlp = np.array([r["mlp_risk"] for r in rows])
            full = np.array([r["full_risk"] for r in rows])
            summary.append({"configuration": name, "w": w,
                            "mlp_mean": float(mlp.mean()), "mlp_se": float(mlp.std(ddof=1) / np.sqrt(len(mlp))),
                            "full_mean": float(full.mean()), "full_se": float(full.std(ddof=1) / np.sqrt(len(full)))})
    (OUT / "nonlinear_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT / "nonlinear_summary.json")


if __name__ == "__main__":
    main()
