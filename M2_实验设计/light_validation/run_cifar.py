"""Lightweight real-feature two-domain mixing validation on CIFAR-10."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "M1_T1定理"))
sys.path.insert(0, str(ROOT / "M2_实验设计" / "n1_toy"))
from verify_full_ridge_zero_derivative import derivative_coefficients
from estimators import e2_full_ridge, e2_grams, e4_mlp, mc_risk_model

OUT = Path(__file__).with_name("outputs")
MU_GRID = [0.001, 0.01, 0.05, 0.1, 0.2]
W_GRID = [0.0, 0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 1.0]
L2_GRID = [0.0, 1e-4, 1e-3]
SEEDS = [11, 23, 37, 53, 71]


def sample(rng, X, y_class, classes, n, beta, noise):
    indices = np.flatnonzero(np.isin(y_class, classes))
    chosen = rng.choice(indices, size=n, replace=False)
    Xs = X[chosen]
    ys = Xs @ beta + np.sqrt(noise) * rng.normal(size=n)
    return Xs, ys


def split_fit_validation(rob, fraction=0.8):
    X, y = rob
    n_fit = int(len(y) * fraction)
    return (X[:n_fit], y[:n_fit]), (X[n_fit:], y[n_fit:])


def excess_risk(model_beta, X, beta):
    error = model_beta - beta
    return float(np.mean((X @ error)**2))


def validation_mse(model_beta, X, y):
    return float(np.mean((X @ model_beta - y)**2))


def fit_models(rob, hum, mu):
    grams = e2_grams(rob, hum)
    return {w: e2_full_ridge(rob, hum, w, mu, grams=grams) for w in W_GRID}


def main():
    data = np.load(OUT / "cifar_features.npz")
    X_train, X_test = data["X_train"].astype(np.float64), data["X_test"].astype(np.float64)
    y_train, y_test = data["y_train"], data["y_test"]
    p = X_train.shape[1]
    j = np.arange(1, p + 1, dtype=float)
    rng = np.random.default_rng(20261030)
    signs = rng.choice([-1.0, 1.0], size=10)
    beta = np.zeros(p)
    beta[:10] = signs * j[:10]**-0.5
    test_target = X_test[np.isin(y_test, np.arange(5))]
    sigma_r = np.cov(X_train[np.isin(y_train, np.arange(5))], rowvar=False)
    sigma_h = np.cov(X_train[np.isin(y_train, np.arange(5, 10))], rowvar=False)
    cells, mlp_cells = [], []
    for shape in ["lowfreq", "white", "highfreq"]:
        if shape == "lowfreq":
            delta = j**-1.0
        elif shape == "white":
            delta = np.ones(p)
        else:
            delta = (j / p)**2
        delta *= 0.5 / np.linalg.norm(delta)
        if beta @ delta > 0:
            delta = -delta
        reps = 200 if shape == "lowfreq" else 50
        for n_r in [50, 100, 200, 500, 1000, 2000]:
            n_h = 6 * n_r
            for rep in range(reps):
                seed = 1000 + rep + 10000 * n_r
                rng = np.random.default_rng(seed)
                rob_full = sample(rng, X_train, y_train, np.arange(5), n_r, beta, 1.0)
                hum = sample(rng, X_train, y_train, np.arange(5, 10), n_h, beta + delta, 4.0)
                rob, validation = split_fit_validation(rob_full)
                models, validation_scores = {}, {}
                for mu in MU_GRID:
                    models[mu] = fit_models(rob, hum, mu)
                    validation_scores[mu] = {
                        w: validation_mse(models[mu][w], validation[0], validation[1])
                        for w in W_GRID
                    }
                best_mu_by_w = {
                    w: min(MU_GRID, key=lambda mu: validation_scores[mu][w])
                    for w in W_GRID
                }
                best_w = min(
                    W_GRID,
                    key=lambda w: validation_scores[best_mu_by_w[w]][w],
                )
                mu = best_mu_by_w[best_w]
                test_risks = {
                    w: excess_risk(models[best_mu_by_w[w]][w], test_target, beta)
                    for w in W_GRID
                }
                baseline = test_risks[0.0]
                oracle_baseline = min(
                    excess_risk(models[candidate_mu][0.0], test_target, beta)
                    for candidate_mu in MU_GRID
                )
                oracle_selected = min(
                    excess_risk(models[candidate_mu][w], test_target, beta)
                    for candidate_mu in MU_GRID for w in W_GRID
                )
                row = {"shape": shape, "n_r": n_r, "n_h": n_h, "rep": rep,
                       "mu": mu, "validation_best_w": best_w,
                       "validation_best_mu_by_w": best_mu_by_w,
                       "test_risks": test_risks, "baseline_risk": baseline,
                       "selected_risk": test_risks[best_w],
                       "validation_gain": baseline - test_risks[best_w],
                       "oracle_baseline_risk": oracle_baseline,
                       "oracle_selected_risk": oracle_selected,
                       "oracle_gain": oracle_baseline - oracle_selected,
                       "fixed_half_gain": baseline - test_risks[0.5]}
                cells.append(row)
            print(shape, n_r, "done", flush=True)
        (OUT / f"cifar_{shape}_runs.json").write_text(json.dumps(cells, indent=2), encoding="utf-8")
        cells = []

    for shape in ["lowfreq"]:
        delta = j**-1.0
        delta *= 0.5 / np.linalg.norm(delta)
        if beta @ delta > 0:
            delta = -delta
        for n_r in [100, 500, 1000]:
            n_h = 6 * n_r
            baseline_l2 = []
            for l2 in L2_GRID:
                scores = []
                for seed in SEEDS[:2]:
                    rng = np.random.default_rng(seed + n_r)
                    rob_full = sample(rng, X_train, y_train, np.arange(5), n_r, beta, 1.0)
                    hum = sample(rng, X_train, y_train, np.arange(5, 10), n_h, beta + delta, 4.0)
                    rob, validation = split_fit_validation(rob_full)
                    model = e4_mlp(rob, hum, 0.0, l2, width=32, steps=500, batch=256, seed=seed)
                    scores.append(mc_risk_model(model, validation[0], validation[1]))
                baseline_l2.append(float(np.mean(scores)))
            selected_l2 = L2_GRID[int(np.argmin(baseline_l2))]
            for seed in SEEDS:
                rng = np.random.default_rng(1000 + seed + n_r)
                rob_full = sample(rng, X_train, y_train, np.arange(5), n_r, beta, 1.0)
                hum = sample(rng, X_train, y_train, np.arange(5, 10), n_h, beta + delta, 4.0)
                rob, validation = split_fit_validation(rob_full)
                for w in [0.0, 0.1, 0.3, 0.5, 1.0]:
                    model = e4_mlp(rob, hum, w, selected_l2, width=32, steps=500,
                                   batch=256, seed=seed)
                    mlp_cells.append({"n_r": n_r, "n_h": n_h, "seed": seed, "w": w,
                                      "l2": selected_l2,
                                      "risk": mc_risk_model(model, test_target, test_target @ beta),
                                      "baseline_l2_scores": dict(zip(L2_GRID, baseline_l2))})
            print("MLP", n_r, "done", flush=True)
    (OUT / "cifar_mlp_runs.json").write_text(json.dumps(mlp_cells, indent=2), encoding="utf-8")

    theory = {}
    for shape in ["lowfreq", "white", "highfreq"]:
        delta = {"lowfreq": j**-1.0, "white": np.ones(p), "highfreq": (j / p)**2}[shape]
        delta *= 0.5 / np.linalg.norm(delta)
        if beta @ delta > 0:
            delta = -delta
        theory[shape] = derivative_coefficients(sigma_r, sigma_h, beta, delta, 0.05, 1.0)
    (OUT / "cifar_theory.json").write_text(json.dumps(theory, indent=2), encoding="utf-8")
    print("wrote CIFAR outputs")


if __name__ == "__main__":
    main()
