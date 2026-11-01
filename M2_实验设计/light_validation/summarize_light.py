"""Aggregate lightweight nonlinear and CIFAR validation outputs."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).with_name("outputs")


def stats(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def main():
    nonlinear = json.loads((OUT / "nonlinear_summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name in zip(axes, ["p2_aligned", "p2_anti_n70"]):
        rows = [r for r in nonlinear if r["configuration"] == name]
        rows.sort(key=lambda r: r["w"])
        w = np.array([r["w"] for r in rows])
        ax.errorbar(w, [r["mlp_mean"] for r in rows],
                    yerr=[1.96 * r["mlp_se"] for r in rows], fmt="o-", label="MLP")
        ax.errorbar(w, [r["full_mean"] for r in rows],
                    yerr=[1.96 * r["full_se"] for r in rows], fmt="s--", label="full ridge")
        ax.set_title(name)
        ax.set_xlabel("w")
        ax.set_ylabel("target risk")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "nonlinear_risk_curves.png", dpi=180)
    plt.close(fig)

    cifar_summary = []
    for shape in ["lowfreq", "white", "highfreq"]:
        rows = json.loads((OUT / f"cifar_{shape}_runs.json").read_text())
        for n_r in sorted({r["n_r"] for r in rows}):
            subset = [r for r in rows if r["n_r"] == n_r]
            validation_gain, validation_se = stats([r["validation_gain"] for r in subset])
            oracle_gain, oracle_se = stats([r["oracle_gain"] for r in subset])
            half_gain, half_se = stats([r["fixed_half_gain"] for r in subset])
            best_w = {}
            for w in sorted({r["validation_best_w"] for r in subset}):
                best_w[str(w)] = sum(r["validation_best_w"] == w for r in subset) / len(subset)
            cifar_summary.append({"shape": shape, "n_r": n_r,
                                  "validation_gain": validation_gain,
                                  "validation_se": validation_se,
                                  "oracle_gain": oracle_gain, "oracle_se": oracle_se,
                                  "fixed_half_gain": half_gain, "fixed_half_se": half_se,
                                  "validation_best_w_frequency": best_w})
    (OUT / "cifar_summary.json").write_text(json.dumps(cifar_summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, shape in zip(axes, ["lowfreq", "white", "highfreq"]):
        rows = [r for r in cifar_summary if r["shape"] == shape]
        ns = np.array([r["n_r"] for r in rows])
        ax.errorbar(ns, [r["validation_gain"] for r in rows],
                    yerr=[1.96 * r["validation_se"] for r in rows], fmt="o-", label="validation-selected")
        ax.errorbar(ns, [r["fixed_half_gain"] for r in rows],
                    yerr=[1.96 * r["fixed_half_se"] for r in rows], fmt="s--", label="fixed w=0.5")
        ax.set_xscale("log")
        ax.set_title(shape)
        ax.set_xlabel("n_r")
        ax.set_ylabel("gain over target-only")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cifar_gain_curves.png", dpi=180)
    plt.close(fig)

    mlp = json.loads((OUT / "cifar_mlp_runs.json").read_text())
    mlp_summary = []
    for n_r in sorted({r["n_r"] for r in mlp}):
        for w in sorted({r["w"] for r in mlp}):
            rows = [r for r in mlp if r["n_r"] == n_r and r["w"] == w]
            mean, se = stats([r["risk"] for r in rows])
            mlp_summary.append({"n_r": n_r, "w": w, "risk_mean": mean, "risk_se": se})
    (OUT / "cifar_mlp_summary.json").write_text(json.dumps(mlp_summary, indent=2), encoding="utf-8")

    lines = ["# Lightweight validation summary", "",
             "All CIFAR results use channel-first RGB averaging followed by frozen grayscale PCA fitted only on training data. Labels are controlled synthetic regression labels, not the original CIFAR classes.", "",
             "## Nonlinear arm", "",
             "See `nonlinear_summary.json` and `nonlinear_risk_curves.png`. Five seeds are used with baseline L2 calibration; full-ridge and MLP use the same data within each arm.", "",
             "## Real-feature arm", "",
             "For each mixing weight, ridge regularization is selected by target validation-label MSE; the mixing weight is then selected by the same criterion. Neither the true regression vector nor test data participates in selection. CIFAR-MLP baseline L2 is likewise calibrated with validation labels.", "",
             "Oracle gains are separately computed test-grid diagnostics: both the target-only baseline and the mixed model minimize their respective test grids. They are not deployable selections.", "",
             "| n_r | low-frequency gain | white gain | high-frequency gain |",
             "|---:|---:|---:|---:|"]
    for n_r in sorted({r["n_r"] for r in cifar_summary}):
        by_shape = {r["shape"]: r for r in cifar_summary if r["n_r"] == n_r}
        lines.append("| {} | {:.5f} | {:.5f} | {:.5f} |".format(
            n_r, by_shape["lowfreq"]["validation_gain"],
            by_shape["white"]["validation_gain"],
            by_shape["highfreq"]["validation_gain"]))
    lines += ["",
              "See `cifar_summary.json`, `cifar_mlp_summary.json`, and `cifar_gain_curves.png` for means, standard errors, selected-weight frequencies, and figures.", "",
              "Evidence is limited to controlled full-ridge/small-MLP models and real CIFAR features. It is not external validation of a nonlinear robot policy."]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote summaries and figures")


if __name__ == "__main__":
    main()
