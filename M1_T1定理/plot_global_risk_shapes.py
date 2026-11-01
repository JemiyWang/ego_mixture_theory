"""Plot certified population-risk shapes and the one-parameter branch diagram."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from verify_global_risk_shape_p2_family import numeric_branches


def risk_at(scale, weights):
    a = np.array([1.0, 0.5])
    b = np.array([1.0, 1 / 16])
    mu = 0.007
    c = np.array([0.02, 0.95])
    k = np.array([0.000026, 0.000006 * scale])
    d = (1 - weights[:, None]) * a + weights[:, None] * b + mu
    return np.sum(k * (weights[:, None] - c)**2 / d**2, axis=1)


def main():
    output = Path(__file__).resolve().parents[1] / "paper_cvpr" / "fig" / "global_risk_shapes.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    weights = np.linspace(0, 1, 1200)
    risks = risk_at(1.0, weights)
    axes[0].plot(weights, risks, color="#1764ab", linewidth=2.2, label="$R_0(w)$ at $t=1$")
    roots, values = numeric_branches(1.0)
    colors = ["#2ca02c", "#d62728", "#2ca02c"]
    labels = ["global min", "local max", "second local min"]
    for root, value, color, label in zip(roots, values, colors, labels):
        axes[0].scatter([root], [value], s=48, color=color, zorder=5, label=label)
    axes[0].set(xlabel="mixing weight $w$", ylabel="population risk",
                title="Two valleys in a certified example")
    axes[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="upper center")

    first, second, maximum, global_values = [], [], [], []
    scales = np.linspace(0.75, 1.35, 301)
    for scale in scales:
        roots, values = numeric_branches(scale)
        endpoints = risk_at(scale, np.array([0.0, 1.0]))
        if len(roots) == 1:
            first.append(values[0])
            second.append(np.nan)
            maximum.append(np.nan)
        else:
            first.append(values[0])
            second.append(values[-1])
            maximum.append(values[1] if len(values) == 3 else np.nan)
        global_values.append(min(np.min(values), np.min(endpoints)))
    first, second, maximum, global_values = map(np.array, [first, second, maximum, global_values])
    axes[1].plot(scales, first, color="#1764ab", linewidth=1.5, label="first local-min branch")
    axes[1].plot(scales, second, color="#2ca02c", linewidth=1.5, label="second local-min branch")
    axes[1].plot(scales, maximum, color="#d62728", linewidth=1.2,
                 linestyle="--", label="local-max branch")
    axes[1].plot(scales, global_values, color="black", linewidth=2.2,
                 linestyle=":", label="global minimum")
    for x, label in [(0.9375848094, "$t_1$"), (1.0608592591, "$t_*$"),
                     (1.2003419385, "$t_2$")]:
        axes[1].axvline(x, color="gray", linewidth=1.0, alpha=0.8)
        axes[1].text(x, axes[1].get_ylim()[1], label, ha="center", va="top", fontsize=9)
    axes[1].set(xlabel="relative coordinate weight $t$", ylabel="stationary risk",
                title="Valley creation, switching, and annihilation")
    axes[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
