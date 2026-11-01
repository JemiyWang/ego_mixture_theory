"""Compare the proved second-order risk with the legacy full-ridge surrogate."""
import json
import sys
from pathlib import Path

import numpy as np

from verify_full_ridge_second_order import base_configurations

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "M2_实验设计" / "n1_toy"))
from theory import mse_t1b, risk_from_mse


def main():
    source = Path(__file__).with_name("full_ridge_second_order_risk_results.json")
    results = json.loads(source.read_text(encoding="utf-8"))
    configs = {name: config for name, config, _ in base_configurations()}
    rows = []
    for cell in results["cells"]:
        a, b, beta, delta, mu, sigma_r2, sigma_h2, _ = configs[cell["configuration"]]
        legacy = risk_from_mse(mse_t1b(
            cell["lambda"], a, b, delta, beta, beta + delta,
            cell["n_r"], cell["n_h"], sigma_r2, sigma_h2, mu), a)
        mc = cell["conditional_mc_mean"]
        rows.append({
            "configuration": cell["configuration"], "lambda": cell["lambda"],
            "n_r": cell["n_r"], "conditional_mc_mean": mc,
            "proved_second_order": cell["prediction"],
            "legacy_surrogate": float(legacy),
            "proved_minus_mc": cell["prediction"] - mc,
            "legacy_minus_mc": float(legacy - mc),
            "legacy_relative_error": float((legacy - mc) / mc),
        })
    output = {
        "note": "The legacy MSE_full,num is a numerical comparator, not the proved second-order object.",
        "legacy_median_absolute_relative_error": float(np.median([abs(row["legacy_relative_error"]) for row in rows])),
        "rows": rows,
    }
    target = Path(__file__).with_name("full_ridge_second_order_vs_legacy_surrogate.json")
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"legacy median absolute relative error: {output['legacy_median_absolute_relative_error']:.4%}")
    for row in rows:
        if row["configuration"] == "p20_spectral_A" and row["lambda"] == 0.3:
            print(row)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
