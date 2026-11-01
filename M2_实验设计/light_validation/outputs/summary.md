# Lightweight validation summary

All CIFAR results use channel-first RGB averaging followed by frozen grayscale PCA fitted only on training data. Labels are controlled synthetic regression labels, not the original CIFAR classes.

## Nonlinear arm

See `nonlinear_summary.json` and `nonlinear_risk_curves.png`. Five seeds are used with baseline L2 calibration; full-ridge and MLP use the same data within each arm.

## Real-feature arm

For each mixing weight, ridge regularization is selected by target validation-label MSE; the mixing weight is then selected by the same criterion. Neither the true regression vector nor test data participates in selection. CIFAR-MLP baseline L2 is likewise calibrated with validation labels.

Oracle gains are separately computed test-grid diagnostics: both the target-only baseline and the mixed model minimize their respective test grids. They are not deployable selections.

| n_r | low-frequency gain | white gain | high-frequency gain |
|---:|---:|---:|---:|
| 50 | 0.00260 | 0.16193 | 0.30420 |
| 100 | -0.00486 | 0.06037 | 0.16370 |
| 200 | -0.00739 | 0.02461 | 0.06313 |
| 500 | -0.00377 | 0.00254 | 0.02271 |
| 1000 | -0.00146 | -0.00063 | 0.00778 |
| 2000 | -0.00076 | -0.00016 | 0.00156 |

See `cifar_summary.json`, `cifar_mlp_summary.json`, and `cifar_gain_curves.png` for means, standard errors, selected-weight frequencies, and figures.

Evidence is limited to controlled full-ridge/small-MLP models and real CIFAR features. It is not external validation of a nonlinear robot policy.