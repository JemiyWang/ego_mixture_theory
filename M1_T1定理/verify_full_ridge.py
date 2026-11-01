# v16: full-ridge deterministic equivalent (Theorem T1-b, fixed-p regime)
# Monte-Carlo risk of the FULL ridge estimator vs:
#   (a) new closed form:  sum_j a_j [ (1-l)^2 s_r^2 a_j/n_r + l^2 s_h^2 b_j/n_h
#                                     + (l b_j d_j - mu beta_j)^2 ] / D_j^2
#   (b) T1-a form (effective noise s^2 + F): should OVERESTIMATE by the F-terms
# Also: mismatch sandwich B_mis convergence, and error scaling ~ n^{-1/2}.
import numpy as np

rng = np.random.default_rng(20261026)


def run_config(p, n_r, n_h, lam, al_r, al_h, s_r, s_h, mu, dmag, K=400):
    j = np.arange(1, p + 1)
    a = j ** (-al_r); b = j ** (-al_h)
    beta = rng.normal(0, 1, p) * j ** (-1.0)
    delta = rng.normal(0, 1, p) * j ** (-0.5) * dmag
    w_r, w_h = (1 - lam) / n_r, lam / n_h
    D = (1 - lam) * a + lam * b + mu

    # closed forms
    F_r = np.sum(a * beta**2); F_h = np.sum(b * (beta + delta)**2)
    var_r = (1 - lam) ** 2 * s_r**2 * a / n_r
    var_h = lam**2 * s_h**2 * b / n_h
    bias = (lam * b * delta - mu * beta) ** 2
    cf_new = np.sum(a * (var_r + var_h + bias) / D**2)
    cf_t1a = np.sum(a * ((1 - lam) ** 2 * (s_r**2 + F_r) * a / n_r
                         + lam**2 * (s_h**2 + F_h) * b / n_h
                         + bias + 2 * lam**2 * (b * delta) ** 2 / n_h) / D**2)
    bmis_cf = np.sum(a * (lam * b * delta / D) ** 2)

    # Monte Carlo
    risks = np.empty(K)
    bmis = np.empty(K)
    for k in range(K):
        Xr = rng.normal(0, 1, (n_r, p)) * np.sqrt(a)
        Xh = rng.normal(0, 1, (n_h, p)) * np.sqrt(b)
        yr = Xr @ beta + s_r * rng.normal(0, 1, n_r)
        yh = Xh @ (beta + delta) + s_h * rng.normal(0, 1, n_h)
        A = w_r * (Xr.T @ Xr) + w_h * (Xh.T @ Xh) + mu * np.eye(p)
        rhs = w_r * (Xr.T @ yr) + w_h * (Xh.T @ yh)
        bh = np.linalg.solve(A, rhs)
        err = bh - beta
        risks[k] = np.sum(a * err**2)
        # sandwich-only: noiseless labels, beta* = 0 handling: mismatch term alone
        A0 = A
        v = w_h * (Xh.T @ (Xh @ delta))
        bmis[k] = np.sum(a * (np.linalg.solve(A0, v)) ** 2)
    return cf_new, cf_t1a, risks.mean(), bmis_cf, bmis.mean()


print("=== risk: MC vs new closed form vs T1-a form ===")
for (p, n_r, n_h, lam, alr, alh, dmag) in [
        (8, 200, 1000, 0.3, 2.0, 1.5, 0.3),
        (16, 500, 500, 0.5, 1.5, 1.0, 0.5),
        (32, 100, 2000, 0.2, 2.5, 1.5, 1.0),
        (16, 2000, 2000, 0.7, 2.0, 2.0, 0.3)]:
    cf_new, cf_t1a, mc, bmis_cf, bmis_mc = run_config(p, n_r, n_h, lam, alr, alh, 0.3, 0.6, 0.05, dmag)
    print(f"p={p:3d} n_r={n_r:5d} n_h={n_h:5d} lam={lam}: "
          f"MC {mc:.4e} | new {cf_new:.4e} ({(cf_new-mc)/mc:+.2%}) | "
          f"T1-a {cf_t1a:.4e} ({(cf_t1a-mc)/mc:+.2%}) | "
          f"B_mis cf {bmis_cf:.3e} vs MC {bmis_mc:.3e} ({(bmis_mc-bmis_cf)/bmis_cf:+.2%})")

print("=== error scaling in n (relative, should decay) ===")
for n_r in [100, 400, 1600, 6400]:
    cf_new, _, mc, _, _ = run_config(8, n_r, 2 * n_r, 0.4, 2.0, 1.5, 0.3, 0.6, 0.05, 0.5)
    print(f"n_r={n_r:5d}: rel err of new closed form {(cf_new-mc)/mc:+.2%}")
