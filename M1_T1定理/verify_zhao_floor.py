# Numerical verification for the label-shift floor (boundary proposition) of sec/X_suppl.tex.
#   Main loop (200k draws): bounds (i), (ii), (iii) on exact risks.
#   Side loop (500 draws): mixture-floor closed form vs brute-force constrained minimum.
import numpy as np

rng = np.random.default_rng(20261025)

viol_i = viol_ii = viol_iii = 0
ND = 200000

for _ in range(ND):
    m_r, m_h = rng.normal(0, 1, 2)
    eps = abs(m_r - m_h)
    mu_r, mu_h = rng.normal(0, 2, 2)
    s2r, s2h = 10 ** rng.uniform(-2, 0, 2)
    lam = rng.uniform(0.05, 0.95)
    c = rng.normal(0, 1)
    L = abs(c)
    dmu = mu_h - mu_r

    R_r = (c * m_r - mu_r) ** 2 + c**2 + s2r
    R_h = (c * m_h - mu_h) ** 2 + c**2 + s2h
    floor = max(abs(dmu) - L * eps, 0) ** 2

    if R_r + R_h < floor / 2 - 1e-10:
        viol_i += 1
    if (1 - lam) * R_r + lam * R_h < lam * (1 - lam) * floor - 1e-10:
        viol_ii += 1

    dlt = np.sign(dmu) * min(L * eps, abs(dmu))
    nu_r = (1 - lam) * mu_r + lam * (mu_h - dlt)
    Rr_star = (nu_r - mu_r) ** 2
    pred = lam**2 * floor
    if abs(Rr_star - pred) > 1e-8 * max(1, pred):
        viol_iii += 1

print(f"(i)   sum-bound violations:        {viol_i} / {ND}")
print(f"(ii)  mixture-floor violations:    {viol_ii} / {ND}")
print(f"(iii) constrained-optimal target-floor mismatches: {viol_iii} / {ND}")

# side loop: closed form vs brute force
err_cf = 0.0
NB = 500
for _ in range(NB):
    m_r, m_h = rng.normal(0, 1, 2)
    eps = abs(m_r - m_h)
    mu_r, mu_h = rng.normal(0, 2, 2)
    lam = rng.uniform(0.05, 0.95)
    Leps = 10 ** rng.uniform(-2, 1)
    dmu = mu_h - mu_r
    floor = max(abs(dmu) - Leps, 0) ** 2
    cf = lam * (1 - lam) * floor
    nu_r = np.linspace(mu_r - 3 * abs(dmu) - 3, mu_r + 3 * abs(dmu) + 3, 2001)
    dlts = np.linspace(-Leps, Leps, 2001)
    A = (1 - lam) * (nu_r - mu_r) ** 2
    Bmat = lam * (nu_r[None, :] + dlts[:, None] - mu_h) ** 2
    best = (A[None, :] + Bmat).min()
    if cf > 1e-6:
        err_cf = max(err_cf, abs(best - cf) / cf)
    else:
        err_cf = max(err_cf, abs(best - cf))
print(f"(ii') closed form vs brute force ({NB} draws), max err: {err_cf:.2e}")
