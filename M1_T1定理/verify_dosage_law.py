# Numerical verification for the dosage-law results of sec/X_suppl.tex, Sec. "Corollary 2".
# Checks over random configurations:
#   (i)   Corollary (interior minimizers): R'(0+) < 0 and R'(1-) > 0  =>  argmin is interior
#         (general spectra, per-coordinate aggregate, no unimodality used)
#   (ii)  Theorem (dosage law, matched spectra): gain g(n_h) = A1^2/(4 A2(n_h)),
#         lambda*(n_h) law, closed-form kappa_ex, and the n_h^eps inversion
#         all match a per-coordinate grid argmin + bisection solution of the defining equation.
import numpy as np

rng = np.random.default_rng(20261024)
LAM = np.linspace(0, 1, 4001)
LAM_DENSE = np.unique(np.concatenate([
    LAM, np.geomspace(1e-8, 1, 3000), 1 - np.geomspace(1e-8, 1, 3000)]))


def agg_risk(lam, a, b, vr, vh, u, w, mu, nh):
    """Per-coordinate aggregate risk on the grid lam (all coordinates retained)."""
    R = np.zeros_like(lam)
    mf = 2 * u**2 / nh  # mismatch fluctuation
    for j in range(len(a)):
        D0 = a[j] + mu
        Delta = b[j] - a[j]
        N = vr[j] * (1 - lam) ** 2 + vh[j] * lam ** 2 + (u[j] * lam - w[j]) ** 2 + mf[j] * lam ** 2
        R += a[j] * N / (D0 + Delta * lam) ** 2
    return R


def endpoint_slopes(a, b, vr, vh, u, w, mu, nh):
    """R'(0+) and R'(1-) from the endpoint formula of the suppl (Eq. endpoints)."""
    D0 = a + mu
    D1 = b + mu
    s = u * D0 + (b - a) * w
    n0 = vr + w**2
    n1 = -2 * (vr + u * w)
    L0 = n1 * D0 - 2 * (b - a) * n0
    L1 = 2 * ((vh + 2 * u**2 / nh) * D0 + (u - w) * s)
    return np.sum(a * L0 / D0**3), np.sum(a * L1 / D1**3)


# ---------- (i) interior-minimizer corollary, general spectra ----------
tested = confirmed = 0
for _ in range(60000):
    P = 32
    j = np.arange(1, P + 1)
    ar, ah = rng.uniform(0.5, 3), rng.uniform(0.5, 3)
    a = j ** (-ar); b = j ** (-ah)
    mu = 10 ** rng.uniform(-3, -0.3)
    nr = 10 ** rng.uniform(1.5, 4); nh = nr * 10 ** rng.uniform(-1, 1.5)
    s2r, s2h = 10 ** rng.uniform(-2, 0), 10 ** rng.uniform(-2, 0)
    beta = rng.normal(0, 1, P) * j ** (-1.0)
    delta = rng.normal(0, 1, P) * j ** (-0.5) * rng.uniform(0.05, 1.5)
    u = b * delta; w = mu * beta
    vr = s2r * a / nr; vh = s2h * b / nh
    d0, d1 = endpoint_slopes(a, b, vr, vh, u, w, mu, nh)
    if not (d0 < 0 and d1 > 0):
        continue
    tested += 1
    R = agg_risk(LAM_DENSE, a, b, vr, vh, u, w, mu, nh)
    if R.min() < min(R[0], R[-1]):
        confirmed += 1
print(f"(i)  interior-minimizer corollary: {confirmed}/{tested} confirmed (both endpoint signs strict)")

# ---------- (ii) dosage law, matched spectra ----------
err_g = err_lam = err_kex = err_eps = 0.0
n_cfg = 0
for _ in range(300):
    P = 64
    j = np.arange(1, P + 1)
    al = rng.uniform(0.5, 3.0)
    a = j ** (-al); b = a.copy()
    mu = 10 ** rng.uniform(-3, -0.3)
    nr = 10 ** rng.uniform(1.5, 4)
    s2r, s2h = 10 ** rng.uniform(-2, 0), 10 ** rng.uniform(-2, 0)
    beta = rng.normal(0, 1, P) * j ** (-1.0)
    delta = rng.normal(0, 1, P) * j ** (-0.5) * rng.uniform(0.05, 1.0)
    u = a * delta; w = mu * beta
    vr = s2r * a / nr
    D0 = a + mu
    n1 = -2 * (vr + u * w)
    A1 = np.sum(a * n1 / D0**2)
    if A1 >= 0:
        continue  # need strict marginal benefit
    A2r = np.sum(a * (vr + u**2) / D0**2)
    B = np.sum(a * (s2h * a + 2 * u**2) / D0**2)
    C = s2r * np.sum(a**2 / D0**2)
    n_cfg += 1
    for nh in np.geomspace(10, 1e5, 8):
        A2 = A2r + B / nh
        if -A1 >= 2 * A2:
            continue  # boundary case lambda* = 1, not covered by interior law
        # closed forms
        g_cf = A1**2 / (4 * A2)
        lam_cf = -A1 / (2 * A2)
        kex_cf = nr**2 * g_cf / (nh * (C - nr * g_cf)) if C > nr * g_cf else np.inf
        # per-coordinate ground truth, evaluated pointwise at the closed-form lambda*
        vh = s2h * b / nh
        R02 = agg_risk(np.array([0.0, lam_cf]), a, b, vr, vh, u, w, mu, nh)
        g_num = R02[0] - R02[1]           # gain at the closed-form minimizer
        lam_num = LAM[np.argmin(agg_risk(LAM, a, b, vr, vh, u, w, mu, nh))]
        err_g = max(err_g, abs(g_num - g_cf) / max(g_cf, 1e-30))
        err_lam = max(err_lam, abs(lam_num - lam_cf))
        # implicit kappa by bisection on the defining equation, robot side exact
        # R(0; n) = C/n + S with S = sum a w^2/D0^2
        S = np.sum(a * w**2 / D0**2)
        target = R02[0] - g_num
        lo, hi = 0.0, 1.0
        while C / (nr + hi * nh) + S > target and hi < 1e12:
            hi *= 2
        if hi < 1e12:
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if C / (nr + mid * nh) + S > target:
                    lo = mid
                else:
                    hi = mid
            kex_num = 0.5 * (lo + hi)
            if np.isfinite(kex_cf) and kex_cf > 1e-9:
                err_kex = max(err_kex, abs(kex_num - kex_cf) / kex_cf)
        # n_h^eps inversion at eps = 0.3 g(infty)
        ginf = A1**2 / (4 * A2r)
        eps = 0.3 * ginf
        nh_eps = 4 * eps * B / (A1**2 - 4 * eps * A2r)
        g_check = A1**2 / (4 * (A2r + B / nh_eps))
        err_eps = max(err_eps, abs(g_check - eps) / eps)
print(f"(ii) dosage law over {n_cfg} configs with A1<0 (interior n_h range):")
print(f"     max rel err gain g(n_h):        {err_g:.2e}")
print(f"     max abs err lambda*(n_h):       {err_lam:.2e}  (grid step 2.5e-4)")
print(f"     max rel err closed-form kex:    {err_kex:.2e}")
print(f"     max rel err n_h^eps inversion:  {err_eps:.2e}")

# ---------- monotonicity of lambda*(n_h) ----------
mono = True
for _ in range(200):
    P = 32
    j = np.arange(1, P + 1)
    a = j ** (-rng.uniform(0.5, 3.0))
    mu = 10 ** rng.uniform(-3, -0.3)
    nr = 10 ** rng.uniform(1.5, 4)
    s2r, s2h = 10 ** rng.uniform(-2, 0), 10 ** rng.uniform(-2, 0)
    beta = rng.normal(0, 1, P) * j ** (-1.0)
    delta = rng.normal(0, 1, P) * j ** (-0.5)
    u = a * delta; w = mu * beta
    vr = s2r * a / nr
    D0 = a + mu
    A1 = np.sum(a * (-2 * (vr + u * w)) / D0**2)
    if A1 >= 0:
        continue
    A2r = np.sum(a * (vr + u**2) / D0**2)
    B = np.sum(a * (s2h * a + 2 * u**2) / D0**2)
    nhs = np.geomspace(1, 1e6, 50)
    lams = -A1 / (2 * (A2r + B / nhs))
    lams = np.clip(lams, 0, 1)
    if np.any(np.diff(lams) < -1e-12):
        mono = False
print(f"     lambda*(n_h) nondecreasing in all tested configs: {mono}")
