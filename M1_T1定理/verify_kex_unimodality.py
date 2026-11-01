# Numerical verification for the unimodality results of sec/X_suppl.tex, Sec. "Corollary 2".
# Checks, over random parameter draws:
#   (i)   Lemma (affine derivative numerator): MSE'(lambda) = (p + q*lambda)/D^3
#   (ii)  Proposition (no interior maxima): sign pattern L(0) > 0 > L(1) never occurs;
#         interior roots of L are minima (q > 0)
#   (iii) Theorem (matched spectra): aggregate is quadratic with coefficients A1, A2
#         and the closed-form lambda* matches the numerical argmin
#   (iv)  Proposition (convexity region): condition (convreg) implies MSE'' >= 0 on [0,1]
#   (v)   multimodality search on the aggregate risk (power-law and adversarial spectra)
import numpy as np

rng = np.random.default_rng(20261023)
NH = 100.0


def mse(lam, vr, vh, u, w, D0, Delta, nh=NH):
    N = vr * (1 - lam) ** 2 + vh * lam ** 2 + (u * lam - w) ** 2 + 2 * u * u / nh * lam ** 2
    D = D0 + Delta * lam
    return N / D ** 2


def coeffs(vr, vh, u, w, D0, Delta, nh=NH):
    n0 = vr + w * w
    n1 = -2 * (vr + u * w)
    n2 = vr + vh + u * u * (1 + 2 / nh)
    p = n1 * D0 - 2 * Delta * n0
    q = 2 * n2 * D0 - Delta * n1
    return n0, n1, n2, p, q


def draw():
    vr = 10 ** rng.uniform(-3, 1)
    vh = 10 ** rng.uniform(-3, 1)
    u = rng.normal(0, 2)
    w = rng.normal(0, 1)
    a = 10 ** rng.uniform(-2, 0.5)
    b = 10 ** rng.uniform(-2, 0.5)
    mu = 10 ** rng.uniform(-3, -0.3)
    D0, Delta = a + mu, b - a
    return vr, vh, u, w, D0, Delta


lam_g = np.linspace(0, 1, 2001)
h = 1e-6
N_DRAWS = 200000

err_deriv = 0.0
bad_pattern = 0          # L(0) > 0 > L(1) occurrences (must be 0)
interior_roots = 0
worst_second = np.inf    # MSE'' at interior roots (must be > 0)
conv_viol = 0            # convexity condition satisfied but MSE'' < 0 somewhere (must be 0)
conv_tested = 0

for _ in range(N_DRAWS):
    vr, vh, u, w, D0, Delta = draw()
    n0, n1, n2, p, q = coeffs(vr, vh, u, w, D0, Delta)
    L0, L1 = p, p + q

    # (i) derivative identity
    lam = rng.uniform(0.02, 0.98)
    fd = (mse(lam + h, vr, vh, u, w, D0, Delta) - mse(lam - h, vr, vh, u, w, D0, Delta)) / (2 * h)
    ana = (p + q * lam) / (D0 + Delta * lam) ** 3
    err_deriv = max(err_deriv, abs(fd - ana) / max(1e-12, abs(fd)))

    # (ii) forbidden sign pattern
    if L0 > 0 and L1 < 0:
        bad_pattern += 1
    if L0 < 0 < L1:
        interior_roots += 1
        root = -p / q
        assert 0 < root < 1
        sec = (q * (D0 + Delta * root) - 3 * Delta * (p + q * root)) / (D0 + Delta * root) ** 4
        worst_second = min(worst_second, sec)

    # (iv) convexity sufficient condition
    Dmax = max(D0, D0 + Delta)
    Dmin = min(D0, D0 + Delta)
    if 2 * abs(Delta) * (abs(n1) + 2 * n2) * Dmax <= n2 * Dmin ** 2:
        conv_tested += 1
        vals = mse(lam_g, vr, vh, u, w, D0, Delta)
        sec_num = np.gradient(np.gradient(vals, lam_g), lam_g)
        if sec_num[5:-5].min() < -1e-6:
            conv_viol += 1

print(f"(i)  max relative error of derivative identity: {err_deriv:.2e}")
print(f"(ii) forbidden sign pattern L(0)>0>L(1) count: {bad_pattern} / {N_DRAWS}")
print(f"     interior roots: {interior_roots}, worst second derivative at root: {worst_second:.3e}")
print(f"(iv) convexity condition tested on {conv_tested} draws, violations: {conv_viol}")

# (iii) matched spectra: aggregate quadratic + closed-form lambda*
P = 50
a = np.array([10 ** rng.uniform(-2, 0) for _ in range(P)])
a = np.sort(a)[::-1]
mu = 10 ** rng.uniform(-2.5, -0.5)
vr = 10 ** rng.uniform(-3, 0, P) * a
vh = 10 ** rng.uniform(-3, 0, P) * a
u = rng.normal(0, 1, P) * a
w = mu * rng.normal(0, 1, P)
n0 = vr + w ** 2
n1 = -2 * (vr + u * w)
n2 = vr + vh + u ** 2 * (1 + 2 / NH)
D0 = a + mu
R = np.zeros_like(lam_g)
for j in range(P):
    R += a[j] * (vr[j] * (1 - lam_g) ** 2 + vh[j] * lam_g ** 2 + (u[j] * lam_g - w[j]) ** 2
                 + 2 * u[j] ** 2 / NH * lam_g ** 2) / D0[j] ** 2
A2 = np.sum(a * n2 / D0 ** 2)
A1 = np.sum(a * n1 / D0 ** 2)
A0 = np.sum(a * n0 / D0 ** 2)
quad_err = np.max(np.abs(R - (A2 * lam_g ** 2 + A1 * lam_g + A0)))
lam_star = np.clip(-A1 / (2 * A2), 0, 1)
lam_num = lam_g[np.argmin(R)]
print(f"(iii) matched spectra: quadratic fit residual {quad_err:.2e}, "
      f"lambda* closed form {lam_star:.4f} vs argmin {lam_num:.4f}")

# (v) multimodality search on the aggregate
def count_local_minima(vals):
    d = np.sign(np.diff(vals))
    d = d[d != 0]
    return int(np.sum((d[:-1] < 0) & (d[1:] > 0)))

found = 0
for trial in range(4000):
    P = 64
    if trial % 2 == 0:  # power-law spectra
        j = np.arange(1, P + 1)
        ar, ah = rng.uniform(0.5, 3), rng.uniform(0.5, 3)
        a = j ** (-ar); b = j ** (-ah)
    else:               # adversarial random spectra
        a = 10 ** rng.uniform(-3, 0, P); a.sort(); a = a[::-1]
        b = 10 ** rng.uniform(-3, 0, P); b.sort(); b = b[::-1]
    mu = 10 ** rng.uniform(-3, -0.5)
    nr = 10 ** rng.uniform(1, 5); nh = nr * 10 ** rng.uniform(-1, 2)
    beta = rng.normal(0, 1, P) * np.arange(1, P + 1) ** (-1.0)
    kind = trial % 3
    delta = np.zeros(P)
    if kind == 0:   delta = rng.normal(0, 1, P)
    elif kind == 1: delta[:4] = rng.normal(0, 2, 4)
    else:           delta[-8:] = rng.normal(0, 2, 8)
    sr, sh = 10 ** rng.uniform(-2, 0), 10 ** rng.uniform(-2, 0)
    Fr = np.sum(a * beta ** 2); Fh = np.sum(b * (beta + delta) ** 2)
    R = np.zeros_like(lam_g)
    for j in range(P):
        vr = (sr ** 2 + Fr) * a[j] / nr
        vh = (sh ** 2 + Fh) * b[j] / nh
        u = b[j] * delta[j]; w = mu * beta[j]
        D0 = a[j] + mu; Delta = b[j] - a[j]
        N = vr * (1 - lam_g) ** 2 + vh * lam_g ** 2 + (u * lam_g - w) ** 2 + 2 * u ** 2 / nh * lam_g ** 2
        R += a[j] * N / (D0 + Delta * lam_g) ** 2
    k = count_local_minima(R)
    if k > 1:
        found += 1
        if found == 1:
            print(f"     first multimodal instance at trial {trial}: {k} local minima")
print(f"(v)  multimodal aggregates found: {found} / 4000")
