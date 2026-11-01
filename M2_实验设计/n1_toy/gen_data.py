"""高斯双域数据生成器(N1_实施卡 §2 口径)。

x_e = Σ_e^{1/2} u,u 标准高斯,Σ_e = diag(lam_e),lam_e,j = j^{-α_e};
y_r = ⟨β*, x_r⟩ + ε_r, ε_r ~ N(0, σ_r²);
y_h = ⟨β*+δ, x_h⟩ + ε_h, ε_h ~ N(0, σ_h²)。
"""
import numpy as np


def gen_domain(rng, n, lam_spec, beta, sig2):
    p = beta.shape[0]
    X = rng.normal(size=(n, p)) * np.sqrt(lam_spec)
    y = X @ beta + np.sqrt(sig2) * rng.normal(size=n)
    return X, y


def gen_two_domain(rng, n_r, n_h, lam_r, lam_h, beta_star, delta, sig_r2, sig_h2):
    """返回 (rob=(Xr, yr), hum=(Xh, yh));人域真参 β_h = β* + δ。"""
    rob = gen_domain(rng, n_r, lam_r, beta_star, sig_r2)
    hum = gen_domain(rng, n_h, lam_h, beta_star + delta, sig_h2)
    return rob, hum
