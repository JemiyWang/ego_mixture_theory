"""v9：C1 组装的数值检验——n_r^crit 对 μ 的标度指数。

C1 预测（附录 2）：n_r^crit ∝ σ̃_r²/(|δ|_eff · μ^{1+(α_r−α_h+1)/α_r})（反对齐失配）。
检验：对谱系 A（α_r=2.0, α_h=0.9 → 预测指数 1+(2−0.9+1)/2 = 2.05），
扫 μ ∈ 对数网格，对每个 μ 用 v5 公式数值找 n_r^crit（R̄'(0⁺) 变号处），
对 log n_r^crit vs log μ 做线性拟合，比较斜率与 −2.05。
另附：n_r^crit 对 |δ| 的线性反比检验（指数 −1）。
"""
import numpy as np
import json

P = 1000
SIG_R2, SIG_H2 = 1.0, 4.0
S_SMOOTH = 1.0
jj = np.arange(1, P + 1, dtype=float)
beta = jj ** (-(S_SMOOTH + 0.5))
LAM_TEST = np.linspace(0, 0.02, 81)   # 边际符号用 λ→0 邻域

# 谱系 A；反对齐白色失配
lr = jj ** -2.0; lh = jj ** -0.9
delta_dir = -np.ones(P) / np.linalg.norm(np.ones(P))

def feed(spec, b):
    e = spec * b**2
    return e.sum() - e

def dR0(lr, lh, delta, n_r, n_h, mu):
    """有限差分版 R̄'(0⁺)（v5 公式）"""
    a, b = lr, lh
    Vr = (SIG_R2 + feed(lr, beta)) * a / n_r
    Vh = (SIG_H2 + feed(lh, beta + delta)) * b / n_h
    def R(lam):
        D = (1 - lam) * a + lam * b + mu
        N0 = lam * b * delta - mu * beta
        return float((a * (N0**2 + (1 - lam)**2 * Vr + lam**2 * Vh) / D**2).sum())
    R0 = R(0.0)
    # 在 λ 小邻域内找最小二乘斜率
    Rs = np.array([R(l) for l in LAM_TEST])
    slope = np.polyfit(LAM_TEST, Rs, 1)[0]
    return slope

def find_n_r_crit(delta, n_h_ratio, mu):
    """二分找 R̄'(0⁺) 的符号翻转点（n_r 对数网格）。"""
    lo, hi = np.log10(50.0), np.log10(3e5)
    f_lo = dR0(lr, lh, delta, 10**lo, int(10**lo * n_h_ratio), mu)
    f_hi = dR0(lr, lh, delta, 10**hi, int(10**hi * n_h_ratio), mu)
    if f_lo > 0: return 10**lo   # 全域有害
    if f_hi < 0: return np.inf   # 全域有益
    for _ in range(50):
        mid = (lo + hi) / 2
        if dR0(lr, lh, delta, 10**mid, int(10**mid * n_h_ratio), mu) > 0:
            hi = mid
        else:
            lo = mid
    return 10**((lo + hi) / 2)

# 检验 1：n_r^crit vs μ（‖δ‖=1.0，n_h/n_r=10）
mu_grid = np.logspace(-2.5, -0.5, 7)
nc_vs_mu = []
for mu in mu_grid:
    nc = find_n_r_crit(delta_dir * 1.0, 10.0, mu)
    nc_vs_mu.append(nc)
    print(f"μ={mu:.4f}: n_r^crit={nc:.0f}")

mask = np.isfinite(nc_vs_mu)
slope_mu = np.polyfit(np.log10(mu_grid[mask]), np.log10(nc_vs_mu)[mask], 1)[0]
pred_slope = -(1 + (2.0 - 0.9 + 1) / 2.0)
print(f"\nn_r^crit ∝ μ^{slope_mu:.3f}（C1 预测指数 {pred_slope:.3f}）")

# 检验 2：n_r^crit vs ‖δ‖（μ=0.05，n_h/n_r=10）
dnorm_grid = np.array([0.3, 0.5, 1.0, 2.0])
nc_vs_d = []
for dn in dnorm_grid:
    nc = find_n_r_crit(delta_dir * dn, 10.0, 0.05)
    nc_vs_d.append(nc)
    print(f"|δ|={dn}: n_r^crit={nc:.0f}")
slope_d = np.polyfit(np.log10(dnorm_grid), np.log10(nc_vs_d), 1)[0]
print(f"\nn_r^crit ∝ |δ|^{slope_d:.3f}（预测 −1.0，线性反比）")

json.dump(dict(mu_grid=list(map(float, mu_grid)), nc_vs_mu=list(map(float, nc_vs_mu)),
               slope_mu=float(slope_mu), pred_slope_mu=pred_slope,
               dnorm_grid=list(map(float, dnorm_grid)), nc_vs_d=list(map(float, nc_vs_d)),
               slope_d=float(slope_d)),
          open("/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l1_c1_scaling_check.json", "w"), indent=1)
print("saved l1_c1_scaling_check.json")
