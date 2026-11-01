"""核验脚本 3：新种子独立复现 v0.2 两个配置 + v0.3 四配置的割线曲率偏置分解。"""
import numpy as np

print("[J] 新种子独立复现 v0.2 两个配置的 dR_emp（验证脚本实现）")
rng = np.random.default_rng(2024)
P, SIG, MU = 200, 1.0, 0.05
beta_star = rng.normal(size=P)
beta_star /= np.linalg.norm(beta_star)

def emp_secant(n_r, n_h, s2, delta, h, reps):
    beta_h = beta_star + delta
    secs = []
    for rep in range(reps):
        Xr = rng.normal(size=(n_r, P))
        yr = Xr @ beta_star + SIG * rng.normal(size=n_r)
        Xh = rng.normal(scale=np.sqrt(s2), size=(n_h, P))
        yh = Xh @ beta_h + SIG * rng.normal(size=n_h)
        Sr, Sh = Xr.T @ Xr, Xh.T @ Xh
        br, bh = Xr.T @ yr, Xh.T @ yh
        risks = []
        for lam in (0.0, h):
            wr, wh = (1 - lam) / n_r, lam / n_h
            A = wr * Sr + wh * Sh + MU * np.eye(P)
            b = wr * br + wh * bh
            diff = np.linalg.solve(A, b) - beta_star
            risks.append(diff @ diff)
        secs.append((risks[1] - risks[0]) / h)
    secs = np.array(secs)
    return secs.mean(), secs.std(ddof=1) / np.sqrt(reps)

for (n_r, s2, mode, json_val) in [(50, 2.0, "aligned", -55.647),
                                  (200, 2.0, "aligned", -47.017)]:
    delta = 0.5 * beta_star if mode == "aligned" else -0.5 * beta_star
    m, se = emp_secant(n_r, 5 * n_r, s2, delta, 0.005, 200)
    print(f"n_r={n_r} s2={s2} {mode}: 复现 {m:+.2f} ± {se:.2f} (SE) | JSON {json_val:+.2f}")

print()
print("[J/H] v0.3 四配置：经验-理论差 的割线曲率偏置分解")
print("     (割线 ≈ R'(0) + h/2·R''(0)；h=0.001)")
p, mu, B, dphi = 200, 0.05, 1.0, 4.0
configs = [(2000, -2.0, -0.004, +0.004), (2000, 0.0, -0.487, -0.358),
           (10000, -2.0, +0.313, +0.288), (10000, 0.0, -0.062, -0.075)]
s2 = 2.0
for n_r, c, emp, th in configs:
    # 理论曲线的 R''(0)：对 (Nb+Nv)/D^2 求二阶导（用 sympy 结果手算太慢，数值二阶差分）
    D = lambda l: (1 - l) + l * s2 + mu
    R = lambda l: ((l**2 * s2**2 * dphi - 2 * l * s2 * mu * c + mu**2 * B)
                   + p * ((1 - l)**2 / n_r + l**2 * s2 / (5 * n_r))) / D(l)**2
    e = 1e-6
    Rpp = (R(e) - 2 * R(0) + R(-e)) / e**2 if e > 0 else None
    # R 在 0 右侧定义；用右差分
    Rpp = 2 * (R(e) - R(0) - e * (R(e) - R(0)) / e) / e**2  # placeholder
    Rpp = (R(2e-6) - 2 * R(1e-6) + R(0.0)) / 1e-12
    curv_bias = 0.001 / 2 * Rpp
    print(f"n_r={n_r:>6} c={c:+.0f}: emp-th={emp-th:+.4f} | 曲率偏置≈{curv_bias:+.4f} "
          f"| 残差(涨落+噪声)≈{emp-th-curv_bias:+.4f}")
