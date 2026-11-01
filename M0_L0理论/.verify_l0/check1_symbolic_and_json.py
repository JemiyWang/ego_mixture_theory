"""L0 文档核验脚本 1：sympy 符号求导交叉验证 + JSON 数据复算。"""
import json
import numpy as np
import sympy as sp

print("=" * 70)
print("[D] sympy 交叉验证 §3.2 的 R'(0+) 商法则代数")
print("=" * 70)
lam = sp.symbols('lambda')
s2, mu, c, B, Dl, p, sr2, sh2, nr, nh = sp.symbols(
    's2 mu c B Delta p sr2 sh2 nr nh')
D = (1 - lam) + lam * s2 + mu
Nb = lam**2 * s2**2 * Dl - 2 * lam * s2 * mu * c + mu**2 * B     # bias^2 分子
Nv = p * (sr2 * (1 - lam)**2 / nr + sh2 * lam**2 * s2 / nh)       # Var 分子
R = (Nb + Nv) / D**2
dR0 = sp.diff(R, lam).subs(lam, 0)
doc = (-2 * s2 * mu * c * (1 + mu) - 2 * mu**2 * B * (s2 - 1)
       - 2 * p * sr2 * (s2 + mu) / nr) / (1 + mu)**3
print("sympy d/dλ R(0) - 文档公式 化简结果:", sp.simplify(dR0 - doc))
# 分解三项，逐一对照
dR_bias = sp.simplify(sp.diff(Nb / D**2, lam).subs(lam, 0))
dR_var = sp.simplify(sp.diff(Nv / D**2, lam).subs(lam, 0))
print("bias 部分:", sp.simplify(dR_bias),
      " == 文档 [-2s2·mu·c(1+mu) - 2mu^2 B(s2-1)]/(1+mu)^3 ?",
      sp.simplify(dR_bias - (-2*s2*mu*c*(1+mu) - 2*mu**2*B*(s2-1))/(1+mu)**3) == 0)
print("var  部分:", sp.simplify(dR_var),
      " == 文档 [-2 p sr2 (s2+mu)/nr]/(1+mu)^3 ?",
      sp.simplify(dR_var - (-2*p*sr2*(s2+mu)/nr)/(1+mu)**3) == 0)
# mu->0 极限
print("mu→0 极限:", sp.simplify(dR0.subs(mu, 0)), "(文档称 = -2 p sr2 s2 / nr)")

print()
print("=" * 70)
print("[C/G] 用 §3.1 公式从零复算 R_theory 曲线并与 JSON 对比")
print("=" * 70)


def theory_curve(lams, s2_, c_, dphi, B_, n_r, n_h, p_=200, mu_=0.05, sr=1.0, sh=1.0):
    Dv = (1 - lams) + lams * s2_ + mu_
    bias2 = (lams**2 * s2_**2 * dphi - 2 * lams * s2_ * mu_ * c_ + mu_**2 * B_) / Dv**2
    var = p_ * (sr**2 * (1 - lams)**2 / (n_r * Dv**2)
                + sh**2 * lams**2 * s2_ / (n_h * Dv**2))
    return bias2 + var


def dR_closed(s2_, c_, B_, n_r, p_=200, mu_=0.05, sr=1.0):
    return (-2 * s2_ * mu_ * c_ * (1 + mu_) - 2 * mu_**2 * B_ * (s2_ - 1)
            - 2 * p_ * sr**2 * (s2_ + mu_) / n_r) / (1 + mu_)**3


for fname in ["l0_mc_results.json", "l0_mc_v03_results.json"]:
    d = json.load(open(f"/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/{fname}"))
    lams = np.array(d["lambdas"])
    worst_R, worst_dR = 0.0, 0.0
    for r in d["results"]:
        dphi = {"aligned": 0.25, "anti": 0.25, "orth": 0.25}[r["mode"]] if fname == "l0_mc_results.json" else 4.0
        Rt = theory_curve(lams, r["s2"], r["c"], dphi, 1.0, r["n_r"], r["n_h"],
                          p_=d["p"], mu_=d["mu"], sr=d["sig_r"], sh=d["sig_h"])
        worst_R = max(worst_R, float(np.max(np.abs(Rt - np.array(r["R_theory"])))))
        dRt = dR_closed(r["s2"], r["c"], 1.0, r["n_r"], p_=d["p"], mu_=d["mu"], sr=d["sig_r"])
        worst_dR = max(worst_dR, abs(dRt - r["dR_theory"]))
        # dR_emp 与 R_emp 数组的内部一致性
        sec = (r["R_emp"][1] - r["R_emp"][0]) / lams[1]
        assert abs(sec - r["dR_emp"]) < 1e-12, (fname, r["mode"], r["n_r"])
    print(f"{fname}: max|R_theory 复算-JSON| = {worst_R:.3e}, "
          f"max|dR_theory 复算-JSON| = {worst_dR:.3e}, dR_emp 与 R_emp 数组自洽 OK")

print()
print("=" * 70)
print("[G] v0.2 抽 3 配置手算 dR_theory（逐步展开）")
print("=" * 70)
cases = [("n_r=50,s2=0.5,aligned,c=+0.5", 50, 0.5, 0.5),
         ("n_r=50,s2=2,anti,c=-0.5", 50, 2.0, -0.5),
         ("n_r=2000,s2=2,orth,c=0", 2000, 2.0, 0.0)]
for name, n_r, s2_, c_ in cases:
    t1 = -2 * s2_ * 0.05 * c_ * 1.05
    t2 = -2 * 0.05**2 * 1 * (s2_ - 1)
    t3 = -2 * 200 * 1 * (s2_ + 0.05) / n_r
    tot = (t1 + t2 + t3) / 1.05**3
    print(f"{name}: 交叉项={t1:+.6f} 谱宽项={t2:+.6f} 方差项={t3:+.6f} "
          f"分子和={t1+t2+t3:+.6f} /1.05^3={1.05**3:.6f} -> {tot:+.6f}")

d2 = json.load(open("/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l0_mc_results.json"))
for r in d2["results"]:
    if (r["n_r"], r["s2"], r["mode"]) in [(50, 0.5, "aligned"), (50, 2.0, "anti"), (2000, 2.0, "orth")]:
        print(f"  JSON: n_r={r['n_r']} s2={r['s2']} {r['mode']}: dR_theory={r['dR_theory']:+.6f}")

print()
print("=" * 70)
print("[G] v0.2 摘要 claim 2 复核：经验/理论导数比值表")
print("=" * 70)
for r in d2["results"]:
    ratio = r["dR_emp"] / r["dR_theory"]
    print(f"n_r={r['n_r']:>5} s2={r['s2']:>3} {r['mode']:>8}: emp={r['dR_emp']:>10.3f} "
          f"th={r['dR_theory']:>10.3f} 比值={ratio:6.2f}")

print()
print("=" * 70)
print("[F/H] v0.3 全表复算 + n_r^crit")
print("=" * 70)
d3 = json.load(open("/mnt/mnt/data/wjm/ego/doc/ego_mixture_theory/l0_mc_v03_results.json"))
for r in d3["results"]:
    t1 = -2 * r["s2"] * 0.05 * r["c"] * 1.05
    t2 = -2 * 0.05**2 * (r["s2"] - 1)
    t3 = -2 * 200 * (r["s2"] + 0.05) / r["n_r"]
    tot = (t1 + t2 + t3) / 1.05**3
    print(f"n_r={r['n_r']:>6} {r['mode']:>4} c={r['c']:+.1f}: 手算 dR={tot:+.6f} | "
          f"JSON theory={r['dR_theory']:+.6f} emp={r['dR_emp']:+.6f} | "
          f"lam* emp/th={r['lam_star_emp']}/{r['lam_star_theory']}")

# n_r^crit: p=200, sr=1, s2=2, mu=0.05, c=-2, B=1
num = 200 * 1 * (2 + 0.05)
den_full = -2 * 0.05 * (-2) * 1.05 - 0.05**2 * 1 * (2 - 1)
den_negpart = -2 * 0.05 * (-2) * 1.05 - 0.05**2 * 1 * min(2 - 1, 0)  # (s2-1)_- 标准负部解读
print(f"\n分子 p sr2 (s2+mu) = {num}")
print(f"分母(不截断) = {den_full}  -> n_r^crit = {num/den_full:.2f}  (文档附录B: 1976)")
print(f"分母(按 (s2-1)_- 负部函数标准解读) = {den_negpart} -> n_r^crit = {num/den_negpart:.2f}")
print(f"文档附录B算术: 200*2.05/(0.21-0.0025) = {200*2.05/(0.21-0.0025):.2f}")
