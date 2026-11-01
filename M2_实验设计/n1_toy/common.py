"""N1 全局参数与共享构造(严格按 N1_实施卡 §2;谱系/δ 形状口径沿用 M1 l1_com_v5.py)。

与 v5 的两处有意差异(以实施卡为准,run 汇报中注明):
1. β* 归一到 Σ j^{2s}β_j² = 1(A2 椭球约束);v5 脚本未做此归一。
2. δ 归一到目标范数后取负号使 c=δᵀβ*<0(反对齐);v5 的 lowfreq 例为取正(对齐)。
"""
import numpy as np

# --- 全局参数(实施卡 §2,执行时不改) ---
P = 1000                       # 主维度(调试档 100)
S_SMOOTH = 1.0                 # β* 平滑度 s
SIG_R2, SIG_H2 = 1.0, 4.0      # 两域噪声方差
MU = 0.05                      # 岭参数(公式与闭式共用)
N_R, N_H = 1000, 6000          # 主样本量
SEED = 23                      # 全局种子基准(与 v3–v5 一致)
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0]  # 估计器 9 点
N_DENSE = 201                  # 理论线密网格点数
REPS_CLOSED = 200              # 粗批 E1/E2 reps(实施卡 §3,2026-10-25 修订 20→200:
                               # 20 reps 下 λ=0 风险标准误≈12.5%,2% 硬门误判率可观)
S0 = SEED                      # 粗批 E3/E4 单 seed:s0 = 全局基准 23

# E3/E4 超参(实施卡 §3;N1a/b/c/d/f 用显式 L2 项,μ_eff=l2 由构造精确成立)
L2_N1A = 0.05                  # N1a 取 l2=0.05(=μ)
E3_STEPS, E3_LR = 20_000, 3e-3     # 全批 GD,Adam,cosine 衰减,零初始化
E4_WIDTH, E4_STEPS, E4_LR, E4_BATCH = 512, 5_000, 1e-3, 512  # 默认初始化
N_TEST_MC = 50_000             # E4(非线性)测试风险的蒙特卡洛样本数

# 谱区制(实施卡 §2):(α_r, α_h)
SPECTRA = {
    "A": (2.0, 0.9),           # 大差 = 谱系 A
    "mid": (2.0, 1.4),         # 中差
    "zero": (2.0, 2.0),        # 零差
    "B": (1.2, 1.0),           # B 系稳健性
}
DELTA_SHAPES = ("lowfreq", "white", "highfreq")


def grid_dense():
    return np.linspace(0.0, 1.0, N_DENSE)


def coord_index(p=P):
    return np.arange(1, p + 1, dtype=float)


def make_beta_star(p=P, s=S_SMOOTH):
    """β_j = j^{-(s+1/2)},归一到 Σ j^{2s}β_j² = 1(v5 同式 + A2 椭球约束)。"""
    j = coord_index(p)
    b = j ** (-(s + 0.5))
    return b / np.sqrt(np.sum(j ** (2 * s) * b**2))


def make_spectra(alpha_r, alpha_h, p=P):
    """坐标谱 λe_j = j^{-α_e}(v5 口径,不归一)。"""
    j = coord_index(p)
    return j ** (-alpha_r), j ** (-alpha_h)


def make_delta(shape, norm, beta_star, p=P, anti_align=True):
    """δ 方向形状(实施卡 §2):低频 j^{-1};高频 (j/P)²;白色 j⁰。
    归一到目标范数后按需取负号,使 c=δᵀβ*<0(反对齐)或 >0(对齐)。"""
    j = coord_index(p)
    if shape == "lowfreq":
        d = j ** -1.0
    elif shape == "highfreq":
        d = (j / p) ** 2
    elif shape == "white":
        d = np.ones(p)
    else:
        raise ValueError(f"unknown delta shape: {shape}")
    d = d / np.linalg.norm(d) * norm
    c = float(d @ beta_star)
    if anti_align and c > 0:
        d = -d
    elif not anti_align and c < 0:
        d = -d
    return d
