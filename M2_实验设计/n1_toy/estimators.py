"""E1–E4 估计器阶梯(N1_实施卡 §3)。

E1 对角 ridge 闭式(v5 估计量式);E2 完整 ridge 闭式;E3 全批 Adam 线性;E4 小批量 Adam MLP。
逐样本权重约定:w_r = (1-λ)/n_r、w_h = λ/n_h(与 v5 公式同一组 wr/wh)。
"""
import numpy as np


def e1_diag_ridge(rob, hum, lam, mu):
    """E1 对角 ridge:β̂_j = (w_r·T_r + w_h·T_h)_j / (w_r·S_r + w_h·S_h + μ)_j,
    S_e,j = Σ_i x_e,ij², T_e,j = Σ_i x_e,ij·y_e,i(v5 估计量式)。"""
    (Xr, yr), (Xh, yh) = rob, hum
    n_r, n_h = Xr.shape[0], Xh.shape[0]
    wr, wh = (1.0 - lam) / n_r, lam / n_h
    num = wr * (Xr.T @ yr) + wh * (Xh.T @ yh)
    den = wr * (Xr**2).sum(0) + wh * (Xh**2).sum(0) + mu
    return num / den


def e2_grams(rob, hum):
    """预计算 G_e = X_eᵀX_e/n_e、b_e = X_eᵀy_e/n_e,供 E2 跨 λ 复用(数学等价)。"""
    (Xr, yr), (Xh, yh) = rob, hum
    n_r, n_h = Xr.shape[0], Xh.shape[0]
    return ((Xr.T @ Xr) / n_r, (Xh.T @ Xh) / n_h,
            (Xr.T @ yr) / n_r, (Xh.T @ yh) / n_h)


def e2_full_ridge(rob, hum, lam, mu, grams=None):
    """E2 完整 ridge:β̂ = (XᵀWX + μI)⁻¹XᵀWy,逐样本权重 w_r=(1-λ)/n_r、w_h=λ/n_h。
    XᵀWX = (1-λ)G_r + λG_h,XᵀWy = (1-λ)b_r + λb_h(G_e=X_eᵀX_e/n_e)。"""
    if grams is None:
        grams = e2_grams(rob, hum)
    G_r, G_h, b_r, b_h = grams
    A = (1.0 - lam) * G_r + lam * G_h
    A[np.diag_indices_from(A)] += mu
    return np.linalg.solve(A, (1.0 - lam) * b_r + lam * b_h)


def _stack_weighted(rob, hum, lam):
    """拼接两域并给逐样本权重 w_r=(1-λ)/n_r、w_h=λ/n_h(torch float64 张量)。"""
    import torch
    (Xr, yr), (Xh, yh) = rob, hum
    n_r, n_h = Xr.shape[0], Xh.shape[0]
    wr, wh = (1.0 - lam) / n_r, lam / n_h
    X = torch.from_numpy(np.vstack([Xr, Xh]))
    y = torch.from_numpy(np.concatenate([yr, yh]))
    w = torch.from_numpy(np.concatenate([np.full(n_r, wr), np.full(n_h, wh)]))
    return X, y, w


def e3_sgd_linear(rob, hum, lam, l2, steps=20_000, lr0=3e-3, seed=None):
    """E3 全批 Adam 线性(实施卡 §3,2026-10-25 修订):torch Linear(p,1,no bias),零初始化,
    全批 Adam lr=3e-3,20k 步 cosine 衰减;
    损失 = Σ_i w_i(y_i-x_iᵀβ)² + l2·‖β‖²(显式 L2 项,μ_eff=l2 由构造精确成立)。"""
    import torch
    if seed is not None:
        torch.manual_seed(seed)
    X, y, w = _stack_weighted(rob, hum, lam)
    model = torch.nn.Linear(X.shape[1], 1, bias=False, dtype=torch.float64)
    torch.nn.init.zeros_(model.weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        r = y - model(X).squeeze(-1)
        loss = (w * r * r).sum() + l2 * model.weight.pow(2).sum()
        loss.backward()
        opt.step()
        sched.step()
    return model.weight.detach().squeeze(0).numpy().copy()


def e4_mlp(rob, hum, lam, l2, width=512, steps=5_000, lr0=1e-3, batch=512, seed=None):
    """E4 两层 MLP(实施卡 §3):Linear(p,512)-ReLU-Linear(512,1),默认初始化,
    Adam lr=1e-3,batch 512(逐步有放回随机批),5k 步 cosine 衰减;
    损失 = n_tot·mean_batch(w_i r_i²) + l2·Σ‖W‖²(逐样本权重语义同 E3;
    n_tot 倍因子使批均值为全样本加权和的无偏估计;L2 只作用于权重矩阵,不含 bias)。
    返回训练好的 torch 模型(非线性,风险用 mc_risk_model 评估)。"""
    import torch
    if seed is not None:
        torch.manual_seed(seed)
    X, y, w = _stack_weighted(rob, hum, lam)
    n = X.shape[0]
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], width), torch.nn.ReLU(),
        torch.nn.Linear(width, 1)).to(torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=lr0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    weight_mats = [m.weight for m in model if isinstance(m, torch.nn.Linear)]
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,))
        r = y[idx] - model(X[idx]).squeeze(-1)
        loss = n * (w[idx] * r * r).mean() \
            + l2 * sum(W.pow(2).sum() for W in weight_mats)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return model


def e3_sgd_wd(rob, hum, lam, wd, steps=20_000, lr0=3e-3, seed=None, kind="adamw"):
    """E3(N1e 口径):显式 L2=0,weight decay 走优化器层。
    kind='adamw':AdamW decoupled wd(步进 −lr·wd·β);kind='adam':Adam L2-耦合
    (wd·β 加进梯度再过 Adam 归一化)。其余同 e3_sgd_linear(全批/20k/cosine/零初始化)。"""
    import torch
    if seed is not None:
        torch.manual_seed(seed)
    X, y, w = _stack_weighted(rob, hum, lam)
    model = torch.nn.Linear(X.shape[1], 1, bias=False, dtype=torch.float64)
    torch.nn.init.zeros_(model.weight)
    opt_cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[kind]
    opt = opt_cls(model.parameters(), lr=lr0, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        r = y - model(X).squeeze(-1)
        loss = (w * r * r).sum()
        loss.backward()
        opt.step()
        sched.step()
    return model.weight.detach().squeeze(0).numpy().copy()


def mc_risk_model(model, X_test, y_test_star):
    """非线性模型的机器人域测试风险:E_x[(f(x)-xᵀβ*)²](y_test_star 为无噪声标签)。"""
    import torch
    with torch.no_grad():
        pred = model(torch.from_numpy(X_test)).squeeze(-1).numpy()
    return float(np.mean((pred - y_test_star) ** 2))
