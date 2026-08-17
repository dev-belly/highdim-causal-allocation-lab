"""
数据生成过程（DGP）

围绕分层随机试验构造可控蒙特卡洛模拟，支持调节：
- 样本量 n
- 协变量维度 p
- 协变量相关结构（独立 / 可交换 / AR(1) / 分块相关）
- 处理效应异质性（同质 / 异质）
- 处理分配机制（完全随机 / 分层随机）
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder


def generate_covariates(n, p, corr_type="independent", rho=0.5, block_size=None, rng=None):
    """
    生成 n x p 的协变量矩阵，支持多种相关结构。

    Parameters
    ----------
    n : int
        样本量
    p : int
        协变量维度
    corr_type : str
        "independent" | "exchangeable" | "ar1" | "block"
    rho : float
        相关系数强度
    block_size : int, optional
        分块相关时每个块的大小
    rng : np.random.Generator

    Returns
    -------
    X : ndarray, shape (n, p)
    """
    rng = rng or np.random.default_rng()

    if corr_type == "independent":
        Sigma = np.eye(p)
    elif corr_type == "exchangeable":
        Sigma = (1 - rho) * np.eye(p) + rho * np.ones((p, p))
    elif corr_type == "ar1":
        indices = np.arange(p)
        Sigma = rho ** np.abs(indices[:, None] - indices[None, :])
    elif corr_type == "block":
        block_size = block_size or max(1, p // 4)
        Sigma = np.eye(p)
        for i in range(0, p, block_size):
            j = min(i + block_size, p)
            Sigma[i:j, i:j] = (1 - rho) * np.eye(j - i) + rho * np.ones((j - i, j - i))
    else:
        raise ValueError(f"Unknown corr_type: {corr_type}")

    # 保证正定
    Sigma += 1e-6 * np.eye(p)
    X = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
    return X


def generate_strata(X, n_strata=4, rng=None):
    """
    基于协变量的第一个主维度构造离散分层标签。

    Returns
    -------
    strata : ndarray, shape (n,)
    """
    rng = rng or np.random.default_rng()
    # 使用第一个协变量分位数分层
    q = np.quantile(X[:, 0], np.linspace(0, 1, n_strata + 1))
    q[-1] += 1e-8  # 包含右侧
    strata = np.digitize(X[:, 0], q[1:])
    return strata


def assign_treatment(strata, treat_prob=0.5, assignment="complete", rng=None):
    """
    处理分配。

    Parameters
    ----------
    strata : ndarray
        分层标签
    treat_prob : float
        每层内处理概率
    assignment : str
        "complete" 完全随机；"stratified" 分层随机
    rng : np.random.Generator

    Returns
    -------
    W : ndarray, shape (n,)
    """
    rng = rng or np.random.default_rng()
    n = len(strata)
    W = np.zeros(n, dtype=int)

    if assignment == "complete":
        n_treat = int(np.round(n * treat_prob))
        W[:] = 0
        idx = rng.choice(n, size=n_treat, replace=False)
        W[idx] = 1
    elif assignment == "stratified":
        for s in np.unique(strata):
            mask = strata == s
            n_s = mask.sum()
            n_treat_s = int(np.round(n_s * treat_prob))
            idx_s = rng.choice(np.where(mask)[0], size=n_treat_s, replace=False)
            W[idx_s] = 1
    else:
        raise ValueError(f"Unknown assignment: {assignment}")

    return W


def generate_coef(n, p, hetero_type="homogeneous", sparse=True, rng=None):
    """
    生成真实系数（基线 beta 与异质性 gamma）。

    将系数生成从 ``generate_outcomes`` 中抽离，便于训练集与（样本外）测试集
    共享同一组真实系数，从而保证两者来自同一总体。
    """
    rng = rng or np.random.default_rng()
    if sparse:
        s = max(1, min(p, 5))
        beta = np.zeros(p)
        beta[:s] = rng.normal(0, 1, size=s)
    else:
        beta = rng.normal(0, 0.5, size=p)

    gamma = None
    if hetero_type == "linear":
        if sparse:
            s = max(1, min(p, 5))
            gamma = np.zeros(p)
            gamma[:s] = rng.normal(0, 1, size=s)
        else:
            gamma = rng.normal(0, 0.5, size=p)
    return beta, gamma


def generate_outcomes(X, W, ate=1.0, hetero_type="homogeneous",
                      hetero_strength=0.5, noise_sd=1.0,
                      beta=None, gamma=None, rng=None):
    """
    生成潜在结果与观测结果（使用给定的真实系数 beta / gamma）。

    Parameters
    ----------
    X : ndarray
    W : ndarray
    ate : float
        平均处理效应（ATE）
    hetero_type : str
        "homogeneous" | "linear" | "nonlinear"
    hetero_strength : float
        异质性强度
    noise_sd : float
    beta : ndarray
        基线系数（E[Y(0)|X] = X @ beta）
    gamma : ndarray or None
        异质性系数（仅 "linear" 异质时需要）
    rng : np.random.Generator

    Returns
    -------
    Y : ndarray
    Y0 : ndarray
    Y1 : ndarray
    tau_true : ndarray
        个体处理效应
    """
    rng = rng or np.random.default_rng()
    n, p = X.shape

    if beta is None:
        beta, _ = generate_coef(n, p, hetero_type, sparse=True, rng=rng)
    baseline_val = X @ beta

    # 个体处理效应
    if hetero_type == "homogeneous":
        tau_true = np.full(n, ate)
    elif hetero_type == "linear":
        if gamma is None:
            _, gamma = generate_coef(n, p, hetero_type, sparse=True, rng=rng)
        # 标准化使平均 ate 接近目标值
        raw = X @ gamma
        raw = raw - raw.mean()
        scale = raw.std() if raw.std() > 0 else 1.0
        tau_true = ate + hetero_strength * raw / scale
    elif hetero_type == "nonlinear":
        # 非线性交互异质性
        raw = np.sin(X[:, 0]) * X[:, 1] + X[:, 2] ** 2
        raw = raw - raw.mean()
        scale = raw.std() if raw.std() > 0 else 1.0
        tau_true = ate + hetero_strength * raw / scale
    else:
        raise ValueError(f"Unknown hetero_type: {hetero_type}")

    Y0 = baseline_val + rng.normal(0, noise_sd, size=n)
    Y1 = Y0 + tau_true
    Y = W * Y1 + (1 - W) * Y0

    return Y, Y0, Y1, tau_true


def make_dataset(n, p, ate=1.0, corr_type="independent", rho=0.5,
                 block_size=None, hetero_type="homogeneous",
                 hetero_strength=0.5, assignment="stratified",
                 n_strata=4, treat_prob=0.5, noise_sd=1.0,
                 sparse=True, rng=None, coef=None):
    """
    生成一次模拟数据集。若传入 ``coef=(beta, gamma)`` 则复用（保证样本外同总体）。

    Returns
    -------
    dict（含 "beta" / "gamma" 真实系数）
    """
    rng = rng or np.random.default_rng()
    X = generate_covariates(n, p, corr_type, rho, block_size, rng)
    strata = generate_strata(X, n_strata, rng)
    W = assign_treatment(strata, treat_prob, assignment, rng)
    if coef is None:
        beta, gamma = generate_coef(n, p, hetero_type, sparse, rng)
    else:
        beta, gamma = coef
    Y, Y0, Y1, tau_true = generate_outcomes(
        X, W, ate, hetero_type, hetero_strength,
        noise_sd=noise_sd, beta=beta, gamma=gamma, rng=rng
    )
    return {
        "X": X,
        "W": W,
        "Y": Y,
        "strata": strata,
        "Y0": Y0,
        "Y1": Y1,
        "tau_true": tau_true,
        "ate_true": ate,
        "beta": beta,
        "gamma": gamma,
    }
