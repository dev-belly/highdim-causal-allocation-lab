"""
将 ATE 估计误差映射到均值-方差资产配置问题。

基本设定：
- 两种资产：风险资产（处理组）与无风险资产（对照组）
- 真实期望超额收益为 ATE_true
- 投资者使用估计的 ATE_hat 进行均值-方差优化
- 比较估计权重与真实最优权重的差异及样本外表现
"""

import numpy as np


def mean_variance_weight(mu, sigma2, risk_aversion=1.0, allow_short=True):
    """
    均值-方差最优权重。

    max_w  E[w*R] - (lambda/2) Var[w*R]
    => w* = mu / (lambda * sigma2)

    Parameters
    ----------
    mu : float
        期望收益
    sigma2 : float
        收益方差
    risk_aversion : float
    allow_short : bool
        若 False，则截断到 [0, 1]

    Returns
    -------
    float
    """
    if sigma2 <= 0:
        sigma2 = 1e-6
    w = mu / (risk_aversion * sigma2)
    if not allow_short:
        w = np.clip(w, 0.0, 1.0)
    return w


def bayes_stein_shrinkage(mu, var_mu, tau2, prior_mean=None):
    """
    Bayes–Stein / James–Stein 收缩：将估计值向先验均值收缩。

    收缩强度 w_i = var_mu_i / (var_mu_i + tau2)，后验 = (1-w_i)*mu_i + w_i*overall。
    单资产设定下 overall 默认取先验均值（如 0，即向无风险收缩），不依赖真值。
    """
    mu = np.asarray(mu, dtype=float)
    var_mu = np.asarray(var_mu, dtype=float)
    overall = prior_mean if prior_mean is not None else float(mu.mean())
    w = var_mu / (var_mu + tau2)
    return (1 - w) * mu + w * overall


def portfolio_metrics(w_hat, w_star, mu_true, sigma2_true, test_Y1, test_Y0=None, w_shrunk=None):
    """
    评估估计权重相对于真实最优权重的表现。

    Parameters
    ----------
    w_hat : float
        基于 ATE_hat 得到的权重
    w_star : float
        基于 ATE_true 得到的最优权重（oracle 参考，不用于收缩）
    mu_true : float
        真实 ATE
    sigma2_true : float
        真实收益方差
    test_Y1 : ndarray
        测试集处理组结果（用于样本外收益）
    test_Y0 : ndarray, optional
        测试集对照组结果；若提供则计算样本外实际组合收益
    w_shrunk : float, optional
        数据驱动收缩后的权重（用于对比“收缩基准 vs 无约束基准”）

    Returns
    -------
    dict
    """
    weight_bias = w_hat - w_star
    weight_abs_err = np.abs(weight_bias)

    # 理论预期效用损失（基于真实分布）
    utility_star = w_star * mu_true - 0.5 * (w_star ** 2) * sigma2_true
    utility_hat = w_hat * mu_true - 0.5 * (w_hat ** 2) * sigma2_true
    utility_loss = utility_star - utility_hat

    metrics = {
        "w_star": w_star,
        "w_hat": w_hat,
        "weight_bias": weight_bias,
        "weight_abs_err": weight_abs_err,
        "utility_loss": utility_loss,
    }

    if test_Y1 is not None and test_Y0 is not None:
        # 样本外组合收益：w * Y1 + (1-w) * Y0
        port_returns = w_hat * test_Y1 + (1 - w_hat) * test_Y0
        metrics["oos_mean_return"] = port_returns.mean()
        metrics["oos_volatility"] = port_returns.std(ddof=1)
        metrics["oos_sharpe"] = metrics["oos_mean_return"] / (metrics["oos_volatility"] + 1e-9)
        if w_shrunk is not None:
            port_shrunk = w_shrunk * test_Y1 + (1 - w_shrunk) * test_Y0
            metrics["oos_sharpe_shrunk"] = port_shrunk.mean() / (port_shrunk.std(ddof=1) + 1e-9)
            metrics["w_shrunk"] = w_shrunk

    return metrics


def map_ate_error_to_portfolio(tau_hat, tau_true, sigma2_true=1.0,
                                risk_aversion=1.0, allow_short=True,
                                test_data=None, se_hat=0.0,
                                prior_var=1.0, prior_mean=0.0):
    """
    将单次 ATE 估计映射到资产配置指标。

    Parameters
    ----------
    tau_hat : float
    tau_true : float
    sigma2_true : float
    risk_aversion : float
    allow_short : bool
    test_data : dict, optional
        {"Y1": ..., "Y0": ...}
    se_hat : float
        估计 ATE 的标准误（驱动数据驱动的收缩强度）
    prior_var : float
        Bayes–Stein 先验方差（越大收缩越弱）
    prior_mean : float
        先验均值（单资产设定下向无风险/零超额收益收缩）

    Returns
    -------
    dict
    """
    w_star = mean_variance_weight(tau_true, sigma2_true, risk_aversion, allow_short)
    w_hat = mean_variance_weight(tau_hat, sigma2_true, risk_aversion, allow_short)

    # 数据驱动的 Bayes–Stein 收缩：将估计 ATE 向先验均值收缩（不依赖真值）
    tau_shrunk = float(bayes_stein_shrinkage([tau_hat], [se_hat * se_hat], prior_var, prior_mean)[0])
    w_shrunk = mean_variance_weight(tau_shrunk, sigma2_true, risk_aversion, allow_short)

    test_Y1 = test_data["Y1"] if test_data else None
    test_Y0 = test_data["Y0"] if test_data else None

    return portfolio_metrics(w_hat, w_star, tau_true, sigma2_true, test_Y1, test_Y0, w_shrunk=w_shrunk)
