"""
评估指标与蒙特卡洛聚合
"""

import numpy as np
import pandas as pd
from src.estimators import confidence_interval


def run_single_trial(dgp_params, estimator_name, est_kwargs=None, rng=None):
    """
    运行一次模拟 trial，返回估计量结果。
    """
    from src.data_generating_process import make_dataset
    from src.estimators import fit_estimator

    est_kwargs = est_kwargs or {}
    rng = rng or np.random.default_rng()

    data = make_dataset(**dgp_params, rng=rng)
    result = fit_estimator(estimator_name, data["Y"], data["W"], data["X"], **est_kwargs)
    result["ate_true"] = data["ate_true"]
    result["estimator"] = estimator_name
    return result


def aggregate_results(results, ate_true, level=0.95):
    """
    将多次 trial 的结果聚合为指标。

    Parameters
    ----------
    results : list of dict
        每次 trial 的 {tau, se}
    ate_true : float

    Returns
    -------
    dict
    """
    taus = np.array([r["tau"] for r in results])
    ses = np.array([r["se"] for r in results])

    bias = taus.mean() - ate_true
    std_dev = taus.std(ddof=1)
    rmse = np.sqrt(np.mean((taus - ate_true) ** 2))
    mean_se = ses.mean()

    # 置信区间覆盖率
    covered = 0
    for tau, se in zip(taus, ses):
        lo, hi = confidence_interval(tau, se, level)
        if lo <= ate_true <= hi:
            covered += 1
    coverage = covered / len(results)

    return {
        "bias": bias,
        "std_dev": std_dev,
        "rmse": rmse,
        "mean_se": mean_se,
        "coverage": coverage,
        "n_trials": len(results),
    }


def run_monte_carlo(dgp_params, estimators, n_trials=500, seed=0, est_kwargs_map=None):
    """
    对多个估计量运行蒙特卡洛模拟。

    Parameters
    ----------
    dgp_params : dict
        传递给 make_dataset 的参数
    estimators : list of str
    n_trials : int
    seed : int
    est_kwargs_map : dict
        每个估计量的额外参数

    Returns
    -------
    pd.DataFrame
    """
    est_kwargs_map = est_kwargs_map or {}
    rng = np.random.default_rng(seed)

    records = []
    for est_name in estimators:
        results = []
        for t in range(n_trials):
            trial_rng = np.random.default_rng(rng.integers(0, 2 ** 31))
            res = run_single_trial(dgp_params, est_name, est_kwargs_map.get(est_name), rng=trial_rng)
            results.append(res)

        metrics = aggregate_results(results, dgp_params["ate"])
        metrics["estimator"] = est_name
        metrics.update(dgp_params)
        records.append(metrics)

    return pd.DataFrame(records)
