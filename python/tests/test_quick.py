"""端到端冒烟测试：DGP -> 估计量 -> 组合映射 -> 指标聚合。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.estimators as estimators_module  # noqa: E402
from src.data_generating_process import make_dataset  # noqa: E402
from src.estimators import fit_estimator  # noqa: E402
from src.evaluation import aggregate_results  # noqa: E402
from src.experiment import single_trial  # noqa: E402
from src.portfolio import (  # noqa: E402
    map_ate_error_to_portfolio,
    observed_treated_variance,
    portfolio_metrics,
)

ESTIMATORS = [
    "diff_in_means",
    "ols_adjusted",
    "lasso_adjusted",
    "cross_fitting_dml",
]

BASE_DGP = dict(
    n=200,
    p=20,
    ate=1.0,
    corr_type="exchangeable",
    rho=0.5,
    hetero_type="linear",
    hetero_strength=0.5,
    assignment="stratified",
    n_strata=4,
    treat_prob=0.5,
    noise_sd=1.0,
    sparse=True,
)


def test_portfolio_uses_cash_as_zero_return_not_unobserved_control_outcomes():
    metrics = map_ate_error_to_portfolio(
        tau_hat=0.0,
        tau_true=1.0,
        sigma2_true=4.0,
        allow_short=False,
        test_data={"Y1": np.array([2.0, 4.0]), "Y0": np.array([-5.0, 5.0])},
        se_hat=1.0,
    )
    assert metrics["w_hat"] == 0.0
    assert metrics["oos_utility"] == 0.0
    assert metrics["oos_utility_shrunk"] == 0.0


def test_portfolio_risk_variance_uses_observed_treatment_arm_only():
    assert observed_treated_variance(
        np.array([1.0, 3.0, -100.0, 100.0]), np.array([1, 1, 0, 0])
    ) == pytest.approx(2.0)
    assert observed_treated_variance(
        np.array([1.0, 3.0, 999.0, -999.0]), np.array([1, 1, 0, 0])
    ) == pytest.approx(2.0)


def test_oos_utility_uses_configured_risk_aversion():
    metrics = portfolio_metrics(
        w_hat=0.5,
        w_star=0.5,
        mu_true=2.0,
        sigma2_true=2.0,
        test_Y1=np.array([0.0, 2.0]),
        risk_aversion=2.0,
    )
    assert metrics["oos_utility"] == pytest.approx(0.0)


def test_experiment_trial_reports_finite_oos_utility():
    result = single_trial(dict(BASE_DGP, p=5), "diff_in_means", {}, 42)
    assert np.isfinite(result["oos_utility"])
    assert np.isfinite(result["oos_utility_shrunk"])


def run_one_trial(est, seed, n_trials=10):
    results = []
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t)
        data = make_dataset(**BASE_DGP, rng=rng)
        res = fit_estimator(est, data["Y"], data["W"], data["X"])
        test_data = make_dataset(
            **BASE_DGP,
            rng=np.random.default_rng(seed + t + 1_000_000),
            coef=(data["beta"], data["gamma"]),
        )
        port = map_ate_error_to_portfolio(
            res["tau"],
            BASE_DGP["ate"],
            observed_treated_variance(data["Y"], data["W"]),
            test_data={"Y1": test_data["Y1"]},
        )
        res.update(port)
        results.append(res)
    return aggregate_results(results, BASE_DGP["ate"])


@pytest.mark.parametrize("est", ESTIMATORS)
def test_estimator_is_approximately_unbiased(est):
    """每个估计量的偏差应远小于真实 ATE 量级。"""
    metrics = run_one_trial(est, seed=0)
    assert abs(metrics["bias"]) < 0.35, f"{est} bias too large: {metrics['bias']}"


@pytest.mark.parametrize("est", ESTIMATORS)
def test_estimator_metrics_are_valid(est):
    """RMSE 非负，覆盖率落在 [0, 1]。"""
    metrics = run_one_trial(est, seed=100)
    assert metrics["rmse"] >= 0.0
    assert 0.0 <= metrics["coverage"] <= 1.0


def test_covariate_adjustment_reduces_variance():
    """在高维相关协变量下，协变量调整应当降低 RMSE。"""
    base_rmse = run_one_trial("diff_in_means", seed=7)["rmse"]
    dml_rmse = run_one_trial("cross_fitting_dml", seed=7)["rmse"]
    assert dml_rmse <= base_rmse, (
        f"DML rmse={dml_rmse:.4f} should not exceed diff_in_means rmse={base_rmse:.4f}"
    )


def test_cross_fitting_is_reproducible():
    """同一份数据重复拟合必须得到完全一致的 tau（折划分受 random_state 控制）。"""
    data = make_dataset(**BASE_DGP, rng=np.random.default_rng(2024))
    first = fit_estimator("cross_fitting_dml", data["Y"], data["W"], data["X"])
    second = fit_estimator("cross_fitting_dml", data["Y"], data["W"], data["X"])
    assert first["tau"] == pytest.approx(second["tau"], abs=0, rel=0)


def test_cross_fitting_trains_on_k_minus_one_folds(monkeypatch):
    """KFold 返回 train、validation；不能把两组索引反向解包。"""
    train_sizes = []

    class RecordingScaler:
        def fit_transform(self, values):
            train_sizes.append(len(values))
            return np.asarray(values)

        def transform(self, values):
            return np.asarray(values)

    monkeypatch.setattr(estimators_module, "StandardScaler", RecordingScaler)
    data = make_dataset(**dict(BASE_DGP, n=50), rng=np.random.default_rng(77))
    estimator = estimators_module.CrossFittingDML(n_folds=5, alpha=0.05)
    result = estimator.fit(data["Y"], data["W"], data["X"])

    assert train_sizes == [40] * 5
    assert np.isfinite(result["tau"])


def test_cross_fitting_respects_fold_alignment():
    """
    回归测试：交叉拟合的 m_hat 必须与样本真实对齐。

    早期实现用乱序折索引取验证行、却按布尔掩码升序切分，导致预测值错配，
    DML 的 RMSE 反而不做调整的简单差分均值。这里锁定该行为不再复发。
    """
    dgp = dict(BASE_DGP, n=500, p=20)
    taus = []
    for t in range(15):
        data = make_dataset(**dgp, rng=np.random.default_rng(t))
        taus.append(fit_estimator("cross_fitting_dml", data["Y"], data["W"], data["X"])["tau"])
    taus = np.asarray(taus)
    assert np.abs(taus - dgp["ate"]).max() < 0.6, (
        f"DML produced extreme estimates: min={taus.min():.3f}, max={taus.max():.3f}"
    )
    assert taus.std() < 0.25, f"DML estimates too dispersed: sd={taus.std():.3f}"


if __name__ == "__main__":
    for est in ESTIMATORS:
        m = run_one_trial(est, seed=0)
        print(
            f"{est:20s}: bias={m['bias']:+.3f}, rmse={m['rmse']:.3f}, "
            f"coverage={m['coverage']:.2f}"
        )
