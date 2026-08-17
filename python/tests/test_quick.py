"""快速冒烟测试，验证端到端流程。"""

import sys
sys.path.insert(0, "/Users/guoyuanyuan/WorkBuddy/2026-08-17-15-44-27")

from src.data_generating_process import make_dataset
from src.estimators import fit_estimator
from src.evaluation import aggregate_results
from src.portfolio import map_ate_error_to_portfolio
import numpy as np


def test_end_to_end():
    n_trials = 10
    dgp = dict(n=200, p=20, ate=1.0, corr_type="exchangeable", rho=0.5,
               hetero_type="linear", hetero_strength=0.5, assignment="stratified",
               n_strata=4, treat_prob=0.5, noise_sd=1.0, sparse=True)

    for est in ["diff_in_means", "ols_adjusted", "lasso_adjusted", "cross_fitting_dml"]:
        results = []
        for t in range(n_trials):
            rng = np.random.default_rng(t)
            data = make_dataset(**dgp, rng=rng)
            res = fit_estimator(est, data["Y"], data["W"], data["X"])
            test_data = make_dataset(**dgp, rng=np.random.default_rng(t + 1_000_000))
            port = map_ate_error_to_portfolio(
                res["tau"], dgp["ate"], data["Y"].var(ddof=1),
                test_data={"Y1": test_data["Y1"], "Y0": test_data["Y0"]}
            )
            res.update(port)
            results.append(res)
        metrics = aggregate_results(results, dgp["ate"])
        print(f"{est:20s}: bias={metrics['bias']:.3f}, rmse={metrics['rmse']:.3f}, coverage={metrics['coverage']:.2f}")


if __name__ == "__main__":
    test_end_to_end()
