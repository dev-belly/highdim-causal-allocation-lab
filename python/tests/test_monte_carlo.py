"""Monte Carlo comparisons must use the same simulated trials per estimator."""

import pandas as pd

from src.evaluation import run_monte_carlo


def test_estimator_results_do_not_depend_on_comparison_order():
    dgp = dict(n=40, p=3, ate=1.0)
    names = ["diff_in_means", "ols_adjusted"]

    first = run_monte_carlo(dgp, names, n_trials=4, seed=71)
    reversed_order = run_monte_carlo(dgp, names[::-1], n_trials=4, seed=71)
    standalone = run_monte_carlo(dgp, names[:1], n_trials=4, seed=71)

    first = first.set_index("estimator").sort_index()
    pd.testing.assert_frame_equal(
        first, reversed_order.set_index("estimator").sort_index(), check_exact=True
    )
    pd.testing.assert_frame_equal(
        first.loc[names[:1]], standalone.set_index("estimator"), check_exact=True
    )
