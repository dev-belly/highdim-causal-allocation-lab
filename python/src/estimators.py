"""
ATE 估计量实现

1. Difference-in-Means
2. OLS with covariate adjustment
3. Lasso-adjusted (Post-Lasso / partialling-out via Lasso)
4. Cross-fitting / Double Machine Learning (DML) with Lasso
"""

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats


def _add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])


class DifferenceInMeans:
    """简单差分均值估计量（无协变量调整）。"""

    def fit(self, Y, W, X=None):
        n = len(Y)
        mu1 = Y[W == 1].mean()
        mu0 = Y[W == 0].mean()
        tau_hat = mu1 - mu0

        var1 = Y[W == 1].var(ddof=1) if (W == 1).sum() > 1 else 0.0
        var0 = Y[W == 0].var(ddof=1) if (W == 0).sum() > 1 else 0.0
        n1 = (W == 1).sum()
        n0 = (W == 0).sum()
        se = np.sqrt(var1 / n1 + var0 / n0)

        return {"tau": tau_hat, "se": se}


class OLSAdjusted:
    """OLS 调整：Y ~ W + X。"""

    def fit(self, Y, W, X):
        n = len(Y)
        Xa = _add_intercept(X)
        M = np.column_stack([W, Xa])
        beta, *_ = np.linalg.lstsq(M, Y, rcond=None)
        tau_hat = beta[0]

        resid = Y - M @ beta
        sigma2 = np.sum(resid ** 2) / (n - M.shape[1])
        XtX_inv = np.linalg.inv(M.T @ M + 1e-12 * np.eye(M.shape[1]))
        se = np.sqrt(sigma2 * XtX_inv[0, 0])

        return {"tau": tau_hat, "se": se}


class LassoAdjusted:
    """
    Lasso 调整估计量。

    采用 Robinson 偏残差形式：
        Y_i - m(X_i) = tau * (W_i - e(X_i)) + eps_i
    其中 m(X) 与 e(X) 分别用 Lasso 回归 Y 与 W 得到。
    若用户未提供倾向得分 e(X)，默认使用常数处理概率。
    """

    def __init__(self, alpha=None, standardize=True):
        self.alpha = alpha
        self.standardize = standardize

    def fit(self, Y, W, X):
        n, p = X.shape
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X) if self.standardize else X

        # 选择 Lasso 正则化参数
        if self.alpha is None:
            model_y = LassoCV(cv=5, random_state=42, max_iter=10000).fit(Xs, Y)
            alpha_y = model_y.alpha_
        else:
            alpha_y = self.alpha

        lasso_y = Lasso(alpha=alpha_y, max_iter=10000).fit(Xs, Y)
        m_hat = lasso_y.predict(Xs)

        # 倾向得分：这里默认完全/分层随机，使用样本均值
        e_hat = np.full(n, W.mean())

        # 估计 ATE
        pseudo = (Y - m_hat) / (W - e_hat)
        weight = (W - e_hat) ** 2
        tau_hat = np.sum(pseudo * weight) / np.sum(weight)

        # Neyman 方差估计
        resid = (Y - m_hat) - tau_hat * (W - e_hat)
        se = np.sqrt(np.sum(resid ** 2 * (W - e_hat) ** 2) / (np.sum(weight) ** 2))

        return {"tau": tau_hat, "se": se}


class CrossFittingDML:
    """
    交叉拟合 Double Machine Learning（K 折）， nuisance 函数使用 Lasso。

    模型：Y - m(X) = tau * (W - e(X)) + eps
    """

    def __init__(self, n_folds=5, alpha=None, standardize=True):
        self.n_folds = n_folds
        self.alpha = alpha
        self.standardize = standardize

    def fit(self, Y, W, X):
        n, p = X.shape
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X) if self.standardize else X

        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, self.n_folds)

        m_hat = np.zeros(n)
        e_hat = np.full(n, W.mean())  # 默认常数倾向得分

        for fold_idx, val_idx in enumerate(folds):
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            X_train, Y_train = Xs[train_mask], Y[train_mask]
            X_val = Xs[val_mask := ~train_mask]

            if self.alpha is None:
                model_y = LassoCV(cv=min(3, len(Y_train)), random_state=42, max_iter=10000).fit(X_train, Y_train)
                alpha_y = model_y.alpha_
            else:
                alpha_y = self.alpha

            lasso_y = Lasso(alpha=alpha_y, max_iter=10000).fit(X_train, Y_train)
            m_hat[val_idx] = lasso_y.predict(X_val)

        pseudo = (Y - m_hat) / (W - e_hat)
        weight = (W - e_hat) ** 2
        tau_hat = np.sum(pseudo * weight) / np.sum(weight)

        resid = (Y - m_hat) - tau_hat * (W - e_hat)
        se = np.sqrt(np.sum(resid ** 2 * (W - e_hat) ** 2) / (np.sum(weight) ** 2))

        return {"tau": tau_hat, "se": se}


ESTIMATORS = {
    "diff_in_means": DifferenceInMeans,
    "ols_adjusted": OLSAdjusted,
    "lasso_adjusted": LassoAdjusted,
    "cross_fitting_dml": CrossFittingDML,
}


def fit_estimator(name, Y, W, X, **kwargs):
    """统一接口拟合单个估计量。"""
    est = ESTIMATORS[name](**kwargs)
    return est.fit(Y, W, X)


def confidence_interval(tau, se, level=0.95):
    """基于正态近似构造置信区间。"""
    z = stats.norm.ppf(1 - (1 - level) / 2)
    return tau - z * se, tau + z * se
