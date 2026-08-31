"""
ATE 估计量实现

1. Difference-in-Means
2. OLS with covariate adjustment
3. Lasso-adjusted (Post-Lasso / partialling-out via Lasso)
4. Cross-fitting / Double Machine Learning (DML) with Lasso
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def _add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])


class DifferenceInMeans:
    """简单差分均值估计量（无协变量调整）。"""

    def fit(self, Y, W, X=None):
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

    注意：本估计量不做交叉拟合，m_hat 在全部样本上拟合，
    因此在高维情形下的置信区间可能有轻微覆盖不足（欠覆盖）。
    需要严格推断时请使用 CrossFittingDML。
    """

    def __init__(self, alpha=None, standardize=True, random_state=42):
        self.alpha = alpha
        self.standardize = standardize
        self.random_state = random_state

    def fit(self, Y, W, X):
        n, p = X.shape
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X) if self.standardize else X

        # 选择 Lasso 正则化参数
        if self.alpha is None:
            model_y = LassoCV(cv=5, random_state=self.random_state, max_iter=50000).fit(Xs, Y)
            alpha_y = model_y.alpha_
        else:
            alpha_y = self.alpha

        lasso_y = Lasso(alpha=alpha_y, max_iter=50000).fit(Xs, Y)
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

    实现要点
    --------
    1. **折内样本对齐**：折划分由 ``KFold(shuffle=True, random_state=...)`` 生成，
       返回的验证折索引恒为升序，因此 ``X_val`` 的行序与 ``val_idx`` 严格对应。
       早期版本用 ``np.random.shuffle`` + ``np.array_split`` 产生乱序折索引，
       却用布尔掩码 ``X[~train_mask]``（升序）取验证行，导致预测值被错配到
       无关样本上，tau 估计严重失真。
    2. **标准化无泄漏**：每折的 StandardScaler 仅在该折的训练部分拟合。
    3. **可复现**：折划分受 ``random_state`` 控制，不再依赖全局随机状态。
    4. 在随机试验设定下倾向得分真实已知，故使用常数 ``e(X) = mean(W)``，
       这比估计倾向得分更有效（Chernozhukov et al., 2018, §3）。
    """

    def __init__(self, n_folds=5, alpha=None, standardize=True, random_state=42):
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2 for cross-fitting")
        self.n_folds = n_folds
        self.alpha = alpha
        self.standardize = standardize
        self.random_state = random_state

    def fit(self, Y, W, X):
        n, p = X.shape
        if n < self.n_folds:
            raise ValueError(f"n={n} is smaller than n_folds={self.n_folds}")

        m_hat = np.zeros(n)
        e_hat = np.full(n, W.mean())  # 随机试验下倾向得分已知，使用常数

        splitter = KFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        # scikit-learn yields (train_indices, validation_indices).  Reversing
        # these makes every nuisance model train on only one fold and predict
        # the remaining K-1 folds, defeating cross-fitting and distorting ATE.
        for train_idx, val_idx in splitter.split(X):
            if self.standardize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_val = scaler.transform(X[val_idx])
            else:
                X_train, X_val = X[train_idx], X[val_idx]
            Y_train = Y[train_idx]

            if self.alpha is None:
                # 折内 CV 折数受可用样本量约束，避免小样本下退化
                inner_cv = min(5, max(2, len(Y_train) // 10))
                model_y = LassoCV(
                    cv=inner_cv, random_state=self.random_state, max_iter=50000
                ).fit(X_train, Y_train)
                alpha_y = model_y.alpha_
            else:
                alpha_y = self.alpha

            lasso_y = Lasso(alpha=alpha_y, max_iter=50000).fit(X_train, Y_train)
            # val_idx 与 X_val 行序一致（KFold 保证升序），可安全对齐赋值
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
