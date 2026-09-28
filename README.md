# 高维协变量调整与稳健资产配置实验平台

> High-Dimensional Covariate Adjustment & Robust Asset Allocation Lab
>
> 用蒙特卡洛模拟回答两个问题：**在协变量维度 p 与样本量 n 同量级时，哪种 ATE 估计量仍然可信？**
> 以及 **估计误差会如何传导到下游的资产配置权重与样本外风险？**

交互式实验室（React + Plotly，无需后端）：<https://dev-belly.github.io/highdim-causal-allocation-lab/>

---

## 目录

- [为什么值得看](#为什么值得看)
- [方法](#方法)
- [核心结果](#核心结果)
- [一次真实的 bug 修复](#一次真实的-bug-修复)
- [快速开始](#快速开始)
- [仓库结构](#仓库结构)
- [参考文献](#参考文献)
- [License](#license)

---

## 为什么值得看

大多数因果推断笔记停在"跑通 Double ML 的 API"。这个仓库关心的是更工程化的问题：

1. **估计量真的更好吗？** 在 `p/n` 从 0.1 到 0.5 的区间内，把交叉拟合 DML 与不做调整的
   差分均值、OLS、Post-Lasso 放在同一 Monte Carlo 网格上比较 **偏差 / RMSE / 置信区间覆盖率**。
2. **误差如何传导？** ATE 的估计误差不是终点。项目把它映射为均值-方差最优权重，
   再用 **Bayes-Stein 收缩**修正，比较权重偏移和独立测试集上的确定性等价效用。
3. **可复现与可验证。** 全部随机性由显式 `random_state` / `default_rng(seed)` 控制，
   CI 同时执行前端单测、构建与依赖审计，并在 Python 3.10 / 3.12 上跑回归测试，
   包含锁定历史交叉拟合 bug 的专项测试。

---

## 方法

### 数据生成过程（DGP）

分层随机试验，可调节的维度：

| 维度 | 取值 |
|---|---|
| 样本量 `n` | 200 / 500 / 1000 |
| 协变量维度 `p` | 5 / 20 / 100 |
| 相关结构 | 独立 / 可交换 / AR(1) / 分块相关 |
| 处理效应异质性 | 同质 / 线性异质 |
| 分配机制 | 完全随机 / 分层随机 |
| 稀疏性 | 真模型仅部分协变量有效应 |

### 四个估计量

| 估计量 | 做法 | 理论定位 |
|---|---|---|
| `diff_in_means` | 简单组间均值差 | 无调整基线 |
| `ols_adjusted` | `Y ~ W + X` 全量回归 | 低维经典调整 |
| `lasso_adjusted` | Robinson 偏残差 + Lasso 拟合 `m(X)` | 高维调整（无交叉拟合） |
| `cross_fitting_dml` | Robinson 偏残差 + **K 折交叉拟合** Lasso | Chernozhukov et al. (2018) 正交化估计 |

浏览器版的交叉拟合会在每个外层训练折内重新选择 Lasso 正则化参数；验证折的结果不参与该折的拟合或调参。页面默认运行 2 × 2 参数网格、每格 10 次，仅用于快速演示；覆盖率等统计指标需要增加重复次数再解释，计算时间也会增长。

在随机试验设定下，倾向得分 `e(X)` 真实已知，因此使用常数 `mean(W)` ——
这比估计倾向得分更有效，也避免引入额外的 nuisance 估计误差。

### 从 ATE 误差到组合损失

`portfolio.py` 把估计出的 `τ̂` 代入均值-方差问题，得到权重 `ŵ`，
与用真实 `τ` 得到的条件 oracle 权重 `w*` 对比，报告：

- 权重偏移 `‖ŵ − w*‖`
- **Bayes-Stein 收缩**后的权重及其改善幅度
- 样本外确定性等价效用：`mean(w × Y1) − 0.5 × 风险厌恶系数 × var(w × Y1)`（独立测试集）

这里把处理组潜在结果 `Y1` 作为风险资产收益的合成代理，零超额收益作为无风险资产；
对照组潜在结果 `Y0` 只用于因果实验，不是可投资的无风险收益。风险资产方差仅从训练集**实际观测的处理组**估计；
`w*` 使用真实 ATE 作条件 oracle 对照，真实 ATE 不进入实际权重或收缩步骤。单风险资产配零收益资产时，
正权重缩放通常不改变 Sharpe，因此这里不再把 Sharpe 提升作为收缩收益。收缩也不保证改善样本外效用。
样本外指标使用合成数据中完整的 `Y1` 潜在结果作评估真值，现实随机试验无法对全部个体直接观测它。
这是方法演示，不是可交易资产回测。

---

## 核心结果

Monte Carlo 40 次重复，`corr_type=exchangeable, rho=0.5`，
线性异质处理效应，分层随机分配，真实 `ATE = 1.0`：

### RMSE（越小越好）

| 配置 | `diff_in_means` | `ols_adjusted` | `lasso_adjusted` | `cross_fitting_dml` |
|---|---|---|---|---|
| n=200, p=20 | 0.2572 | 0.1524 | **0.1397** | 0.1834 |
| n=500, p=20 | 0.1926 | **0.0874** | 0.0925 | 0.0956 |
| n=1000, p=20 | 0.1929 | **0.0583** | 0.0595 | 0.0665 |
| n=500, p=100 | 0.1756 | 0.0836 | 0.0865 | **0.0814** |

### 95% 置信区间覆盖率（越接近 0.95 越好）

| 配置 | `diff_in_means` | `ols_adjusted` | `lasso_adjusted` | `cross_fitting_dml` |
|---|---|---|---|---|
| n=200, p=20 | 1.00 | 0.97 | 0.97 | **0.95** |
| n=500, p=20 | 1.00 | 0.97 | 0.97 | **0.97** |
| n=1000, p=20 | 0.95 | 0.97 | 0.97 | 1.00 |
| n=500, p=100 | 0.95 | 0.97 | 0.97 | **0.97** |

**怎么读这张表：**

- 协变量调整在低维时收益巨大（n=500, p=20 上 RMSE 从 0.193 降到 0.087，降幅 55%）。
- 维度升高（`p/n` 变大）后，交叉拟合 DML 开始显现优势：n=500, p=100 时它反超 OLS 与 Lasso——
  这正是交叉拟合存在的意义：用样本分割换取对 nuisance 估计误差的鲁棒性。
- `diff_in_means` 的覆盖率长期为 1.00，是**过度保守**的标准误，不是"更准"。

---

## 一次真实的 bug 修复

这个仓库的 `cross_fitting_dml` 曾经**比不做任何调整的简单均值差还差**
（n=500, p=20 上 RMSE 0.2549 vs 0.1926）。这不是调参问题，是实现缺陷：

```python
# 旧实现
indices = np.arange(n)
np.random.shuffle(indices)          # ① 折划分依赖全局随机状态，不可复现
folds = np.array_split(indices, self.n_folds)

for val_idx in folds:               # val_idx 是乱序的，例如 [4, 3]
    train_mask[val_idx] = False
    X_val = Xs[~train_mask]         # ② 布尔掩码按升序取行，例如 [3, 4]
    m_hat[val_idx] = lasso.predict(X_val)   # ③ 预测值被赋给乱序位置 → 错配
```

`X[val_idx]` 与 `X[~train_mask]` 在折内索引非升序时**行序不同**，
于是第 k 个预测值被写到了错误的样本上，交叉拟合的残差被随机打乱。

**修复**：改用 `KFold(shuffle=True, random_state=...)`，它返回的验证折索引恒为升序，
行序与赋值位置严格对齐；同时把标准化改为每折在训练部分拟合，消除数据泄漏。

后续审查还发现一次清理提交把 scikit-learn 返回的 `(train_idx, val_idx)` 反向解包，
导致每折只用 1/K 样本训练、却预测其余 K−1 折。实现现已恢复正确顺序，并用
`test_cross_fitting_trains_on_k_minus_one_folds` 明确锁定每折训练样本数，避免仅靠宽松的
统计阈值漏掉同类回归。

修复前后对比（RMSE，40 次重复）：

| 配置 | 修复前 | 修复后 | 改善 |
|---|---|---|---|
| n=200, p=20 | 0.3941 | **0.1834** | −53% |
| n=500, p=20 | 0.2549 | **0.0956** | −62% |
| n=1000, p=20 | 0.2055 | **0.0665** | −68% |
| n=500, p=100 | 0.2624 | **0.0814** | −69% |

`tests/test_quick.py::test_cross_fitting_respects_fold_alignment` 专门锁定该行为，
防止回归。

---

## 快速开始

交互式前端：

```bash
git clone https://github.com/dev-belly/highdim-causal-allocation-lab.git
cd highdim-causal-allocation-lab
npm ci
npm test
npm run dev
```

Python 实验：

```bash
cd highdim-causal-allocation-lab/python

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

跑测试：

```bash
pytest tests/ -q
```

跑完整蒙特卡洛实验（参数网格在 `config/default.yaml`）：

```bash
python -m src.experiment
```

只重新生成图表与报告（基于已有的 `output/tables/results.csv`）：

```bash
python -m src.generate_outputs
```

最小用法示例：

```python
import numpy as np
from src.data_generating_process import make_dataset
from src.estimators import fit_estimator

dgp = dict(n=500, p=100, ate=1.0, corr_type="exchangeable", rho=0.5,
           hetero_type="linear", hetero_strength=0.5, assignment="stratified",
           n_strata=4, treat_prob=0.5, noise_sd=1.0, sparse=True)

data = make_dataset(**dgp, rng=np.random.default_rng(42))
res = fit_estimator("cross_fitting_dml", data["Y"], data["W"], data["X"])
print(f"tau = {res['tau']:.4f}  (se = {res['se']:.4f})")
```

---

## 仓库结构

```text
highdim-causal-allocation-lab/
├── src/                      # React 交互实验室、Web Worker 与 TypeScript 核心算法
├── index.html                # Vite 开发/构建入口
├── package.json              # 前端测试与构建命令
├── dist/                     # 可部署到 GitHub Pages 的前端构建产物
├── python/
│   ├── config/default.yaml   # 参数网格与估计量配置
│   ├── src/
│   │   ├── data_generating_process.py   # DGP：相关结构 / 异质性 / 分配机制
│   │   ├── estimators.py                # 4 个 ATE 估计量
│   │   ├── evaluation.py                # 蒙特卡洛聚合：偏差 / RMSE / 覆盖率
│   │   ├── portfolio.py                 # 均值-方差权重 + Bayes-Stein 收缩
│   │   ├── experiment.py                # 主实验脚本与图表
│   │   └── generate_outputs.py          # 从 results.csv 重绘报告
│   ├── tests/test_quick.py              # 端到端 + 回归测试
│   ├── notebooks/analysis.ipynb
│   ├── requirements.txt
│   └── pyproject.toml
└── .github/workflows/ci.yml
```

---

## 参考文献

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018).
  *Double/debiased machine learning for treatment and structural parameters.* The Econometrics Journal, 21(1), C1–C68.
- Robinson, P. M. (1988). *Root-N-consistent semiparametric regression.* Econometrica, 56(4), 931–954.
- Belloni, A., Chernozhukov, V., & Hansen, C. (2014). *Inference on treatment effects after selection among high-dimensional controls.*
  The Review of Economic Studies, 81(2), 608–650.
- Jorion, P. (1986). *Bayes-Stein estimation for portfolio analysis.* Journal of Financial and Quantitative Analysis, 21(3), 279–292.

---

## License

[MIT](LICENSE)
