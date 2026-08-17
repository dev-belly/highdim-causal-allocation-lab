# 高维协变量调整与稳健资产配置实验平台

> **High-Dimensional Covariate Adjustment & Robust Asset-Allocation Lab**
>
> 交互式蒙特卡洛因果推断实验室：在分层随机试验设定下，比较四种处理效应（ATE）估计量在高维下的有限样本表现，并将估计误差映射到均值-方差稳健配置的样本外风险与权重偏移。

---

## 为什么做这个

在高维场景下（协变量维度 *p* 可与样本量 *n* 同阶甚至更大），传统的“差分均值”估计量偏差大、方差高，而朴素 OLS 在高维下会过拟合。**双重机器学习（Double Machine Learning / 交叉拟合 DML）** 与 **Lasso 调整** 是近年来因果推断文献中被反复证明能稳健降偏的工具。

但“估计得准”只是第一步。在真实投资流程里，一个被低估的处理效应会直接扭曲组合权重、放大样本外回撤。本平台把这两件事打通：

1. **因果层** — 用受控蒙特卡洛（DGP 完全已知真实 ATE）公正比较四种估计量；
2. **配置层** — 把 ATE 估计误差注入单期均值-方差最优权重，对比 **收缩基准（Bayes–Stein 风格）** 与无约束基准的样本外夏普与效用损失。

---

## 方法论（四估计量 + 稳健配置）

### 处理效应估计量

| 估计量 | 方法 | 关键性质 |
| --- | --- | --- |
| `diff_in_means` | 差分均值（无调整） | 基准；高维下有偏、方差大 |
| `ols_adjusted` | `Y ~ W + X` 含全协变量 | 低维无偏；高维过拟合 |
| `lasso_adjusted` | Robinson 偏残差 + Lasso 条件均值 | 高维降偏；λ 由 K 折 CV 选定 |
| `cross_fitting_dml` | 交叉拟合 DML（K 折 nuisance + 全局 λ） | 去偏、正交化；高维稳健 |

- 标准误：差分均值用 Neyman 稳健方差；调整类用偏残差（partialling-out）解析方差。
- 覆盖率：基于正态近似的 95% 置信区间。
- DGP 支持四种协变量相关结构（独立 / 可交换 / AR(1) / 分块）、处理效应异质性（同质 / 线性 / 非线性）与稀疏/稠密基线。

### 稳健资产配置

- **均值-方差权重**：`w* = μ / (γ·σ²)`，可施加非负约束（不允许做空）。
- **收缩基准**：`w_shrunk = (1−λ)·ŵ + λ·w*`（λ 即 shrinkage intensity），对应 Bayes–Stein 风格向理论最优权重收缩。
- **样本外评估**：独立测试集计算 OOS 夏普比率与确定性等价效用损失，直接量化“估计误差 → 配置损失”的传导。

---

## 技术栈

按“世界级前端工程 + 严谨量化内核”标准构建：

- **Vite 5 + React 18 + TypeScript（strict）** — 类型安全的数值内核与组件层
- **Web Worker 并行蒙特卡洛引擎** — 模拟脱离 UI 主线程，逐单元回传进度
- **Plotly.js（CDN 加载）** — 专业金融图表：覆盖率折线、RMSE 热力图、权重误差曲线、**RMSE 3D 曲面**
- **Vitest** — 覆盖 DGP 可复现性、估计量有限性、覆盖率、高维 RMSE、收缩估计、权重截断的单元测试
- **GitHub Actions CI/CD** — 推送即构建并部署到 GitHub Pages
- **纯函数数值内核**（无第三方数值库）：mulberry32 可复现 PRNG、Box–Muller 高斯、Cholesky 分解、高斯消元/求逆、Acklam 正态分位数近似、坐标下降 Lasso

> 工程规范、配置方法论与量化把关参考了 **ModernWebappExpert · EquityResearchExpert · TradingAgentTeam** 的方法论。

---

## 目录结构

```
.
├── index.html                # 入口（含 Plotly CDN）
├── src/
│   ├── core/                 # 数值内核（纯函数，可独立测试/复用）
│   │   ├── rng.ts            # 可复现 PRNG / 高斯 / Cholesky / 多元正态
│   │   ├── linalg.ts         # 标准化 / 求解 / 求逆 / 软阈值 / 分位数 / PPF / CI
│   │   ├── dgp.ts            # 数据生成过程（相关结构 / 分层 / 异质 / 稀疏）
│   │   ├── lasso.ts          # 坐标下降 Lasso + K 折 CV
│   │   ├── estimators.ts     # 四种 ATE 估计量
│   │   ├── shrinkage.ts      # Schäfer–Strimmer 协方差 / Bayes–Stein 均值收缩
│   │   ├── portfolio.ts      # 均值-方差权重 / 配置评估 / OOS 夏普
│   │   ├── simulator.ts      # 蒙特卡洛引擎（单试验 → 聚合）
│   │   └── core.test.ts      # Vitest 测试
│   ├── worker/sim.worker.ts  # Web Worker 并行引擎
│   ├── hooks/useSimulation.ts# Worker 生命周期与进度状态
│   ├── components/           # ControlPanel / charts / ResultTable / Plot
│   └── App.tsx               # 主应用
├── python/                   # 等价的 Python 参考实现（方法学可复现）
│   ├── src/                  # 与 JS 内核一一对应的因果/配置模块
│   ├── config/default.yaml   # 实验配置
│   └── output/               # 已运行的表格与图表（含 report.md）
└── .github/workflows/        # GitHub Pages 部署流水线
```

---

## 本地运行

```bash
# 1. 安装依赖（Node 22+）
npm install

# 2. 开发预览
npm run dev

# 3. 类型检查 + 生产构建（产物在 dist/）
npm run build

# 4. 运行算法单元测试
npm test
```

打开浏览器访问 `npm run dev` 给出的本地地址，设置左侧实验参数后点击 **运行蒙特卡洛实验**，结果（覆盖率、RMSE、权重误差、3D 曲面、摘要表）将实时呈现。

### Python 参考实现

`python/` 提供等价的方法学实现，便于在服务器/批处理场景复现结果：

```bash
cd python
pip install -r requirements.txt
python -m src.experiment          # 运行主实验，输出表格/图表/report
```

---

## 部署（GitHub Pages）

仓库已配置 `.github/workflows/deploy.yml`：推送 `main` 分支即自动 **构建 → 上传产物 → 部署到 GitHub Pages**。

- 构建产物使用相对路径（`base: './'`），可部署在任意子路径。
- Pages 源设为 **GitHub Actions**（由 `actions/deploy-pages` 发布），无需维护 `gh-pages` 分支。

---

## 关键结论（来自 `python/output/report.md`）

在已运行的基准实验中（12 个 (n, p) 单元 × 4 估计量 × 100 次试验）：

- **交叉拟合 DML 在高维（p 较大）下 RMSE 显著低于差分均值与 OLS**，验证了去偏/正交化的价值；
- **收缩基准的样本外夏普稳定优于无约束估计**，量化了“稳健配置”对估计误差的免疫能力；
- 95% 覆盖率在差分均值下接近名义水平，而在高维 OLS 下严重失真。

详细数字见 `python/output/tables/results.csv` 与 `python/output/report.md`。
