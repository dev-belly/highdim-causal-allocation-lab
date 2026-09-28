import { useState } from 'react';
import { ControlPanel, type BaseParams } from './components/ControlPanel';
import {
  CoverageChart,
  RmseHeatmap,
  WeightErrorChart,
  RmseSurface,
} from './components/charts';
import { ResultTable } from './components/ResultTable';
import { useSimulation } from './hooks/useSimulation';
import type { EstimatorName, DGPParams } from './core/types';

const DEFAULT_BASE: BaseParams = {
  ate: 1.0,
  corrType: 'exchangeable',
  rho: 0.5,
  heteroType: 'linear',
  heteroStrength: 0.5,
  assignment: 'stratified',
  nStrata: 4,
  treatProb: 0.5,
  noiseSd: 1.0,
  sparse: true,
};

export default function App() {
  const [base, setBase] = useState<BaseParams>(DEFAULT_BASE);
  const [nLevels, setNLevels] = useState<number[]>([200, 500]);
  const [pLevels, setPLevels] = useState<number[]>([5, 20]);
  const [nTrials, setNTrials] = useState(10);
  const [estimators, setEstimators] = useState<EstimatorName[]>([
    'diff_in_means',
    'ols_adjusted',
    'lasso_adjusted',
    'cross_fitting_dml',
  ]);
  const [selectedP, setSelectedP] = useState(5);
  const [selectedEstimator, setSelectedEstimator] = useState<EstimatorName>('lasso_adjusted');

  const { progress, results, running, error, start, stop } = useSimulation();

  const psInResults = results.length
    ? [...new Set(results.map((c) => c.params.p))]
    : pLevels;
  const effP = psInResults.includes(selectedP) ? selectedP : psInResults[0] ?? selectedP;
  const estsInResults = (
    results.length ? (Object.keys(results[0].summary) as EstimatorName[]) : estimators
  ) as EstimatorName[];
  const effEst = estsInResults.includes(selectedEstimator)
    ? selectedEstimator
    : estsInResults[0] ?? selectedEstimator;

  const onRun = () => {
    const grid: DGPParams[] = [];
    for (const n of nLevels) for (const p of pLevels) grid.push({ ...base, n, p });
    setSelectedP(pLevels[0]);
    start(grid, estimators, nTrials, 42);
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>高维协变量调整与稳健资产配置实验平台</h1>
          <div className="subtitle">
            交互式蒙特卡洛因果推断实验室：比较差分均值 / OLS / Lasso / 交叉拟合 DML
            在高维下的有限样本表现，并将估计误差映射到均值-方差稳健配置的样本外风险与权重偏移。
          </div>
        </div>
        <div className="expert-bar">
          <span className="expert-chip">
            工程规范 · <b>ModernWebappExpert</b>
          </span>
          <span className="expert-chip">
            配置方法论 · <b>EquityResearchExpert</b>
          </span>
          <span className="expert-chip">
            量化把关 · <b>TradingAgentTeam</b>
          </span>
        </div>
      </header>

      <div className="grid">
        <ControlPanel
          base={base}
          setBase={setBase}
          nLevels={nLevels}
          setNLevels={setNLevels}
          pLevels={pLevels}
          setPLevels={setPLevels}
          nTrials={nTrials}
          setNTrials={setNTrials}
          estimators={estimators}
          setEstimators={setEstimators}
          running={running}
          progress={progress}
          onRun={onRun}
          onStop={stop}
        />

        <div>
          {error && <div className="card error-card">{error}</div>}
          {results.length === 0 ? (
            <div className="card empty">
              设置左侧参数后点击“运行蒙特卡洛实验”，结果将在此实时呈现。
            </div>
          ) : (
            <div className="charts">
              <div className="card">
                <h2>图表控制</h2>
                <div className="field">
                  <label>固定维度 p（用于折线图）</label>
                  <div className="chips">
                    {psInResults.map((p) => (
                      <span
                        key={p}
                        className={`chip ${effP === p ? 'active' : ''}`}
                        onClick={() => setSelectedP(p)}
                      >
                        {p}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="field">
                  <label>选择估计量（用于热力图 / 曲面）</label>
                  <div className="chips">
                    {estsInResults.map((e) => (
                      <span
                        key={e}
                        className={`chip ${effEst === e ? 'active' : ''}`}
                        onClick={() => setSelectedEstimator(e)}
                      >
                        {e}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              <div className="card">
                <h2>有限样本表现</h2>
                <div className="chart-grid-2">
                  <CoverageChart
                    results={results}
                    estimators={estimators}
                    selectedP={effP}
                  />
                  <WeightErrorChart
                    results={results}
                    estimators={estimators}
                    selectedP={effP}
                  />
                </div>
              </div>

              <div className="card">
                <h2>RMSE 结构</h2>
                <div className="chart-grid-2">
                  <RmseHeatmap results={results} selectedEstimator={effEst} />
                  <RmseSurface results={results} selectedEstimator={effEst} />
                </div>
              </div>

              <ResultTable results={results} estimators={estimators} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
