import { useState } from 'react';
import { ControlPanel } from './components/ControlPanel';
import { designError, runParameters, type BaseParams } from './core/design';
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

const RUN_SEED = 42;

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
  const [trialsPerCell, setTrialsPerCell] = useState(0);

  const { progress, results, running, error, start, stop } = useSimulation();

  const psInResults = results.length
    ? [...new Set(results.map((c) => c.params.p))]
    : pLevels;
  const effP = psInResults.includes(selectedP) ? selectedP : psInResults[0] ?? selectedP;
  const estsInResults = results.length ? Object.keys(results[0].summary) as EstimatorName[] : estimators;
  const effEst = estsInResults.includes(selectedEstimator)
    ? selectedEstimator
    : estsInResults[0] ?? selectedEstimator;

  const runIssue = nLevels.length && pLevels.length && estimators.length
    ? designError(base, nLevels, pLevels, nTrials)
    : '请至少选择一个样本量、一个维度和一个估计量。';
  const stage = running ? '运行中' : error ? '运行失败' : progress.done === progress.total ? '运行完成' : '已停止';

  function onRun() {
    if (running || runIssue) return;
    const settings = runParameters(base);
    const grid: DGPParams[] = nLevels.flatMap((n) => pLevels.map((p) => ({ ...settings, n, p })));
    setTrialsPerCell(nTrials);
    setSelectedP(pLevels[0]);
    start(grid, estimators, nTrials, RUN_SEED);
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>高维因果与稳健配置</h1>
          <div className="subtitle">
            运行蒙特卡洛模拟，对照四种估计量的有限样本表现，再比较它们带来的配置偏差。
          </div>
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
          runIssue={runIssue}
          onRun={onRun}
          onStop={stop}
        />

        <div>
          {progress.total > 0 && (
            <section className="card run-summary" aria-labelledby="run-heading">
              <div>
                <h2 id="run-heading">本次实验</h2>
                <p role="status">{stage} · {progress.done} / {progress.total} 组完成</p>
              </div>
              <dl>
                <div><dt>每组试验</dt><dd>{trialsPerCell}</dd></div>
                <div><dt>随机种子</dt><dd>{RUN_SEED}</dd></div>
              </dl>
            </section>
          )}
          {error && <div className="card error-card" role="alert">{error}</div>}
          {results.length === 0 ? (
            <div className="card empty">{running ? '正在计算首组结果…' : '完成的参数组将在这里显示。'}</div>
          ) : (
            <div className="charts">
              <div className="card">
                <h2>图表控制</h2>
                <div className="field">
                  <label>固定维度 p（用于折线图）</label>
                  <div className="chips">
                    {psInResults.map((p) => (
                      <button
                        key={p}
                        type="button"
                        aria-pressed={effP === p}
                        className={`chip ${effP === p ? 'active' : ''}`}
                        onClick={() => setSelectedP(p)}
                      >
                        {p}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="field">
                  <label>选择估计量（用于热力图 / 曲面）</label>
                  <div className="chips">
                    {estsInResults.map((e) => (
                      <button
                        key={e}
                        type="button"
                        aria-pressed={effEst === e}
                        className={`chip ${effEst === e ? 'active' : ''}`}
                        onClick={() => setSelectedEstimator(e)}
                      >
                        {e}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="card">
                <h2>有限样本表现</h2>
                <div className="chart-grid-2">
                  <CoverageChart
                    results={results}
                    estimators={estsInResults}
                    selectedP={effP}
                  />
                  <WeightErrorChart
                    results={results}
                    estimators={estsInResults}
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

              <ResultTable results={results} estimators={estsInResults} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
