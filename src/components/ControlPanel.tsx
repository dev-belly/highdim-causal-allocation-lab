import { MAX_TRIALS, MIN_TRIALS, maximumStrata, type BaseParams } from '../core/design';
import type { CorrType, HeteroType, Assignment, EstimatorName } from '../core/types';

interface ChipOption<T> {
  value: T;
  label: string;
}

interface ControlPanelProps {
  base: BaseParams;
  setBase: (b: BaseParams) => void;
  nLevels: number[];
  setNLevels: (v: number[]) => void;
  pLevels: number[];
  setPLevels: (v: number[]) => void;
  nTrials: number;
  setNTrials: (v: number) => void;
  estimators: EstimatorName[];
  setEstimators: (v: EstimatorName[]) => void;
  running: boolean;
  progress: { done: number; total: number };
  runIssue: string;
  onRun: () => void;
  onStop: () => void;
}

const N_OPTIONS = [200, 500, 1000, 2000];
const P_OPTIONS = [5, 20, 50, 100];
const EST_OPTIONS: ChipOption<EstimatorName>[] = [
  { value: 'diff_in_means', label: '差分均值' },
  { value: 'ols_adjusted', label: 'OLS 调整' },
  { value: 'lasso_adjusted', label: 'Lasso 调整' },
  { value: 'cross_fitting_dml', label: '交叉拟合 DML' },
];
const CORR_OPTIONS: ChipOption<CorrType>[] = [
  { value: 'independent', label: '独立' },
  { value: 'exchangeable', label: '可交换' },
  { value: 'ar1', label: 'AR(1)' },
  { value: 'block', label: '分块' },
];
const HET_OPTIONS: ChipOption<HeteroType>[] = [
  { value: 'homogeneous', label: '同质' },
  { value: 'linear', label: '线性异质' },
  { value: 'nonlinear', label: '非线性异质' },
];
const ASSIGN_OPTIONS: ChipOption<Assignment>[] = [
  { value: 'stratified', label: '分层随机' },
  { value: 'complete', label: '完全随机' },
];

function toggle<T>(arr: T[], v: T): T[] {
  return arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v];
}

function numericDisplay(value: number) {
  return Number.isFinite(value) ? value : '';
}

export function ControlPanel(props: ControlPanelProps) {
  const {
    base, setBase, nLevels, setNLevels, pLevels, setPLevels,
    nTrials, setNTrials, estimators, setEstimators, running, progress, runIssue, onRun, onStop,
  } = props;

  const set = <K extends keyof BaseParams>(key: K, val: BaseParams[K]) =>
    setBase({ ...base, [key]: val });

  const gridSize = nLevels.length * pLevels.length;

  return (
    <form className="card" aria-labelledby="design-heading" noValidate onSubmit={(event) => {
      event.preventDefault();
      if (!running && !runIssue) onRun();
    }}>
      <h2 id="design-heading">实验设计</h2>

      <div className="field">
        <label htmlFor="effect">处理效应 ATE</label>
        <input
          id="effect"
          type="number"
          step="any"
          value={numericDisplay(base.ate)}
          onChange={(event) => set('ate', event.currentTarget.valueAsNumber)}
        />
      </div>

      <div className="field">
        <label>协变量相关结构</label>
        <div className="chips">
          {CORR_OPTIONS.map((o) => (
            <button
              key={o.value}
              type="button"
              aria-pressed={base.corrType === o.value}
              className={`chip ${base.corrType === o.value ? 'active' : ''}`}
              onClick={() => set('corrType', o.value)}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>

      <div className="row">
        <div className="field">
          <label htmlFor="correlation">相关系数 ρ</label>
          <input
            id="correlation"
            type="number"
            step="any"
            disabled={base.corrType === 'independent'}
            value={numericDisplay(base.rho)}
            onChange={(event) => set('rho', event.currentTarget.valueAsNumber)}
          />
        </div>
        <div className="field">
          <label htmlFor="noise">噪声 σ</label>
          <input
            id="noise"
            type="number"
            step="any"
            min="0"
            value={numericDisplay(base.noiseSd)}
            onChange={(event) => set('noiseSd', event.currentTarget.valueAsNumber)}
          />
        </div>
      </div>

      <div className="field">
        <label>处理效应异质性</label>
        <div className="chips">
          {HET_OPTIONS.map((o) => (
            <button
              key={o.value}
              type="button"
              aria-pressed={base.heteroType === o.value}
              className={`chip ${base.heteroType === o.value ? 'active' : ''}`}
              onClick={() => set('heteroType', o.value)}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>

      <div className="row">
        <div className="field">
          <label htmlFor="heterogeneity">异质强度</label>
          <input
            id="heterogeneity"
            type="number"
            step="any"
            min="0"
            disabled={base.heteroType === 'homogeneous'}
            value={numericDisplay(base.heteroStrength)}
            onChange={(event) => set('heteroStrength', event.currentTarget.valueAsNumber)}
          />
        </div>
        <div className="field">
          <label htmlFor="strata">分层数</label>
          <input
            id="strata"
            type="number"
            step="1"
            min="2"
            max={nLevels.length ? maximumStrata(nLevels) : undefined}
            disabled={base.assignment === 'complete'}
            value={numericDisplay(base.nStrata)}
            onChange={(event) => set('nStrata', event.currentTarget.valueAsNumber)}
          />
        </div>
      </div>

      <div className="field">
        <label>分配机制</label>
        <div className="chips">
          {ASSIGN_OPTIONS.map((o) => (
            <button
              key={o.value}
              type="button"
              aria-pressed={base.assignment === o.value}
              className={`chip ${base.assignment === o.value ? 'active' : ''}`}
              onClick={() => set('assignment', o.value)}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>

      <div className="field">
        <label>样本量扫描 n</label>
        <div className="chips">
          {N_OPTIONS.map((v) => (
            <button
              key={v}
              type="button"
              aria-pressed={nLevels.includes(v)}
              className={`chip ${nLevels.includes(v) ? 'active' : ''}`}
              onClick={() => setNLevels(toggle(nLevels, v))}
            >
              {v}
            </button>
          ))}
        </div>
      </div>

      <div className="field">
        <label>维度扫描 p</label>
        <div className="chips">
          {P_OPTIONS.map((v) => (
            <button
              key={v}
              type="button"
              aria-pressed={pLevels.includes(v)}
              className={`chip ${pLevels.includes(v) ? 'active' : ''}`}
              onClick={() => setPLevels(toggle(pLevels, v))}
            >
              {v}
            </button>
          ))}
        </div>
      </div>

      <div className="field">
        <label htmlFor="trials">每组蒙特卡洛试验次数</label>
        <input
          id="trials"
          type="number"
          step="1"
          min={MIN_TRIALS}
          max={MAX_TRIALS}
          value={numericDisplay(nTrials)}
          onChange={(event) => setNTrials(event.currentTarget.valueAsNumber)}
        />
        <div className="hint">默认 10 次便于快速演示；比较覆盖率时请增加重复次数，运行时间也会相应增长。</div>
      </div>

      <div className="field">
        <label>估计量（可多选）</label>
        <div className="chips">
          {EST_OPTIONS.map((o) => (
            <button
              key={o.value}
              type="button"
              aria-pressed={estimators.includes(o.value)}
              className={`chip ${estimators.includes(o.value) ? 'active' : ''}`}
              onClick={() => setEstimators(toggle(estimators, o.value))}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>

      {running ? (
        <button className="btn btn-ghost" type="button" onClick={onStop}>
          停止实验
        </button>
      ) : (
        <>
          <p id="run-check" className={runIssue ? 'run-check invalid' : 'run-check'}>
            {runIssue || `${gridSize} 组参数 · ${estimators.length} 种估计量 · 共 ${gridSize * nTrials} 次试验`}
          </p>
          <button className="btn btn-primary" type="submit" disabled={Boolean(runIssue)} aria-describedby="run-check">
            运行蒙特卡洛实验
          </button>
        </>
      )}

      {running && (
        <div className="progress-wrap">
          <progress className="progress-track" aria-label="已完成参数组数" value={progress.done} max={progress.total} />
          <div className="progress-text">
            <span>模拟中…</span>
            <span>
              {progress.done} / {progress.total}
            </span>
          </div>
        </div>
      )}
    </form>
  );
}
