import type { CorrType, HeteroType, Assignment, EstimatorName, DGPParams } from '../core/types';

export type BaseParams = Omit<DGPParams, 'n' | 'p'>;

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

export function ControlPanel(props: ControlPanelProps) {
  const {
    base, setBase, nLevels, setNLevels, pLevels, setPLevels,
    nTrials, setNTrials, estimators, setEstimators, running, progress, onRun, onStop,
  } = props;

  const set = <K extends keyof BaseParams>(key: K, val: BaseParams[K]) =>
    setBase({ ...base, [key]: val });

  const pct = progress.total ? Math.round((progress.done / progress.total) * 100) : 0;

  return (
    <div className="card">
      <h2>实验设计</h2>

      <div className="field">
        <label>处理效应 ATE</label>
        <input
          type="number"
          step="0.1"
          value={base.ate}
          onChange={(e) => set('ate', Number(e.target.value))}
        />
      </div>

      <div className="field">
        <label>协变量相关结构</label>
        <div className="chips">
          {CORR_OPTIONS.map((o) => (
            <span
              key={o.value}
              className={`chip ${base.corrType === o.value ? 'active' : ''}`}
              onClick={() => set('corrType', o.value)}
            >
              {o.label}
            </span>
          ))}
        </div>
      </div>

      <div className="row">
        <div className="field">
          <label>相关系数 ρ</label>
          <input
            type="number"
            step="0.05"
            min="0"
            max="0.95"
            value={base.rho}
            onChange={(e) => set('rho', Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label>噪声 σ</label>
          <input
            type="number"
            step="0.1"
            value={base.noiseSd}
            onChange={(e) => set('noiseSd', Number(e.target.value))}
          />
        </div>
      </div>

      <div className="field">
        <label>处理效应异质性</label>
        <div className="chips">
          {HET_OPTIONS.map((o) => (
            <span
              key={o.value}
              className={`chip ${base.heteroType === o.value ? 'active' : ''}`}
              onClick={() => set('heteroType', o.value)}
            >
              {o.label}
            </span>
          ))}
        </div>
      </div>

      <div className="row">
        <div className="field">
          <label>异质强度</label>
          <input
            type="number"
            step="0.1"
            value={base.heteroStrength}
            onChange={(e) => set('heteroStrength', Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label>分层数</label>
          <input
            type="number"
            step="1"
            value={base.nStrata}
            onChange={(e) => set('nStrata', Number(e.target.value))}
          />
        </div>
      </div>

      <div className="field">
        <label>分配机制</label>
        <div className="chips">
          {ASSIGN_OPTIONS.map((o) => (
            <span
              key={o.value}
              className={`chip ${base.assignment === o.value ? 'active' : ''}`}
              onClick={() => set('assignment', o.value)}
            >
              {o.label}
            </span>
          ))}
        </div>
      </div>

      <div className="field">
        <label>样本量扫描 n</label>
        <div className="chips">
          {N_OPTIONS.map((v) => (
            <span
              key={v}
              className={`chip ${nLevels.includes(v) ? 'active' : ''}`}
              onClick={() => setNLevels(toggle(nLevels, v))}
            >
              {v}
            </span>
          ))}
        </div>
      </div>

      <div className="field">
        <label>维度扫描 p</label>
        <div className="chips">
          {P_OPTIONS.map((v) => (
            <span
              key={v}
              className={`chip ${pLevels.includes(v) ? 'active' : ''}`}
              onClick={() => setPLevels(toggle(pLevels, v))}
            >
              {v}
            </span>
          ))}
        </div>
      </div>

      <div className="field">
        <label>蒙特卡洛试验次数</label>
        <input
          type="number"
          step="10"
          min="10"
          value={nTrials}
          onChange={(e) => setNTrials(Math.max(10, Number(e.target.value)))}
        />
        <div className="hint">每个 (n,p) 单元重复抽样次数</div>
      </div>

      <div className="field">
        <label>估计量（可多选）</label>
        <div className="chips">
          {EST_OPTIONS.map((o) => (
            <span
              key={o.value}
              className={`chip ${estimators.includes(o.value) ? 'active' : ''}`}
              onClick={() => setEstimators(toggle(estimators, o.value))}
            >
              {o.label}
            </span>
          ))}
        </div>
      </div>

      {running ? (
        <button className="btn btn-ghost" onClick={onStop}>
          停止
        </button>
      ) : (
        <button className="btn btn-primary" onClick={onRun}>
          运行蒙特卡洛实验
        </button>
      )}

      {running && (
        <div className="progress-wrap">
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
          <div className="progress-text">
            <span>模拟中…</span>
            <span>
              {progress.done} / {progress.total}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
