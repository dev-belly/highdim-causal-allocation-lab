// 蒙特卡洛模拟引擎：单试验 → 聚合指标
// 纯函数，可在主线程、Web Worker 或 Node 测试中复用

import { mulberry32 } from './rng';
import { makeDataset } from './dgp';
import { fitEstimator } from './estimators';
import { evaluateAllocation } from './portfolio';
import { confidenceInterval, mean, std, variance } from './linalg';
import type {
  DGPParams,
  EstimatorName,
  TrialMetric,
  EstimatorSummary,
} from './types';

const DEFAULT_ALLOC = {
  riskAversion: 1,
  allowShort: false,
  priorVar: 1.0,
  priorMean: 0.0,
};

/** 只用随机试验中实际观测到的处理组结果估计风险资产方差。 */
export function observedTreatedVariance(Y: number[], W: number[]): number {
  const treated = Y.filter((_, i) => W[i] === 1);
  if (treated.length < 2) throw new Error('At least two treated observations are required');
  return Math.max(variance(treated), 1e-6);
}

/** 运行一次试验：返回每个估计量的指标（含独立测试集的样本外评估） */
export function runTrial(
  params: DGPParams,
  estimators: EstimatorName[],
  seed: number,
): TrialMetric[] {
  const rng = mulberry32(seed);
  const data = makeDataset(params, rng);
  // 样本外测试集复用同一组真实系数（同总体），仅重新抽取协变量/处理/噪声
  const testRng = mulberry32(seed + 1_000_000);
  const test = makeDataset(params, testRng, data.coef);
  // 未接受处理者的 Y1 是不可观测的潜在结果，不能用于选择权重。
  const sigma2 = observedTreatedVariance(data.Y, data.W);

  const out: TrialMetric[] = [];
  for (const name of estimators) {
    const res = fitEstimator(name, data);
    const pf = evaluateAllocation(
      res.tau,
      params.ate,
      res.se,
      sigma2,
      test.Y1,
      DEFAULT_ALLOC,
    );
    out.push({
      tau: res.tau,
      se: res.se,
      ateTrue: params.ate,
      estimator: name,
      ...pf,
    });
  }
  return out;
}

/** 聚合多次试验：偏差、标准差、RMSE、平均 SE、覆盖率、配置指标 */
export function aggregate(
  results: TrialMetric[][],
  ateTrue: number,
): Record<EstimatorName, EstimatorSummary> {
  const byEst: Record<string, TrialMetric[]> = {};
  for (const trial of results) {
    for (const m of trial) {
      if (!byEst[m.estimator]) byEst[m.estimator] = [];
      byEst[m.estimator].push(m);
    }
  }
  const summary = {} as Record<EstimatorName, EstimatorSummary>;
  for (const est of Object.keys(byEst) as EstimatorName[]) {
    const arr = byEst[est];
    const taus = arr.map((m) => m.tau);
    const ses = arr.map((m) => m.se);
    const bias = mean(taus) - ateTrue;
    const sd = std(taus);
    const rmse = Math.sqrt(mean(taus.map((t) => (t - ateTrue) ** 2)));
    const meanSe = mean(ses);
    let covered = 0;
    for (let i = 0; i < arr.length; i++) {
      const [lo, hi] = confidenceInterval(taus[i], ses[i], 0.95);
      if (lo <= ateTrue && ateTrue <= hi) covered++;
    }
    const coverage = covered / Math.max(1, arr.length);
    summary[est] = {
      bias,
      sd,
      rmse,
      meanSe,
      coverage,
      meanWeightAbsErr: mean(arr.map((m) => m.weightAbsErr)),
      meanUtilityLoss: mean(arr.map((m) => m.utilityLoss)),
      meanOosUtility: mean(arr.map((m) => m.oosUtility)),
      meanOosUtilityShrunk: mean(arr.map((m) => m.oosUtilityShrunk)),
    };
  }
  return summary;
}
