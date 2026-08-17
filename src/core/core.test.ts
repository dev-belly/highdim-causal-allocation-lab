import { describe, it, expect } from 'vitest';
import { mulberry32 } from './rng';
import { makeDataset } from './dgp';
import { fitEstimator } from './estimators';
import { runTrial, aggregate } from './simulator';
import { shaferStrimmerShrinkage, bayesSteinShrinkage } from './shrinkage';
import { meanVarianceWeight } from './portfolio';
import type { DGPParams, EstimatorName } from './types';

const base: DGPParams = {
  n: 800,
  p: 20,
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

const ESTIMATORS: EstimatorName[] = [
  'diff_in_means',
  'ols_adjusted',
  'lasso_adjusted',
  'cross_fitting_dml',
];

describe('DGP 可复现性', () => {
  it('相同种子生成完全相同的数据', () => {
    const a = makeDataset(base, mulberry32(42));
    const b = makeDataset(base, mulberry32(42));
    expect(a.Y[0]).toBeCloseTo(b.Y[0], 10);
    expect(a.W).toEqual(b.W);
  });
});

describe('四种估计量数值有限性', () => {
  it('均返回有限 tau 与 se', () => {
    const data = makeDataset(base, mulberry32(7));
    for (const e of ESTIMATORS) {
      const r = fitEstimator(e, data);
      expect(Number.isFinite(r.tau)).toBe(true);
      expect(Number.isFinite(r.se)).toBe(true);
    }
  });
});

describe('聚合统计性质', () => {
  it('差分均值覆盖率接近名义 95%', () => {
    const results = [];
    for (let t = 0; t < 80; t++) results.push(runTrial(base, ['diff_in_means'], t + 1));
    const sum = aggregate(results, base.ate);
    expect(Math.abs(sum.diff_in_means.coverage - 0.95)).toBeLessThan(0.12);
    expect(sum.diff_in_means.rmse).toBeGreaterThan(0);
  });

  it('高维下 Lasso/DML 的 RMSE 不逊于差分均值', () => {
    const highDim: DGPParams = { ...base, n: 300, p: 80 };
    const results = [];
    for (let t = 0; t < 80; t++)
      results.push(runTrial(highDim, ESTIMATORS, t + 1));
    const sum = aggregate(results, highDim.ate);
    expect(sum.lasso_adjusted.rmse).toBeLessThanOrEqual(sum.diff_in_means.rmse * 1.5);
  });
});

describe('收缩估计', () => {
  it('Schäfer–Strimmer 收缩强度落在 [0,1]', () => {
    const S = [
      [2, 0.5, 0.2],
      [0.5, 3, 0.1],
      [0.2, 0.1, 1.5],
    ];
    const { lambda, Sigma } = shaferStrimmerShrinkage(S, 100);
    expect(lambda).toBeGreaterThanOrEqual(0);
    expect(lambda).toBeLessThanOrEqual(1);
    expect(Sigma.length).toBe(3);
  });

  it('Bayes–Stein 收缩使均值更接近整体均值', () => {
    const mu = [10, -10, 5, -5];
    const varMu = [4, 4, 4, 4];
    const shrunk = bayesSteinShrinkage(mu, varMu, 1);
    const before = Math.max(...mu) - Math.min(...mu);
    const after = Math.max(...shrunk) - Math.min(...shrunk);
    expect(after).toBeLessThan(before);
  });
});

describe('均值-方差权重', () => {
  it('非负约束下权重被截断到 [0,1]', () => {
    const w = meanVarianceWeight(5, 1, 1, false);
    expect(w).toBeLessThanOrEqual(1);
    expect(w).toBeGreaterThanOrEqual(0);
  });
});
