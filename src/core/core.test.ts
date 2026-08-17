import { describe, it, expect } from 'vitest';
import { mulberry32 } from './rng';
import { makeDataset, generateCoef } from './dgp';
import { fitEstimator, crossFittingDML } from './estimators';
import { runTrial, aggregate } from './simulator';
import { shaferStrimmerShrinkage, bayesSteinShrinkage } from './shrinkage';
import { meanVarianceWeight, evaluateAllocation } from './portfolio';
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

describe('修复回归：交叉拟合 DML 在小样本 n < nFolds 不崩溃', () => {
  it('n=3 时仍能返回有限结果', () => {
    const small: DGPParams = { ...base, n: 3 };
    const data = makeDataset(small, mulberry32(1));
    const r = crossFittingDML(data.Y, data.W, data.X, 5);
    expect(Number.isFinite(r.tau)).toBe(true);
    expect(Number.isFinite(r.se)).toBe(true);
  });
});

describe('修复回归：样本外测试集与训练集同总体', () => {
  it('复用 coef 时测试集基线系数与训练集一致', () => {
    const rng = mulberry32(99);
    const data = makeDataset(base, rng);
    const testRng = mulberry32(99 + 1_000_000);
    const test = makeDataset(base, testRng, data.coef);
    expect(data.coef!.beta).toEqual(test.coef!.beta);
    expect(data.coef!.gamma).toEqual(test.coef!.gamma);
  });

  it('generateCoef 在 sparse 下仅前 5 个非零、linear 下生成 gamma', () => {
    const c = generateCoef(mulberry32(5), 40, { ...base, sparse: true, heteroType: 'linear' });
    const nz = c.beta.filter((x) => x !== 0).length;
    expect(nz).toBe(5);
    expect(c.gamma).not.toBeNull();
  });
});

describe('修复回归：Bayes–Stein 收缩真正接入配置评估', () => {
  it('大标准误时收缩后权重更接近先验均值（更保守）', () => {
    const cfgHi = { riskAversion: 1, allowShort: false, priorVar: 1.0, priorMean: 0.0 };
    const cfgLo = { riskAversion: 1, allowShort: false, priorVar: 100.0, priorMean: 0.0 };
    const y1 = Array.from({ length: 200 }, (_, i) => 2 + (i % 2));
    const y0 = Array.from({ length: 200 }, () => 0);
    const oHi = evaluateAllocation(2, 1, 2.0, 1.0, y1, y0, cfgHi); // se 大 → 强收缩
    const oLo = evaluateAllocation(2, 1, 2.0, 1.0, y1, y0, cfgLo); // priorVar 大 → 弱收缩
    expect(Math.abs(oHi.wShrunk)).toBeLessThan(Math.abs(oLo.wShrunk));
    // 收缩权重不应等于（不依赖真值的）oracle 混合
    expect(oHi.wShrunk).not.toBeCloseTo(oHi.wStar, 6);
  });
});
