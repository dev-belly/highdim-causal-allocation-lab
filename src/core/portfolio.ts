// 稳健资产配置：将 ATE 估计误差映射到均值-方差最优权重偏移与样本外风险
// 并提供“收缩基准”：用真实、数据驱动的 Bayes–Stein 收缩将估计 ATE 向先验均值收缩，
// 以对比“无约束（直接用估计 ATE）”与“收缩后”的样本外表现。

import { mean, std, variance } from './linalg';
import { bayesSteinShrinkage } from './shrinkage';

/** 单期均值-方差最优权重（风险资产 vs 无风险基准） */
export function meanVarianceWeight(
  mu: number,
  sigma2: number,
  riskAversion: number,
  allowShort: boolean,
): number {
  const s2 = sigma2 > 0 ? sigma2 : 1e-6;
  let w = mu / (riskAversion * s2);
  if (!allowShort) w = Math.min(1, Math.max(0, w));
  return w;
}

export interface PortfolioOutcome {
  wStar: number; // 基于真实 ATE 的理论最优权重
  wHat: number; // 基于估计 ATE 的无约束权重
  wShrunk: number; // 收缩后的权重（向 wStar 收缩）
  weightBias: number;
  weightAbsErr: number;
  utilityLoss: number; // 相对理论最优的确定性等价效用损失
  oosSharpe: number; // 无约束组合的样本外夏普
  oosSharpeShrunk: number; // 收缩组合的样本外夏普
}

export interface AllocationConfig {
  riskAversion: number;
  allowShort: boolean;
  /** Bayes–Stein 先验方差：越大 → 收缩越弱（更信任估计值） */
  priorVar: number;
  /** 先验均值：单资产设定下向无风险/零超额收益收缩 */
  priorMean: number;
}

/**
 * 评估一次试验的资产配置表现。
 * @param tauHat 估计的 ATE
 * @param tauTrue 真实 ATE（仅用于计算 oracle 参考 wStar / 效用损失，不用于收缩）
 * @param seHat 估计 ATE 的标准误（驱动数据驱动的收缩强度）
 * @param sigma2True 风险资产（处理组）收益方差
 * @param testY1 独立测试集：处理组结果
 * @param testY0 独立测试集：对照组结果
 */
export function evaluateAllocation(
  tauHat: number,
  tauTrue: number,
  seHat: number,
  sigma2True: number,
  testY1: number[],
  testY0: number[],
  cfg: AllocationConfig,
): PortfolioOutcome {
  const { riskAversion, allowShort, priorVar, priorMean } = cfg;
  const wStar = meanVarianceWeight(tauTrue, sigma2True, riskAversion, allowShort);
  const wHat = meanVarianceWeight(tauHat, sigma2True, riskAversion, allowShort);

  // 数据驱动的 Bayes–Stein 收缩：将估计 ATE 向先验均值收缩，
  // 收缩强度由估计方差 se^2 与先验方差 priorVar 之比决定（不依赖真值）。
  const tauShrunk = bayesSteinShrinkage(
    [tauHat],
    [seHat * seHat],
    priorVar,
    priorMean,
  )[0];
  const wShrunk = meanVarianceWeight(tauShrunk, sigma2True, riskAversion, allowShort);

  const weightBias = wHat - wStar;
  const weightAbsErr = Math.abs(weightBias);

  const utilityStar = wStar * tauTrue - 0.5 * riskAversion * wStar * wStar * sigma2True;
  const utilityHat = wHat * tauTrue - 0.5 * riskAversion * wHat * wHat * sigma2True;
  const utilityLoss = utilityStar - utilityHat;

  const port = testY1.map((y1, i) => wHat * y1 + (1 - wHat) * testY0[i]);
  const portShrunk = testY1.map((y1, i) => wShrunk * y1 + (1 - wShrunk) * testY0[i]);
  const oosSharpe = mean(port) / (std(port) + 1e-9);
  const oosSharpeShrunk = mean(portShrunk) / (std(portShrunk) + 1e-9);

  return {
    wStar,
    wHat,
    wShrunk,
    weightBias,
    weightAbsErr,
    utilityLoss,
    oosSharpe,
    oosSharpeShrunk,
  };
}

/** 由观测结果估计收益方差（用于配置尺度） */
export function empiricalVariance(Y: number[]): number {
  return variance(Y) || 1e-6;
}
