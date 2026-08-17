// 稳健资产配置：将 ATE 估计误差映射到均值-方差最优权重偏移与样本外风险
// 并提供“收缩基准”（Bayes–Stein 风格向真实最优权重收缩）以对比无约束估计

import { mean, std, variance } from './linalg';

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
  shrinkageIntensity: number; // 收缩基准的收缩强度（0=无约束，1=完全用理论权重）
}

/**
 * 评估一次试验的资产配置表现。
 * @param tauHat 估计的 ATE
 * @param tauTrue 真实 ATE
 * @param sigma2True 收益方差（真实/样本）
 * @param testY1 独立测试集：处理组结果
 * @param testY0 独立测试集：对照组结果
 */
export function evaluateAllocation(
  tauHat: number,
  tauTrue: number,
  sigma2True: number,
  testY1: number[],
  testY0: number[],
  cfg: AllocationConfig,
): PortfolioOutcome {
  const { riskAversion, allowShort, shrinkageIntensity } = cfg;
  const wStar = meanVarianceWeight(tauTrue, sigma2True, riskAversion, allowShort);
  const wHat = meanVarianceWeight(tauHat, sigma2True, riskAversion, allowShort);
  const wShrunk =
    (1 - shrinkageIntensity) * wHat + shrinkageIntensity * wStar;

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
