// 数据生成过程（DGP）：围绕分层随机试验构造可控蒙特卡洛数据
// 支持：样本量 n、协变量维度 p、四种相关结构、处理效应异质性、稀疏基线

import { RNG, gaussian, multivariateNormal, shuffle } from './rng';
import { quantile, mean, std } from './linalg';
import type { DGPParams, Dataset } from './types';

/** 构造协方差矩阵（保证正定） */
export function covarianceMatrix(
  p: number,
  corrType: string,
  rho: number,
  blockSize?: number,
): number[][] {
  const Sigma: number[][] = Array.from({ length: p }, () =>
    new Array(p).fill(0),
  );
  if (corrType === 'independent') {
    for (let i = 0; i < p; i++) Sigma[i][i] = 1;
  } else if (corrType === 'exchangeable') {
    for (let i = 0; i < p; i++)
      for (let j = 0; j < p; j++) Sigma[i][j] = i === j ? 1 : rho;
  } else if (corrType === 'ar1') {
    for (let i = 0; i < p; i++)
      for (let j = 0; j < p; j++) Sigma[i][j] = Math.pow(Math.abs(rho), Math.abs(i - j));
  } else if (corrType === 'block') {
    const bs = blockSize || Math.max(1, Math.floor(p / 4));
    for (let i = 0; i < p; i++)
      for (let j = 0; j < p; j++) {
        const bi = Math.floor(i / bs);
        const bj = Math.floor(j / bs);
        Sigma[i][j] = bi === bj ? (i === j ? 1 : rho) : 0;
      }
  } else {
    for (let i = 0; i < p; i++) Sigma[i][i] = 1;
  }
  for (let i = 0; i < p; i++) Sigma[i][i] += 1e-6;
  return Sigma;
}

export function generateCovariates(
  rng: RNG,
  n: number,
  p: number,
  corrType: string,
  rho: number,
  blockSize?: number,
): number[][] {
  const Sigma = covarianceMatrix(p, corrType, rho, blockSize);
  const m = new Array(p).fill(0);
  const X: number[][] = new Array(n);
  for (let i = 0; i < n; i++) X[i] = multivariateNormal(rng, m, Sigma);
  return X;
}

/** 基于第一协变量的分位数分层 */
export function generateStrata(rng: RNG, X: number[][], nStrata: number): number[] {
  void rng;
  const first = X.map((r) => r[0])
    .slice()
    .sort((a, b) => a - b);
  const qs: number[] = [];
  for (let k = 0; k <= nStrata; k++) qs.push(quantile(first, k / nStrata));
  qs[nStrata] += 1e-8;
  return X.map((r) => {
    let s = 0;
    while (s < nStrata - 1 && r[0] >= qs[s + 1]) s++;
    return s;
  });
}

/** 处理分配：完全随机或分层随机 */
export function assignTreatment(
  rng: RNG,
  strata: number[],
  treatProb: number,
  assignment: string,
): number[] {
  const n = strata.length;
  const W = new Array(n).fill(0);
  if (assignment === 'complete') {
    const nTreat = Math.round(n * treatProb);
    const idx = shuffle([...Array(n).keys()], rng);
    for (let k = 0; k < nTreat; k++) W[idx[k]] = 1;
  } else {
    const groups = new Map<number, number[]>();
    for (let i = 0; i < n; i++) {
      if (!groups.has(strata[i])) groups.set(strata[i], []);
      groups.get(strata[i])!.push(i);
    }
    for (const idx of groups.values()) {
      const nTreat = Math.round(idx.length * treatProb);
      const shuffled = shuffle(idx, rng);
      for (let k = 0; k < nTreat; k++) W[shuffled[k]] = 1;
    }
  }
  return W;
}

/** 生成真实系数（基线 beta 与异质性 gamma），供训练集与测试集共享，
 * 保证“样本外”评估来自同一总体（同一组真实系数）。 */
export function generateCoef(
  rng: RNG,
  p: number,
  params: DGPParams,
): { beta: number[]; gamma: number[] | null } {
  let beta: number[];
  if (params.sparse) {
    const s = Math.max(1, Math.min(p, 5));
    beta = new Array(p).fill(0);
    for (let j = 0; j < s; j++) beta[j] = gaussian(rng);
  } else {
    beta = new Array(p).fill(0).map(() => gaussian(rng) * 0.5);
  }
  let gamma: number[] | null = null;
  if (params.heteroType === 'linear') {
    if (params.sparse) {
      const s = Math.max(1, Math.min(p, 5));
      gamma = new Array(p).fill(0);
      for (let j = 0; j < s; j++) gamma[j] = gaussian(rng);
    } else {
      gamma = new Array(p).fill(0).map(() => gaussian(rng) * 0.5);
    }
  }
  return { beta, gamma };
}

/** 生成潜在结果与观测结果（使用给定的真实系数 coef） */
export function generateOutcomes(
  rng: RNG,
  X: number[][],
  W: number[],
  params: DGPParams,
  coef: { beta: number[]; gamma: number[] | null },
): { Y: number[]; Y0: number[]; Y1: number[]; tau: number[] } {
  const n = X.length;
  const { beta, gamma } = coef;
  const baseline = X.map((r) => r.reduce((acc, v, j) => acc + v * beta[j], 0));

  let tau: number[];
  if (params.heteroType === 'homogeneous') {
    tau = new Array(n).fill(params.ate);
  } else if (params.heteroType === 'linear') {
    const g = gamma as number[];
    const raw = X.map((r) => r.reduce((acc, v, j) => acc + v * g[j], 0));
    const m = mean(raw);
    const sd = std(raw) || 1;
    tau = raw.map((v) => params.ate + (params.heteroStrength * (v - m)) / sd);
  } else {
    const raw = X.map((r) => Math.sin(r[0]) * r[1] + r[2] * r[2]);
    const m = mean(raw);
    const sd = std(raw) || 1;
    tau = raw.map((v) => params.ate + (params.heteroStrength * (v - m)) / sd);
  }

  const Y0 = baseline.map((v) => v + gaussian(rng) * params.noiseSd);
  const Y1 = Y0.map((v, i) => v + tau[i]);
  const Y = W.map((w, i) => (w ? Y1[i] : Y0[i]));
  return { Y, Y0, Y1, tau };
}

/** 生成一次完整模拟数据；若传入 coef 则复用（保证样本外评估的同总体性） */
export function makeDataset(
  params: DGPParams,
  rng: RNG,
  coef?: { beta: number[]; gamma: number[] | null },
): Dataset {
  const X = generateCovariates(
    rng,
    params.n,
    params.p,
    params.corrType,
    params.rho,
    params.blockSize,
  );
  const strata = generateStrata(rng, X, params.nStrata);
  const W = assignTreatment(rng, strata, params.treatProb, params.assignment);
  const c = coef ?? generateCoef(rng, params.p, params);
  const { Y, Y0, Y1, tau } = generateOutcomes(rng, X, W, params, c);
  return { X, W, Y, strata, Y0, Y1, tau, ateTrue: params.ate, coef: c };
}
