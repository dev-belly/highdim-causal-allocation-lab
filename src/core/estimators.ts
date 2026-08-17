// 四种处理效应（ATE）估计量
// 1) 差分均值  2) OLS 调整  3) Lasso 调整（Robinson 偏残差）
// 4) 交叉拟合 DML（Double Machine Learning）

import { mean, solve, inverse } from './linalg';
import { lassoFit, lassoCV } from './lasso';
import type { Dataset, EstimatorName, EstimatorResult } from './types';

/** 1) 差分均值：无协变量调整 */
export function diffInMeans(Y: number[], W: number[]): EstimatorResult {
  const y1: number[] = [];
  const y0: number[] = [];
  for (let i = 0; i < Y.length; i++) (W[i] ? y1 : y0).push(Y[i]);
  const n1 = y1.length;
  const n0 = y0.length;
  const m1 = mean(y1);
  const m0 = mean(y0);
  const v1 = n1 > 1 ? varianceOf(y1) : 0;
  const v0 = n0 > 1 ? varianceOf(y0) : 0;
  const tau = m1 - m0;
  const se = Math.sqrt(v1 / Math.max(1, n1) + v0 / Math.max(1, n0));
  return { tau, se };
}

function varianceOf(a: number[]): number {
  const n = a.length;
  if (n < 2) return 0;
  const m = mean(a);
  let s = 0;
  for (const v of a) s += (v - m) * (v - m);
  return s / (n - 1);
}

/** 2) OLS 调整：Y ~ W + X（含处理指示与全部协变量） */
export function olsAdjusted(
  Y: number[],
  W: number[],
  X: number[][],
): EstimatorResult {
  const n = Y.length;
  const M = Y.map((_, i) => [1, W[i], ...X[i]]);
  const k = M[0].length;
  const MtM: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  const MtY = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < k; a++) {
      MtY[a] += M[i][a] * Y[i];
      for (let b = 0; b < k; b++) MtM[a][b] += M[i][a] * M[i][b];
    }
  }
  for (let a = 0; a < k; a++) MtM[a][a] += 1e-8;
  const beta = solve(MtM, MtY);
  const tau = beta[1];
  const MtMinv = inverse(MtM);
  let sse = 0;
  for (let i = 0; i < n; i++) {
    let fit = 0;
    for (let a = 0; a < k; a++) fit += M[i][a] * beta[a];
    const r = Y[i] - fit;
    sse += r * r;
  }
  const sigma2 = sse / Math.max(1, n - k);
  const se = Math.sqrt(sigma2 * Math.max(1e-12, MtMinv[1][1]));
  return { tau, se };
}

/** 3) Lasso 调整：Robinson 偏残差形式，m(X) 用 Lasso */
export function lassoAdjusted(
  Y: number[],
  W: number[],
  X: number[][],
  fixedLambda: number | null = null,
): EstimatorResult {
  const n = Y.length;
  const lambda = fixedLambda ?? lassoCV(X, Y);
  const model = lassoFit(X, Y, lambda);
  const mHat = model.predict(X);
  const eHat = mean(W);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const w = W[i] - eHat;
    num += (Y[i] - mHat[i]) * w;
    den += w * w;
  }
  den = den || 1e-12;
  const tau = num / den;
  let seAcc = 0;
  for (let i = 0; i < n; i++) {
    const w = W[i] - eHat;
    const resid = Y[i] - mHat[i] - tau * w;
    seAcc += resid * resid * w * w;
  }
  const se = Math.sqrt(seAcc / (den * den));
  return { tau, se };
}

/** 4) 交叉拟合 DML：K 折，nuisance 用 Lasso，全局 lambda 由 CV 选定 */
export function crossFittingDML(
  Y: number[],
  W: number[],
  X: number[][],
  nFolds = 5,
  fixedLambda: number | null = null,
): EstimatorResult {
  const n = Y.length;
  const foldSize = Math.floor(n / nFolds);
  const folds: number[][] = Array.from({ length: nFolds }, () => []);
  for (let i = 0; i < n; i++) folds[Math.floor(i / foldSize) % nFolds].push(i);
  const mHat = new Array(n).fill(0);
  const lambda = fixedLambda ?? lassoCV(X, Y);
  for (const valIdx of folds) {
    const valSet = new Set(valIdx);
    const trainIdx = [...Array(n).keys()].filter((i) => !valSet.has(i));
    const Xtr = trainIdx.map((i) => X[i]);
    const ytr = trainIdx.map((i) => Y[i]);
    const model = lassoFit(Xtr, ytr, lambda);
    for (const i of valIdx) mHat[i] = model.predict([X[i]])[0];
  }
  const eHat = mean(W);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const w = W[i] - eHat;
    num += (Y[i] - mHat[i]) * w;
    den += w * w;
  }
  den = den || 1e-12;
  const tau = num / den;
  let seAcc = 0;
  for (let i = 0; i < n; i++) {
    const w = W[i] - eHat;
    const resid = Y[i] - mHat[i] - tau * w;
    seAcc += resid * resid * w * w;
  }
  const se = Math.sqrt(seAcc / (den * den));
  return { tau, se };
}

export interface EstimatorOptions {
  fixedLambda?: number | null;
  nFolds?: number;
}

export function fitEstimator(
  name: EstimatorName,
  data: Dataset,
  opts: EstimatorOptions = {},
): EstimatorResult {
  const { Y, W, X } = data;
  switch (name) {
    case 'diff_in_means':
      return diffInMeans(Y, W);
    case 'ols_adjusted':
      return olsAdjusted(Y, W, X);
    case 'lasso_adjusted':
      return lassoAdjusted(Y, W, X, opts.fixedLambda ?? null);
    case 'cross_fitting_dml':
      return crossFittingDML(Y, W, X, opts.nFolds ?? 5, opts.fixedLambda ?? null);
  }
}
