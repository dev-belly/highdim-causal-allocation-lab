// Lasso 回归：坐标下降求解 + K 折交叉验证选正则化参数
// 用于高维（p ≫ n 或 p 较大）场景下的条件均值估计

import { columnStandardize, softThreshold } from './linalg';

export interface LassoModel {
  predict(Xnew: number[][]): number[];
}

/** 坐标下降 Lasso（输入已标准化）。返回可在任意 Xnew 上预测的函数 */
export function lassoFit(
  X: number[][],
  y: number[],
  lambda: number,
  maxIter = 200,
  tol = 1e-7,
): LassoModel {
  const n = X.length;
  if (n === 0 || y.length !== n || X.some((row) => row.length !== X[0].length)) {
    throw new Error('Lasso requires non-empty, rectangular X and matching y');
  }
  const p = X[0].length;
  const { Xs, means, stds } = columnStandardize(X);
  const yMean = y.reduce((sum, value) => sum + value, 0) / n;
  const xjxj = new Array(p);
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += Xs[i][j] * Xs[i][j];
    xjxj[j] = s || 1e-12;
  }
  const beta = new Array(p).fill(0);
  let r = y.map((value) => value - yMean);
  for (let it = 0; it < maxIter; it++) {
    let maxChange = 0;
    for (let j = 0; j < p; j++) {
      let xjr = 0;
      for (let i = 0; i < n; i++) xjr += Xs[i][j] * r[i];
      const z = (xjr + beta[j] * xjxj[j]) / xjxj[j];
      const nb = softThreshold(z, lambda);
      const delta = nb - beta[j];
      if (delta !== 0) {
        for (let i = 0; i < n; i++) r[i] -= delta * Xs[i][j];
      }
      maxChange = Math.max(maxChange, Math.abs(delta));
      beta[j] = nb;
    }
    if (maxChange < tol) break;
  }
  return {
    predict(Xnew: number[][]) {
      return Xnew.map((row) => {
        let s = yMean;
        for (let j = 0; j < p; j++) s += ((row[j] - means[j]) / stds[j]) * beta[j];
        return s;
      });
    },
  };
}

/** K 折交叉验证选择最优 lambda（log 空间 20 个候选） */
export function lassoCV(X: number[][], y: number[], nFolds = 5): number {
  const n = X.length;
  if (n < 2 || y.length !== n) {
    throw new Error('Lasso cross-validation requires at least two matching samples');
  }
  const lambdas: number[] = [];
  for (let k = 0; k < 20; k++) lambdas.push(Math.pow(10, -3 + (k * 3) / 19));
  const k = Math.max(2, Math.min(nFolds, n));
  const foldOf = new Array(n);
  for (let i = 0; i < n; i++) foldOf[i] = i % k;
  const cvErr = new Array(lambdas.length).fill(0);
  for (let f = 0; f < k; f++) {
    const trainIdx: number[] = [];
    const valIdx: number[] = [];
    for (let i = 0; i < n; i++) (foldOf[i] === f ? valIdx : trainIdx).push(i);
    const Xtr = trainIdx.map((i) => X[i]);
    const ytr = trainIdx.map((i) => y[i]);
    const Xv = valIdx.map((i) => X[i]);
    const yv = valIdx.map((i) => y[i]);
    lambdas.forEach((lam, fk) => {
      const model = lassoFit(Xtr, ytr, lam);
      const pred = model.predict(Xv);
      let e = 0;
      for (let i = 0; i < valIdx.length; i++) {
        const d = pred[i] - yv[i];
        e += d * d;
      }
      cvErr[fk] += e / Math.max(1, valIdx.length);
    });
  }
  let best = 0;
  let bestErr = Infinity;
  cvErr.forEach((e, k) => {
    if (e < bestErr) {
      bestErr = e;
      best = k;
    }
  });
  return lambdas[best];
}
