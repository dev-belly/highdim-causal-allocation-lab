// 收缩估计（Shrinkage Estimation）
// 1) Schäfer–Strimmer 协方差收缩：基于渐近方差的解析收缩强度
// 2) Bayes–Stein 均值收缩：将收益估计向整体先验收缩
// 对应项目目标：“对比收缩估计与无约束基准”

import { mean } from './linalg';

/**
 * Schäfer–Strimmer 协方差矩阵收缩（线性收缩至对角目标）。
 * 收缩强度 lambda 由样本协方差的渐近方差比解析给出，无需重采样。
 */
export function shaferStrimmerShrinkage(
  S: number[][],
  n: number,
): { Sigma: number[][]; lambda: number } {
  const p = S.length;
  let num = 0; // Σ Var(s_ij) over i<=j
  let den = 0; // Σ s_ij^2 over i<=j
  for (let a = 0; a < p; a++) {
    for (let b = a; b < p; b++) {
      const s2 = S[a][b] * S[a][b];
      den += s2;
      let v: number;
      if (a === b) v = (2 * S[a][a] * S[a][a]) / (n - 1);
      else v = (S[a][a] * S[b][b] + S[a][b] * S[a][b]) / (n - 1);
      num += v;
    }
  }
  const lambda = Math.min(1, Math.max(0, num / (den || 1e-12)));
  const Sigma = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (_, j) => {
      if (i === j) return S[i][j];
      return (1 - lambda) * S[i][j];
    }),
  );
  return { Sigma, lambda };
}

/**
 * Bayes–Stein 均值收缩：将每个 mu_i 向整体均值收缩。
 * 收缩权重 w_i = var_i / (var_i + tau2)，后验 = (1-w_i)*mu_i + w_i*overall。
 */
export function bayesSteinShrinkage(
  mu: number[],
  varMu: number[],
  tau2: number,
): number[] {
  const overall = mean(mu);
  return mu.map((m, i) => {
    const v = varMu[i] || 1e-12;
    const w = v / (v + tau2);
    return (1 - w) * m + w * overall;
  });
}
