// 线性代数与统计工具：标准化、线性方程组求解、矩阵求逆、软阈值、
// 分位数、正态分位数函数（Acklam 近似）、置信区间

export function mean(a: number[]): number {
  if (a.length === 0) return 0;
  let s = 0;
  for (const v of a) s += v;
  return s / a.length;
}

export function variance(a: number[]): number {
  const n = a.length;
  if (n < 2) return 0;
  const m = mean(a);
  let s = 0;
  for (const v of a) s += (v - m) * (v - m);
  return s / (n - 1);
}

export function std(a: number[]): number {
  return Math.sqrt(variance(a));
}

export function quantile(sorted: number[], q: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  const pos = (n - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function softThreshold(x: number, lambda: number): number {
  if (x > lambda) return x - lambda;
  if (x < -lambda) return x + lambda;
  return 0;
}

/** 列标准化（零均值、单位方差），返回标准化矩阵与统计量 */
export function columnStandardize(X: number[][]): {
  Xs: number[][];
  means: number[];
  stds: number[];
} {
  const n = X.length;
  const p = X[0].length;
  const means = new Array(p).fill(0);
  const stds = new Array(p).fill(1);
  for (let j = 0; j < p; j++) {
    let m = 0;
    for (let i = 0; i < n; i++) m += X[i][j];
    m /= n;
    means[j] = m;
    let v = 0;
    for (let i = 0; i < n; i++) {
      const d = X[i][j] - m;
      v += d * d;
    }
    v /= n;
    stds[j] = Math.sqrt(v) || 1;
  }
  const Xs = X.map((row) => row.map((val, j) => (val - means[j]) / stds[j]));
  return { Xs, means, stds };
}

/** 高斯消元求解 A·x = b */
export function solve(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[piv][i])) piv = k;
    [M[i], M[piv]] = [M[piv], M[i]];
    const d = M[i][i] || 1e-12;
    for (let j = i; j <= n; j++) M[i][j] /= d;
    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const f = M[k][i];
        for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j];
      }
    }
  }
  return M.map((r) => r[n]);
}

/** 矩阵求逆（高斯-约当消元） */
export function inverse(A: number[][]): number[][] {
  const n = A.length;
  const M = A.map((row, i) => [
    ...row,
    ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  ]);
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[piv][i])) piv = k;
    [M[i], M[piv]] = [M[piv], M[i]];
    const d = M[i][i] || 1e-12;
    for (let j = 0; j < 2 * n; j++) M[i][j] /= d;
    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const f = M[k][i];
        for (let j = 0; j < 2 * n; j++) M[k][j] -= f * M[i][j];
      }
    }
  }
  return M.map((r) => r.slice(n));
}

/** Acklam 近似：标准正态分位数函数（逆 CDF） */
export function normalPPF(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416,
  ];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let q: number;
  let r: number;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (p <= phigh) {
    q = p - 0.5;
    r = q * q;
    return (
      (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
      q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return (
    -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  );
}

/** 基于正态近似的置信区间 */
export function confidenceInterval(
  tau: number,
  se: number,
  level = 0.95,
): [number, number] {
  const z = normalPPF(1 - (1 - level) / 2);
  return [tau - z * se, tau + z * se];
}
