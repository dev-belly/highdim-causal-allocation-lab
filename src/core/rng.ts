// 可复现伪随机数、高斯采样、Cholesky 分解、多元正态、Fisher-Yates 洗牌

export type RNG = () => number;

/** mulberry32：快速、确定性的种子 PRNG，保证蒙特卡洛可复现 */
export function mulberry32(seed: number): RNG {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 标准正态（Box-Muller 变换） */
export function gaussian(rng: RNG): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Cholesky 分解：返回下三角 L，使 L·Lᵀ = A */
export function cholesky(A: number[][]): number[][] {
  const n = A.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        L[i][j] = Math.sqrt(Math.max(s, 1e-12));
      } else {
        L[i][j] = s / (L[j][j] || 1e-12);
      }
    }
  }
  return L;
}

/** 多元正态采样：mean + L·z，z~N(0,I) */
export function multivariateNormal(rng: RNG, mean: number[], cov: number[][]): number[] {
  const n = mean.length;
  const L = cholesky(cov);
  const z: number[] = new Array(n);
  for (let i = 0; i < n; i++) z[i] = gaussian(rng);
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = mean[i];
    for (let k = 0; k <= i; k++) s += L[i][k] * z[k];
    out[i] = s;
  }
  return out;
}

/** 原地 Fisher-Yates 洗牌 */
export function shuffle<T>(arr: T[], rng: RNG): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}
