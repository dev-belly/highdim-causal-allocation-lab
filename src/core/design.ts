import type { DGPParams } from './types';

export type BaseParams = Omit<DGPParams, 'n' | 'p'>;

export const MIN_TRIALS = 10;
export const MAX_TRIALS = 1_000;

export function maximumStrata(nLevels: number[]): number {
  return Math.floor(Math.min(...nLevels) / 2);
}

export function designError(base: BaseParams, nLevels: number[], pLevels: number[], nTrials: number): string {
  if (!Number.isFinite(base.ate)) return '请输入有效的处理效应 ATE。';
  if (!Number.isFinite(base.noiseSd) || base.noiseSd < 0) return '噪声标准差 σ 必须是非负数。';
  if (base.heteroType !== 'homogeneous' && (!Number.isFinite(base.heteroStrength) || base.heteroStrength < 0)) {
    return '异质强度必须是非负数。';
  }
  if (base.assignment === 'stratified') {
    if (!Number.isInteger(base.nStrata) || base.nStrata < 2 || base.nStrata > maximumStrata(nLevels)) {
      return `分层数须为 2 至 ${maximumStrata(nLevels)} 的整数，让最小样本的每层至少有 2 个观测。`;
    }
  }
  if (base.corrType !== 'independent') {
    const maxP = Math.max(...pLevels);
    const clusterSize = base.corrType === 'block' ? (base.blockSize ?? Math.floor(maxP / 4)) : maxP;
    const lower = base.corrType === 'ar1' || clusterSize <= 1 ? -1 : -1 / (clusterSize - 1);
    if (!Number.isFinite(base.rho) || base.rho <= lower || base.rho >= 1) {
      return lower === -1
        ? '相关系数 ρ 必须大于 −1 且小于 1。'
        : `当前维度的 ρ 必须大于 −1/${clusterSize - 1} 且小于 1，协方差矩阵才是正定的。`;
    }
  }
  if (!Number.isInteger(nTrials) || nTrials < MIN_TRIALS || nTrials > MAX_TRIALS) {
    return `每组试验次数须为 ${MIN_TRIALS} 至 ${MAX_TRIALS} 的整数。`;
  }
  return '';
}

export function runParameters(base: BaseParams): BaseParams {
  return {
    ...base,
    rho: base.corrType === 'independent' ? 0 : base.rho,
    heteroStrength: base.heteroType === 'homogeneous' ? 0 : base.heteroStrength,
    nStrata: base.assignment === 'complete' ? 2 : base.nStrata,
  };
}
