// 全局类型定义 —— 高维因果推断与稳健资产配置实验平台

export type CorrType = 'independent' | 'exchangeable' | 'ar1' | 'block';
export type HeteroType = 'homogeneous' | 'linear' | 'nonlinear';
export type Assignment = 'complete' | 'stratified';
export type EstimatorName =
  | 'diff_in_means'
  | 'ols_adjusted'
  | 'lasso_adjusted'
  | 'cross_fitting_dml';

export interface DGPParams {
  n: number;
  p: number;
  ate: number;
  corrType: CorrType;
  rho: number;
  blockSize?: number;
  heteroType: HeteroType;
  heteroStrength: number;
  assignment: Assignment;
  nStrata: number;
  treatProb: number;
  noiseSd: number;
  sparse: boolean;
}

export interface Dataset {
  X: number[][];
  W: number[];
  Y: number[];
  strata: number[];
  Y0: number[];
  Y1: number[];
  tau: number[];
  ateTrue: number;
}

export interface EstimatorResult {
  tau: number;
  se: number;
}

export interface TrialMetric extends EstimatorResult {
  ateTrue: number;
  estimator: EstimatorName;
  wStar: number;
  wHat: number;
  wShrunk: number;
  weightBias: number;
  weightAbsErr: number;
  utilityLoss: number;
  oosSharpe: number;
  oosSharpeShrunk: number;
}

export interface EstimatorSummary {
  bias: number;
  sd: number;
  rmse: number;
  meanSe: number;
  coverage: number;
  meanWeightAbsErr: number;
  meanUtilityLoss: number;
  meanOosSharpe: number;
  meanOosSharpeShrunk: number;
}

export interface GridCellResult {
  params: DGPParams;
  summary: Record<EstimatorName, EstimatorSummary>;
}

export interface SimProgress {
  type: 'progress';
  done: number;
  total: number;
  cell: GridCellResult;
}

export interface SimDone {
  type: 'done';
  allSummary: GridCellResult[];
}

export type WorkerMessage = SimProgress | SimDone;
