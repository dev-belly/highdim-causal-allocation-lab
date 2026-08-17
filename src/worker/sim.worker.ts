/// <reference lib="webworker" />
// 蒙特卡洛模拟 Web Worker：在后台线程并行（单线程但脱离 UI）跑试验，
// 逐单元回传进度，结束后回传全部聚合结果。

import { runTrial, aggregate } from '../core/simulator';
import type {
  DGPParams,
  EstimatorName,
  GridCellResult,
  WorkerMessage,
} from '../core/types';

interface RunRequest {
  grid: DGPParams[];
  estimators: EstimatorName[];
  nTrials: number;
  seed: number;
}

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent<RunRequest>) => {
  const { grid, estimators, nTrials, seed } = e.data;
  const allSummary: GridCellResult[] = [];
  for (let c = 0; c < grid.length; c++) {
    const params = grid[c];
    const results = [];
    for (let t = 0; t < nTrials; t++) {
      results.push(runTrial(params, estimators, seed + t * 7919 + c * 104729));
    }
    const summary = aggregate(results, params.ate);
    const cell: GridCellResult = { params, summary };
    allSummary.push(cell);
    const msg: WorkerMessage = {
      type: 'progress',
      done: c + 1,
      total: grid.length,
      cell,
    };
    ctx.postMessage(msg);
  }
  const done: WorkerMessage = { type: 'done', allSummary };
  ctx.postMessage(done);
};
