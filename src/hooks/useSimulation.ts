import { useRef, useState, useCallback, useEffect } from 'react';
import type {
  DGPParams,
  EstimatorName,
  GridCellResult,
  WorkerMessage,
} from '../core/types';

interface SimState {
  progress: { done: number; total: number };
  results: GridCellResult[];
  running: boolean;
  start: (grid: DGPParams[], estimators: EstimatorName[], nTrials: number, seed?: number) => void;
  stop: () => void;
}

/** 管理模拟 Web Worker 的生命周期与进度状态 */
export function useSimulation(): SimState {
  const workerRef = useRef<Worker | null>(null);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [results, setResults] = useState<GridCellResult[]>([]);
  const [running, setRunning] = useState(false);

  const stop = useCallback(() => {
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
    }
    setRunning(false);
  }, []);

  const start = useCallback(
    (grid: DGPParams[], estimators: EstimatorName[], nTrials: number, seed = 42) => {
      if (workerRef.current) workerRef.current.terminate();
      const worker = new Worker(new URL('../worker/sim.worker.ts', import.meta.url), {
        type: 'module',
      });
      workerRef.current = worker;
      setRunning(true);
      setResults([]);
      setProgress({ done: 0, total: grid.length });

      worker.onmessage = (e: MessageEvent<WorkerMessage>) => {
        const msg = e.data;
        if (msg.type === 'progress') {
          setProgress({ done: msg.done, total: msg.total });
          setResults((prev) => [...prev, msg.cell]);
        } else if (msg.type === 'done') {
          setResults(msg.allSummary);
          setRunning(false);
          worker.terminate();
          workerRef.current = null;
        }
      };
      worker.postMessage({ grid, estimators, nTrials, seed });
    },
    [],
  );

  useEffect(() => {
    return () => {
      if (workerRef.current) workerRef.current.terminate();
    };
  }, []);

  return { progress, results, running, start, stop };
}
