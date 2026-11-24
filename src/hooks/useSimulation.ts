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
  error: string | null;
  start: (grid: DGPParams[], estimators: EstimatorName[], nTrials: number, seed?: number) => void;
  stop: () => void;
}

/** 管理模拟 Web Worker 的生命周期与进度状态 */
export function useSimulation(): SimState {
  const workerRef = useRef<Worker | null>(null);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [results, setResults] = useState<GridCellResult[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
    }
    setRunning(false);
  }, []);

  const start = useCallback(
    (grid: DGPParams[], estimators: EstimatorName[], nTrials: number, seed = 42) => {
      stop();
      setError(null);
      setResults([]);
      setProgress({ done: 0, total: grid.length });

      let worker: Worker;
      try {
        worker = new Worker(new URL('../worker/sim.worker.ts', import.meta.url), { type: 'module' });
      } catch {
        setError('无法启动模拟线程，请刷新后重试。');
        return;
      }
      workerRef.current = worker;
      setRunning(true);

      worker.onmessage = (e: MessageEvent<WorkerMessage>) => {
        if (workerRef.current !== worker) return;
        const msg = e.data;
        if (msg.type === 'progress') {
          setProgress({ done: msg.done, total: msg.total });
          setResults((prev) => [...prev, msg.cell]);
        } else if (msg.type === 'done') {
          setResults(msg.allSummary);
          stop();
        }
      };
      function fail(message: string) {
        if (workerRef.current !== worker) return;
        setError(message);
        stop();
      }
      worker.onerror = (event) => fail(event.message || '模拟运行失败，请检查参数后重试。');
      worker.onmessageerror = () => fail('模拟结果无法解析，请刷新后重试。');
      try {
        worker.postMessage({ grid, estimators, nTrials, seed });
      } catch {
        fail('无法发送模拟参数，请刷新后重试。');
      }
    },
    [stop],
  );

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  return { progress, results, running, error, start, stop };
}
