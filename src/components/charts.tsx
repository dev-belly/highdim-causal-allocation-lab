import { useMemo } from 'react';
import { Plot, PALETTE } from './Plot';
import type { GridCellResult, EstimatorName } from '../core/types';

interface ChartProps {
  results: GridCellResult[];
  estimators?: EstimatorName[];
}

function uniqueSorted(values: number[]): number[] {
  return [...new Set(values)].sort((a, b) => a - b);
}

/** 覆盖率折线图（固定某 p，随 n 变化，多估计量） */
export function CoverageChart({ results, estimators = [], selectedP }: ChartProps & { selectedP: number }) {
  const data = useMemo(() => {
    const ns = uniqueSorted(results.filter((c) => c.params.p === selectedP).map((c) => c.params.n));
    return estimators.map((est, i) => {
      const cells = results
        .filter((c) => c.params.p === selectedP && ns.includes(c.params.n))
        .sort((a, b) => a.params.n - b.params.n);
      return {
        x: cells.map((c) => c.params.n),
        y: cells.map((c) => c.summary[est]?.coverage ?? null),
        name: est,
        mode: 'lines+markers',
        line: { color: PALETTE[i % PALETTE.length], width: 2.5 },
        marker: { size: 6 },
      };
    });
  }, [results, estimators, selectedP]);

  const layout = {
    title: { text: `95% CI 覆盖率（p=${selectedP}）`, font: { size: 14 } },
    xaxis: { title: '样本量 n', type: 'log' },
    yaxis: { title: '覆盖率', range: [0.8, 1] },
    shapes: [
      {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        y0: 0.95,
        y1: 0.95,
        line: { color: '#f87171', width: 1.5, dash: 'dash' },
      },
    ],
  };
  return <Plot data={data} layout={layout} className="chart-box" />;
}

/** RMSE 热力图（固定某估计量，n × p） */
export function RmseHeatmap({ results, selectedEstimator }: ChartProps & { selectedEstimator: EstimatorName }) {
  const { z, x, y } = useMemo(() => {
    const ns = uniqueSorted(results.map((c) => c.params.n));
    const ps = uniqueSorted(results.map((c) => c.params.p));
    const zz = ns.map((n) =>
      ps.map((p) => {
        const cell = results.find((c) => c.params.n === n && c.params.p === p);
        return cell ? cell.summary[selectedEstimator]?.rmse ?? null : null;
      }),
    );
    return { z: zz, x: ps, y: ns };
  }, [results, selectedEstimator]);

  const data = [
    {
      type: 'heatmap',
      z,
      x,
      y,
      colorscale: 'YlOrRd',
      colorbar: { title: 'RMSE', thickness: 12 },
      hovertemplate: 'n=%{y}, p=%{x}<br>RMSE=%{z:.3f}<extra></extra>',
    },
  ];
  const layout = {
    title: { text: `RMSE 热力图 — ${selectedEstimator}`, font: { size: 14 } },
    xaxis: { title: '协变量维度 p' },
    yaxis: { title: '样本量 n' },
  };
  return <Plot data={data} layout={layout} className="chart-box" />;
}

/** 资产配置权重绝对误差折线（固定某 p，随 n） */
export function WeightErrorChart({ results, estimators = [], selectedP }: ChartProps & { selectedP: number }) {
  const data = useMemo(() => {
    const ns = uniqueSorted(results.filter((c) => c.params.p === selectedP).map((c) => c.params.n));
    return estimators.map((est, i) => {
      const cells = results
        .filter((c) => c.params.p === selectedP && ns.includes(c.params.n))
        .sort((a, b) => a.params.n - b.params.n);
      return {
        x: cells.map((c) => c.params.n),
        y: cells.map((c) => c.summary[est]?.meanWeightAbsErr ?? null),
        name: est,
        mode: 'lines+markers',
        line: { color: PALETTE[i % PALETTE.length], width: 2.5 },
        marker: { size: 6 },
      };
    });
  }, [results, estimators, selectedP]);

  const layout = {
    title: { text: `组合权重绝对误差（p=${selectedP}）`, font: { size: 14 } },
    xaxis: { title: '样本量 n', type: 'log' },
    yaxis: { title: 'E|ŵ − w*|' },
  };
  return <Plot data={data} layout={layout} className="chart-box" />;
}

/** RMSE 3D 曲面（估计量 × n × p） */
export function RmseSurface({ results, selectedEstimator }: ChartProps & { selectedEstimator: EstimatorName }) {
  const { z, x, y } = useMemo(() => {
    const ns = uniqueSorted(results.map((c) => c.params.n));
    const ps = uniqueSorted(results.map((c) => c.params.p));
    const zz = ns.map((n) =>
      ps.map((p) => {
        const cell = results.find((c) => c.params.n === n && c.params.p === p);
        return cell ? cell.summary[selectedEstimator]?.rmse ?? null : null;
      }),
    );
    return { z: zz, x: ps, y: ns };
  }, [results, selectedEstimator]);

  const data = [
    {
      type: 'surface',
      z,
      x,
      y,
      colorscale: 'Viridis',
      colorbar: { title: 'RMSE', thickness: 12 },
    },
  ];
  const layout = {
    title: { text: `RMSE 曲面 — ${selectedEstimator}`, font: { size: 14 } },
    scene: {
      xaxis: { title: 'p' },
      yaxis: { title: 'n' },
      zaxis: { title: 'RMSE' },
      bgcolor: 'rgba(0,0,0,0)',
    },
    margin: { t: 46, r: 10, b: 10, l: 10 },
  };
  return <Plot data={data} layout={layout} className="chart-box" />;
}
