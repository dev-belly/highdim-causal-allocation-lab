import type { GridCellResult, EstimatorName } from '../core/types';

interface Props {
  results: GridCellResult[];
  estimators: EstimatorName[];
}

function covBadge(cov: number) {
  const d = Math.abs(cov - 0.95);
  if (d < 0.03) return <span className="badge badge-ok">{(cov * 100).toFixed(1)}%</span>;
  if (d < 0.08) return <span className="badge badge-warn">{(cov * 100).toFixed(1)}%</span>;
  return <span className="badge badge-bad">{(cov * 100).toFixed(1)}%</span>;
}

export function ResultTable({ results, estimators }: Props) {
  if (results.length === 0) return null;

  const summary = estimators.map((est) => {
    const cells = results.filter((c) => c.summary[est]);
    const avg = (key: keyof GridCellResult['summary'][EstimatorName]) =>
      cells.reduce((s, c) => s + (c.summary[est][key] as number), 0) / Math.max(1, cells.length);
    const cov = avg('coverage');
    const oos = avg('meanOosSharpe');
    const oosS = avg('meanOosSharpeShrunk');
    const lift = oos !== 0 ? ((oosS - oos) / Math.abs(oos)) * 100 : 0;
    return {
      est,
      bias: avg('bias'),
      rmse: avg('rmse'),
      cov,
      wErr: avg('meanWeightAbsErr'),
      oos,
      oosS,
      lift,
    };
  });

  const rows = results
    .slice()
    .sort((a, b) => a.params.n - b.params.n || a.params.p - b.params.p);

  return (
    <>
      <div className="card">
        <h2>估计量对比摘要（跨参数单元平均）</h2>
        <div className="table-scroll">
          <table className="matrix">
            <thead>
              <tr>
                <th>估计量</th>
                <th>Bias</th>
                <th>RMSE</th>
                <th>覆盖率</th>
                <th>E|ŵ−w*|</th>
                <th>OOS Sharpe</th>
                <th>收缩 Sharpe</th>
                <th>收缩提升</th>
              </tr>
            </thead>
            <tbody>
              {summary.map((s) => (
                <tr key={s.est}>
                  <td>{s.est}</td>
                  <td>{s.bias.toFixed(3)}</td>
                  <td>{s.rmse.toFixed(3)}</td>
                  <td>{covBadge(s.cov)}</td>
                  <td>{s.wErr.toFixed(3)}</td>
                  <td>{s.oos.toFixed(3)}</td>
                  <td>{s.oosS.toFixed(3)}</td>
                  <td style={{ color: s.lift >= 0 ? 'var(--good)' : 'var(--bad)' }}>
                    {s.lift >= 0 ? '+' : ''}
                    {s.lift.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="note">
          覆盖率基于 95% 名义水平；收缩 Sharpe 为 Bayes–Stein 风格向理论最优权重收缩后的样本外夏普，
          体现“收缩基准 vs 无约束基准”的稳健性改善。
        </div>
      </div>

      <div className="card">
        <h2>参数矩阵（RMSE）</h2>
        <div className="table-scroll">
          <table className="matrix">
            <thead>
              <tr>
                <th>n</th>
                <th>p</th>
                {estimators.map((e) => (
                  <th key={e}>{e}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((c, idx) => (
                <tr key={idx}>
                  <td>{c.params.n}</td>
                  <td>{c.params.p}</td>
                  {estimators.map((e) => (
                    <td key={e}>{(c.summary[e]?.rmse ?? NaN).toFixed(3)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
