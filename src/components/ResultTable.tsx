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
    const oos = avg('meanOosUtility');
    const oosS = avg('meanOosUtilityShrunk');
    const lift = oosS - oos;
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
                <th>样本外效用</th>
                <th>收缩后效用</th>
                <th>效用差</th>
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
                    {s.lift.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="note">
          覆盖率基于 95% 名义水平；样本外效用 = 平均组合收益 − 0.5 × 风险厌恶系数 × 收益方差。
          对照组潜在结果不作为无风险收益；收缩强度由估计标准误决定，不使用真实 ATE。效用差可以为负。
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
