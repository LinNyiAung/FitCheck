import type { FitResponse } from "../api";

const fmt = (x: number | null | undefined) => (x == null ? "–" : x.toFixed(1));

export function ResultsPanel({ result, error }: { result: FitResponse | null; error: string | null }) {
  return (
    <aside className="panel right">
      <h2 className="first">Result</h2>
      {error && <div className="warn">{error}</div>}
      {result?.warnings.map((w) => (
        <div className="warn" key={w}>{w}</div>
      ))}

      {!result ? (
        <p className="empty">No body yet.</p>
      ) : (
        <>
          <table>
            <thead>
              <tr><th /><th>You</th><th>Model</th><th>Δ</th></tr>
            </thead>
            <tbody>
              {result.measurements.map((m) => {
                const tolerance = m.key === "weight" ? 4 : 1;
                const cls = m.error == null ? "" : Math.abs(m.error) <= tolerance ? "ok" : "off";
                return (
                  <tr key={m.key}>
                    <td>{m.label}</td>
                    <td>{fmt(m.target)}</td>
                    <td>{fmt(m.achieved)}</td>
                    <td className={cls}>
                      {m.error == null ? "" : `${m.error > 0 ? "+" : ""}${m.error.toFixed(1)}`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="meta">
            Fitted in {result.seconds}s ({result.iterations} steps). Values you left empty show
            what the model estimated for you.
          </p>
          <details>
            <summary>Model parameters</summary>
            <pre>{JSON.stringify(result.shape, null, 2)}</pre>
          </details>
        </>
      )}
    </aside>
  );
}
