import { DASH, formatNumber } from "@/lib/format";

type Cell = string | number | null | undefined;

/**
 * A list of bundle records as a table: the `keys` columns first, then one
 * column per metric. A record's own key order is alphabetical, so the caller
 * declares the order. Each metric cell holds the number, its Monte Carlo
 * standard error beside it, and, only where fewer fits fed it than
 * `replications`, the count of fits that did. A `null` is a dash, never a zero.
 */
export function StudyTable({
  caption,
  rows,
  keys,
  metrics,
  replications,
}: {
  caption: string;
  rows: readonly Record<string, Cell>[];
  keys: readonly string[];
  metrics: readonly string[];
  replications: number;
}) {
  return (
    <div className="table-scroll">
      <table className="data-table">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>
            {keys.map((key) => (
              <th key={key} scope="col">
                {key}
              </th>
            ))}
            {metrics.map((metric) => (
              <th key={metric} scope="col" className="num">
                {metric}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={keys.map((key) => String(row[key])).join("/") + index}>
              {keys.map((key, column) =>
                column === 0 ? (
                  <th key={key} scope="row">
                    {text(row[key])}
                  </th>
                ) : (
                  <td key={key}>{text(row[key])}</td>
                ),
              )}
              {metrics.map((metric) => (
                <MetricCell key={metric} row={row} metric={metric} replications={replications} />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function text(cell: Cell): string {
  return cell === null || cell === undefined ? DASH : String(cell);
}

function number(cell: Cell): number | null {
  return typeof cell === "number" ? cell : null;
}

function MetricCell({
  row,
  metric,
  replications,
}: {
  row: Record<string, Cell>;
  metric: string;
  replications: number;
}) {
  const value = number(row[metric]);
  const mcse = number(row[`${metric}_mcse`]);
  const used = number(row[`${metric}_n_used`]);
  return (
    <td className="num">
      {formatNumber(value)}
      {value !== null ? <span className="label block">± {formatNumber(mcse)}</span> : null}
      {used !== null && used < replications ? (
        <span className="label block">
          {used} of {replications} fits
        </span>
      ) : null}
    </td>
  );
}
