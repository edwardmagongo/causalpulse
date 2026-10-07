import Link from "next/link";

import { formatNumber } from "@/lib/format";
import type { Cell, Matrix, MatrixRow } from "@/lib/study/matrix";
import type { View } from "@/lib/study/query";

import { NoValue, cellReason, cellWord, scenarioGroups, sizeLabel } from "./MatrixHeatmap";

/**
 * The rows in the order the view asks for: by the sorted column's value, or by
 * estimator name. A cell with no number sorts last in either direction. A sort
 * that names no column of this matrix leaves the study's order.
 */
export function sortedRows(matrix: Matrix, view: Pick<View, "sort" | "direction">): MatrixRow[] {
  if (view.sort === null) return matrix.rows;
  const sign = view.direction === "desc" ? -1 : 1;
  if (view.sort === "estimator") {
    return [...matrix.rows].sort(
      (a, b) => sign * (a.estimator < b.estimator ? -1 : a.estimator > b.estimator ? 1 : 0),
    );
  }
  const index = matrix.columns.findIndex((column) => column.key === view.sort);
  if (index < 0) return matrix.rows;
  const valueOf = (row: MatrixRow): number | null => {
    const cell = row.cells[index];
    return cell.kind === "value" ? cell.value : null;
  };
  return [...matrix.rows].sort((a, b) => {
    const x = valueOf(a);
    const y = valueOf(b);
    if (x === null || y === null) return x === y ? 0 : x === null ? 1 : -1;
    return sign * (x - y);
  });
}

/**
 * The matrix as a plain table: one row per estimator, the number and its Monte
 * Carlo error as text, with the fits behind it, and no colour. Each column
 * head is a link that sorts by that column, and again to flip the direction.
 */
export function MatrixTable({
  matrix,
  view,
  href,
}: {
  matrix: Matrix;
  view: View;
  href: (view: View) => string;
}) {
  const sized = view.level === "scenario_by_size";
  const sortHref = (key: string) =>
    href({ ...view, sort: key, direction: view.sort === key && view.direction === "asc" ? "desc" : "asc" });
  const ariaSort = (key: string) =>
    view.sort === key ? (view.direction === "asc" ? ("ascending" as const) : ("descending" as const)) : undefined;

  return (
    <div className="table-scroll">
      <table className="data-table matrix">
        <caption className="sr-only">{matrix.metric} by estimator and scenario</caption>
        <thead>
          {sized ? (
            <tr>
              {/* Not a header: it fills the corner above the estimators, whose own head is in the next row. */}
              <td>Scenario</td>
              {scenarioGroups(matrix.columns).map((group) => (
                <th key={group.scenario} scope="colgroup" colSpan={group.span} className="num value">
                  {group.scenario}
                </th>
              ))}
            </tr>
          ) : null}
          <tr>
            <th scope="col" aria-sort={ariaSort("estimator")}>
              <Link href={sortHref("estimator")} scroll={false} className="sort-link">
                Estimator
              </Link>
            </th>
            {matrix.columns.map((column) => (
              <th key={column.key} scope="col" className="num" aria-sort={ariaSort(column.key)}>
                <Link href={sortHref(column.key)} scroll={false} className="sort-link value">
                  {sized ? sizeLabel(column) : column.scenario}
                </Link>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedRows(matrix, view).map((row) => (
            <tr key={row.estimator}>
              <th scope="row">{row.estimator}</th>
              {row.cells.map((cell, index) => (
                <TableCell key={matrix.columns[index].key} cell={cell} />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TableCell({ cell }: { cell: Cell }) {
  if (cell.kind !== "value") {
    return (
      <td className="num text-ink-soft" data-kind={cell.kind} title={cellReason(cell)}>
        <NoValue cell={cell} mark={cellWord(cell, false)} />
      </td>
    );
  }
  return (
    <td className="num" data-kind="value" title={cellReason(cell)}>
      {formatNumber(cell.value)}
      <span className="label block">± {formatNumber(cell.mcse)}</span>
      <span className="label block">{cellReason(cell)}</span>
    </td>
  );
}
