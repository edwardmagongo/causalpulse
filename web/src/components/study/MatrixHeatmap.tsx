import type { CSSProperties } from "react";

import { formatNumber } from "@/lib/format";
import type { Cell, Column, Level, Matrix } from "@/lib/study/matrix";
import { position } from "@/lib/study/scale";
import type { Scale } from "@/lib/study/scale";

/** A column's size at the second level: `500`, or `150 × 7` for a panel. Absent at the first level. */
export function sizeLabel(column: Column): string | null {
  if (column.nUnits === undefined) return null;
  return column.nPeriods === null || column.nPeriods === undefined
    ? String(column.nUnits)
    : `${column.nUnits} × ${column.nPeriods}`;
}

/** Why a cell reads as it does: the fits behind a number, or the reason there is none. */
export function cellReason(cell: Cell): string {
  switch (cell.kind) {
    case "value":
      return cell.nRefused > 0
        ? `${cell.nUsed} of ${cell.nFitted} fits, ${cell.nRefused} refused`
        : `${cell.nUsed} of ${cell.nFitted} fits`;
    case "not_eligible":
      return "Not eligible for this scenario";
    case "refused":
      return `Every fit was refused (${cell.nRefused})`;
    case "not_applicable":
      return `No fit feeds this metric (0 of ${cell.nFitted})`;
    case "undefined":
      return "The metric has no value here";
    case "absent":
      return "No row in the study";
  }
}

/** The word a cell with no number shows; an ineligible pair is left blank in the heatmap. */
export function cellWord(cell: Exclude<Cell, { kind: "value" }>, blankIneligible: boolean): string {
  switch (cell.kind) {
    case "not_eligible":
      return blankIneligible ? "" : "not eligible";
    case "refused":
      return "refused";
    case "undefined":
      return "undefined";
    case "not_applicable":
    case "absent":
      return "—";
  }
}

/**
 * What a cell with no number holds. The visible mark, if any, is hidden from a
 * screen reader; the reason (the cell's `title`, which touch and keyboard users
 * never see) is read instead, and hidden from the eye. An ineligible pair has
 * no mark.
 */
export function NoValue({ cell, mark }: { cell: Exclude<Cell, { kind: "value" }>; mark: string }) {
  return (
    <>
      {mark === "" ? null : <span aria-hidden="true">{mark}</span>}
      <span className="sr-only">{cellReason(cell)}</span>
    </>
  );
}

/** The scenarios of a matrix in order, each with the number of columns it spans. */
export function scenarioGroups(columns: readonly Column[]): { scenario: string; span: number }[] {
  const groups: { scenario: string; span: number }[] = [];
  for (const column of columns) {
    const last = groups[groups.length - 1];
    if (last !== undefined && last.scenario === column.scenario) last.span += 1;
    else groups.push({ scenario: column.scenario, span: 1 });
  }
  return groups;
}

/**
 * The estimator-by-scenario matrix as a heatmap: a real table, scenarios across
 * (and, at the second level, sizes under them), estimators down. A number is
 * coloured by where it sits on the scale (`--pos`, mixed in `globals.css`); a
 * cell with no number is a word or a hatch, never a colour of the scale.
 */
export function MatrixHeatmap({ matrix, scale, level }: { matrix: Matrix; scale: Scale; level: Level }) {
  const sized = level === "scenario_by_size";
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
                <th key={group.scenario} scope="colgroup" colSpan={group.span} className="value">
                  {group.scenario}
                </th>
              ))}
            </tr>
          ) : null}
          <tr>
            <th scope="col">Estimator</th>
            {matrix.columns.map((column) => (
              <th key={column.key} scope="col" className={sized ? undefined : "value"}>
                {sized ? <span className="value">{sizeLabel(column)}</span> : column.scenario}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.rows.map((row) => (
            <tr key={row.estimator}>
              <th scope="row">{row.estimator}</th>
              {row.cells.map((cell, index) => (
                <HeatCell key={matrix.columns[index].key} cell={cell} scale={scale} />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HeatCell({ cell, scale }: { cell: Cell; scale: Scale }) {
  const title = cellReason(cell);
  if (cell.kind !== "value") {
    return (
      <td className="matrix-cell value" data-kind={cell.kind} title={title}>
        <NoValue cell={cell} mark={cellWord(cell, true)} />
      </td>
    );
  }
  return (
    <td
      className="matrix-cell heat value"
      data-kind="value"
      data-scale={scale.kind}
      style={{ "--pos": position(cell.value, scale).toFixed(4) } as CSSProperties}
      title={title}
    >
      <span className="block">{formatNumber(cell.value)}</span>
      <span className="label block">± {formatNumber(cell.mcse)}</span>
      {cell.nUsed < cell.nFitted || cell.nRefused > 0 ? (
        <span className="label block">{title}</span>
      ) : null}
    </td>
  );
}
