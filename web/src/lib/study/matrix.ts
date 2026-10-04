import { isEstimatorFamily, isShape } from "./types";
import type { EstimatorFamily, FamilyRow, Shape, StudyBundle } from "./types";

export type Level = "scenario" | "scenario_by_size";

export type Cell =
  | { kind: "value"; value: number; mcse: number | null; nUsed: number; nFitted: number }
  | { kind: "not_eligible" }
  | { kind: "refused"; nRefused: number }
  | { kind: "not_applicable"; nFitted: number } // fitted, but no fit feeds this metric
  | { kind: "undefined"; nUsed: number } // fitted and used, the metric has no value here
  | { kind: "absent" }; // no row for the pair, or a fitted row without a usable value or `_n_used` for the metric

export type Column = {
  key: string;
  scenario: string;
  shape: Shape;
  nUnits?: number;
  nPeriods?: number | null;
};

export type MatrixRow = { estimator: string; family: EstimatorFamily; cells: Cell[] };

export type Matrix = { metric: string; columns: Column[]; rows: MatrixRow[] };

const LEVELS: readonly Level[] = ["scenario", "scenario_by_size"];

function numberOrNull(value: number | string | null | undefined): number | null {
  return typeof value === "number" ? value : null;
}

/**
 * What one estimator on one scenario says about one metric. A row with no fit
 * is a reason (`refused` or `not_eligible`), never a number; a fitted row whose
 * metric is `null` says whether no fit fed it (`not_applicable`) or fits did
 * and it is undefined (`undefined`). A row that does not carry the metric at
 * all is `absent`.
 */
export function cellOf(row: FamilyRow | undefined, metric: string): Cell {
  if (row === undefined) return { kind: "absent" };
  if (row.n_fitted === 0) {
    return row.n_refused > 0 ? { kind: "refused", nRefused: row.n_refused } : { kind: "not_eligible" };
  }
  const value = row[metric];
  const nUsed = row[`${metric}_n_used`];
  if (typeof nUsed !== "number") return { kind: "absent" };
  if (value === null) {
    return nUsed === 0 ? { kind: "not_applicable", nFitted: row.n_fitted } : { kind: "undefined", nUsed };
  }
  if (typeof value !== "number") return { kind: "absent" };
  return {
    kind: "value",
    value,
    mcse: numberOrNull(row[`${metric}_mcse`]),
    nUsed,
    nFitted: row.n_fitted,
  };
}

function columnKey(scenario: string, nUnits: number | undefined, nPeriods: number | null | undefined) {
  return nUnits === undefined ? scenario : `${scenario}:${nUnits}:${nPeriods ?? "-"}`;
}

function keyOfRow(row: FamilyRow): string {
  return columnKey(row.scenario, row.n_units, row.n_periods);
}

/** A scenario's columns at one level: itself, or one per size, by units then periods. */
function columnsOf(rows: FamilyRow[], scenario: string, shape: Shape, level: Level): Column[] {
  if (level === "scenario") return [{ key: scenario, scenario, shape }];
  const seen = new Map<string, Column>();
  for (const row of rows) {
    if (row.scenario !== scenario || row.n_units === undefined) continue;
    const key = keyOfRow(row);
    if (!seen.has(key)) {
      seen.set(key, { key, scenario, shape, nUnits: row.n_units, nPeriods: row.n_periods ?? null });
    }
  }
  return [...seen.values()].sort(
    (a, b) => (a.nUnits ?? 0) - (b.nUnits ?? 0) || (a.nPeriods ?? -1) - (b.nPeriods ?? -1),
  );
}

export function matrixOf(
  study: StudyBundle,
  view: {
    family: string;
    metric: string;
    level: Level;
    shape: Shape | "all";
    estimatorFamily: EstimatorFamily | "all";
  },
): Matrix {
  const rows = study.families[view.family]?.[view.level] ?? [];
  const byPair = new Map<string, FamilyRow>();
  for (const row of rows) byPair.set(`${row.method}|${keyOfRow(row)}`, row);

  // `loadStudy` refuses a bundle with a shape or family this build does not
  // know, so these throws are unreachable after it. Reaching one means a study
  // got here without the load check, and a quiet drop would show a matrix with
  // a column or a row missing.
  const scenarios = study.scenarios.map((scenario) => {
    if (!isShape(scenario.shape)) {
      throw new Error(`scenario ${scenario.name} has a shape this build does not know: "${scenario.shape}"`);
    }
    return { name: scenario.name, shape: scenario.shape };
  });
  const known = study.estimators.map((estimator) => {
    if (!isEstimatorFamily(estimator.family)) {
      throw new Error(`estimator ${estimator.name} has a family this build does not know: "${estimator.family}"`);
    }
    return { name: estimator.name, family: estimator.family };
  });

  const columns = scenarios.flatMap((scenario) =>
    view.shape === "all" || scenario.shape === view.shape
      ? columnsOf(rows, scenario.name, scenario.shape, view.level)
      : [],
  );

  const estimators = known.filter(
    (estimator) => view.estimatorFamily === "all" || estimator.family === view.estimatorFamily,
  );

  return {
    metric: view.metric,
    columns,
    rows: estimators.map((estimator) => ({
      estimator: estimator.name,
      family: estimator.family,
      cells: columns.map((column) => cellOf(byPair.get(`${estimator.name}|${column.key}`), view.metric)),
    })),
  };
}

/**
 * The metric families the page can show: those in `family_metrics` that
 * `families` also holds, in `family_metrics` order. `false_positive_rate` is a
 * table, not a family, and a family a bundle lacks is not offered.
 */
export function offeredFamilies(
  study: StudyBundle,
): { family: string; metrics: string[]; levels: Level[] }[] {
  return Object.entries(study.family_metrics).flatMap(([family, metrics]) => {
    const held = study.families[family];
    if (held === undefined) return [];
    return [{ family, metrics, levels: LEVELS.filter((level) => held[level] !== undefined) }];
  });
}
