/**
 * The shape of `src/data/benchmark-study.json`, written by hand. The bundle is
 * assigned to `StudyBundle` in `bundle.ts`, so a field the generator drops or
 * retypes fails `tsc` there instead of surfacing on the page. A key it adds
 * does not: a JSON import gets no excess-property check, and the index
 * signatures accept any column. Added keys are pinned on the Python side by
 * `tests/report/test_bundle.py`.
 */
import type { Family } from "@/lib/api/types";

/** An estimator's family: phase 1's catalogue family, not a second copy. */
export type EstimatorFamily = Family;

/** A scenario's data shape. */
export type Shape = "cross_section" | "panel" | "donor_pool";

/**
 * `family` and `shape` are `string | null` here, not their unions: TypeScript
 * reads a JSON import's literals as `string`, so a union would make the drift
 * guard reject the committed file for a reason that is not drift, and the
 * writer fills each from a lookup that can miss, which it writes as `null`.
 * `isEstimatorFamily` and `isShape` narrow them once, at the load boundary
 * (`checkVocabulary` in `bundle.ts`), which refuses a bundle holding a value
 * they do not recognise, a `null` included.
 */
export type Estimator = { name: string; family: string | null };

export type Scenario = { name: string; shape: string | null };

const SHAPES: readonly string[] = ["cross_section", "panel", "donor_pool"];
const ESTIMATOR_FAMILIES: readonly string[] = ["cross_sectional", "panel", "donor_pool"];

export function isShape(value: string | null): value is Shape {
  return value !== null && SHAPES.includes(value);
}

export function isEstimatorFamily(value: string | null): value is EstimatorFamily {
  return value !== null && ESTIMATOR_FAMILIES.includes(value);
}

export type Provenance = {
  cells_done: number;
  cells_expected: number;
  /** `null` when the run's manifest records no commit. */
  git_commit: string | null;
  replications: number;
  role: string;
  tier: string;
};

export type Overview = {
  coverage_mcse: number;
  n_cells: number;
  n_estimators: number;
  n_scenarios: number;
  replications: number;
};

export type Expectation = {
  id: string;
  narrative: string;
  statement: string;
  verdict: string;
};

export type Expectations = {
  n_failed: number;
  n_inconclusive: number;
  n_met: number;
  rows: Expectation[];
};

/**
 * One estimator on one scenario, or on one scenario at one size. The counters
 * are fixed; every other column is a metric (`bias`), its Monte Carlo standard
 * error (`bias_mcse`) or the number of fits that fed it (`bias_n_used`).
 */
export type FamilyRow = {
  method: string;
  scenario: string;
  n_units?: number;
  n_periods?: number | null;
  n_fitted: number;
  n_ineligible: number;
  n_refused: number;
  [metric: string]: number | string | null | undefined;
};

/** Metric family, then level (`scenario`, `scenario_by_size`), then rows. */
export type Families = Record<string, Record<string, FamilyRow[]>>;

/** A row of a published report table: a method, maybe an interval, then one number per scenario. */
export type ReportTableRow = {
  method: string;
  interval?: string;
  [scenario: string]: number | string | null | undefined;
};

export type ReportTables = {
  coverage_note: string;
  exploratory_note: string;
  tables: Record<string, ReportTableRow[]>;
};

/** One method on one conditional-effect scenario. A metric no fit fed, and its error, are `null`. */
export type CateRow = {
  method: string;
  scenario: string;
  cate_calibration_slope: number | null;
  cate_calibration_slope_mcse: number | null;
  cate_calibration_slope_n_used: number;
  cate_correlation: number | null;
  cate_correlation_mcse: number | null;
  cate_correlation_n_used: number;
  cate_rmse: number | null;
  cate_rmse_mcse: number | null;
  cate_rmse_n_used: number;
};

export type Cate = {
  cells_done: number;
  cells_expected: number;
  note: string;
  replications: number;
  table: CateRow[];
};

/**
 * One method and interval on scenario P7. A metric no fit fed, and its error,
 * are `null`; `interval` is an empty string when the method has no interval.
 */
export type P7Row = {
  bias: number | null;
  bias_mcse: number | null;
  bias_n_used: number;
  coverage: number | null;
  coverage_mcse: number | null;
  coverage_n_used: number;
  interval: string;
  method: string;
  rmse: number | null;
  rmse_mcse: number | null;
  rmse_n_used: number;
};

export type P7 = {
  cells_done: number;
  cells_expected: number;
  note: string;
  replications: number;
  table: P7Row[];
};

/** What the study did not cover: a name, as the study writes it, and the reason. */
export type NotCovered = { name: string; reason: string };

export type Limitations = {
  not_covered: NotCovered[];
  statements: string[];
};

export type StudyBundle = {
  schema_version: number;
  report_sha256: string;
  provenance: Provenance[];
  overview: Overview;
  expectations: Expectations;
  estimators: Estimator[];
  scenarios: Scenario[];
  /** Level name to the columns that identify a cell at that level. */
  levels: Record<string, string[]>;
  /** Metric family to the metrics it holds, in the study's order. */
  family_metrics: Record<string, string[]>;
  /** Metric name to the study's own definition. */
  metric_definitions: Record<string, string>;
  families: Families;
  report_tables: ReportTables;
  /** `null` when the report has no such section: the writer emits `null`, not a missing key. */
  cate: Cate | null;
  p7: P7 | null;
  limitations: Limitations;
};

/** What `checkVersion` says: the schema versions this build can read, or the one it found. */
export type VersionCheck =
  | { ok: true }
  | { ok: false; reason: "version"; found: number; known: readonly number[] };

/** What `checkVocabulary` says: `unknown` names each shape or family this build does not know. */
export type VocabularyCheck = { ok: true } | { ok: false; reason: "vocabulary"; unknown: string[] };

/** The result of loading the study: the bundle, or the check that refused it. */
export type StudyLoad =
  | { ok: true; study: StudyBundle }
  | Exclude<VersionCheck, { ok: true }>
  | Exclude<VocabularyCheck, { ok: true }>;
