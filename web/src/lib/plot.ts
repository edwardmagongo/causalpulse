import type { ComparisonGroup, Estimate } from "@/lib/api/types";

/** One method's point and primary interval. */
export type PlotRow = {
  method: string;
  point: number;
  lower: number;
  upper: number;
  basis: string;
  level: number;
};

/** One `(estimand, scale)` group: the only estimates that may share an axis. */
export type PlotPanel = {
  estimand: string;
  scale: string;
  rows: PlotRow[];
  domain: [number, number];
};

const PADDING = 0.05;

/**
 * One panel per comparison group, in the API's order.
 *
 * The grouping is the API's, not this module's: `compare()` groups estimates
 * by `(estimand, scale)`, and reusing its groups is what keeps a risk
 * difference and an ATT off the same axis. A method that produced no
 * estimate has nothing to draw and is left out; a group left with no rows
 * yields no panel.
 */
export function panels(estimates: Estimate[], comparisons: ComparisonGroup[]): PlotPanel[] {
  const summaries = new Map(
    estimates.flatMap((estimate) =>
      estimate.summary ? [[estimate.method, estimate.summary] as const] : [],
    ),
  );
  return comparisons.flatMap((group) => {
    const rows = group.methods.flatMap((method) => {
      const summary = summaries.get(method);
      if (!summary) return [];
      return [
        {
          method,
          point: summary.point,
          lower: summary.ci_lower,
          upper: summary.ci_upper,
          basis: summary.ci_basis,
          level: summary.ci_level,
        },
      ];
    });
    if (rows.length === 0) return [];
    return [{ estimand: group.estimand, scale: group.scale, rows, domain: domainOf(rows) }];
  });
}

/**
 * The axis range: every point and interval end, padded on both sides.
 *
 * A panel whose values all coincide has no span to pad by, so it is padded by
 * a share of the value itself, or by one when that value is zero. The range
 * is therefore never empty and `position` never divides by zero.
 */
export function domainOf(rows: PlotRow[]): [number, number] {
  const values = rows.flatMap((row) => [row.lower, row.point, row.upper]);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = high - low;
  const pad = span > 0 ? span * PADDING : Math.abs(low) * PADDING || 1;
  return [low - pad, high + pad];
}

/** Where a value sits along the axis, from 0 at the left end to 1 at the right. */
export function position(value: number, [low, high]: [number, number]): number {
  return (value - low) / (high - low);
}

/** Whether zero falls inside the axis, and so whether its reference line is drawn. */
export function showsZero([low, high]: [number, number]): boolean {
  return low < 0 && high > 0;
}
