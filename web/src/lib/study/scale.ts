/**
 * How a metric's values map onto a colour. A metric with a value it should sit
 * at (zero bias, nominal coverage) diverges from it; one where lower or higher
 * is simply less or more (an error, a width, a runtime) runs from its low to
 * its high. These centres are a proposal, settled with the user, not a result.
 */

/** The nominal coverage of every interval the study reports. */
export const NOMINAL_COVERAGE = 0.95;

export type Scale =
  | { kind: "diverging"; centre: number; halfWidth: number; centreLabel: string }
  | { kind: "sequential"; low: number; high: number }
  | { kind: "flat" }; // nothing to compare: one value, or none

type Rule = { kind: "diverging"; centre: number; centreLabel: string } | { kind: "sequential" };

const SEQUENTIAL: Rule = { kind: "sequential" };

const RULES: Record<string, Rule> = {
  bias: { kind: "diverging", centre: 0, centreLabel: "0" },
  relative_bias: { kind: "diverging", centre: 0, centreLabel: "0" },
  se_calibration: { kind: "diverging", centre: 1, centreLabel: "1" },
  type_m: { kind: "diverging", centre: 1, centreLabel: "1" },
  false_positive_rate: {
    kind: "diverging",
    centre: 1 - NOMINAL_COVERAGE,
    centreLabel: String(Number((1 - NOMINAL_COVERAGE).toFixed(12))),
  },
  rmse: SEQUENTIAL,
  mae: SEQUENTIAL,
  mean_ci_width: SEQUENTIAL,
  type_s: SEQUENTIAL,
  power: SEQUENTIAL,
  runtime_seconds: SEQUENTIAL,
};

const COVERAGE_RULE: Rule = {
  kind: "diverging",
  centre: NOMINAL_COVERAGE,
  centreLabel: String(NOMINAL_COVERAGE),
};

function ruleFor(metric: string): Rule | undefined {
  if (metric.startsWith("coverage_")) return COVERAGE_RULE;
  return Object.hasOwn(RULES, metric) ? RULES[metric] : undefined;
}

/** Whether a metric has a deliberate rule, rather than the unknown-metric default. */
export function hasScaleRule(metric: string): boolean {
  return ruleFor(metric) !== undefined;
}

export function scaleFor(metric: string, values: number[]): Scale {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return { kind: "flat" };
  const rule = ruleFor(metric) ?? SEQUENTIAL;
  if (rule.kind === "diverging") {
    const halfWidth = Math.max(...finite.map((value) => Math.abs(value - rule.centre)));
    if (halfWidth === 0) return { kind: "flat" };
    return { kind: "diverging", centre: rule.centre, halfWidth, centreLabel: rule.centreLabel };
  }
  const low = Math.min(...finite);
  const high = Math.max(...finite);
  if (low === high) return { kind: "flat" };
  return { kind: "sequential", low, high };
}

/** Diverging: -1 to 1 about the centre. Sequential: 0 to 1. Flat: 0. Clamped. */
export function position(value: number, scale: Scale): number {
  if (scale.kind === "flat") return 0;
  if (scale.kind === "diverging") {
    return Math.max(-1, Math.min(1, (value - scale.centre) / scale.halfWidth));
  }
  return Math.max(0, Math.min(1, (value - scale.low) / (scale.high - scale.low)));
}
