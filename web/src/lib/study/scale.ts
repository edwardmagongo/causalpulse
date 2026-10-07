/**
 * How a metric's values map onto a colour. A metric with a value it should sit
 * at (zero bias, nominal coverage) diverges from it; one where lower or higher
 * is simply less or more (an error, a width, a runtime) runs from its low to
 * its high. These centres are a proposal, settled with the user, not a result.
 */

/** The nominal coverage of every interval the study reports. */
export const NOMINAL_COVERAGE = 0.95;

/**
 * The rate at which a true null is rejected at that level. Written out rather
 * than computed as `1 - NOMINAL_COVERAGE`, which is 0.050000000000000044 and
 * would make a matrix of exact 0.05 values not flat; a test ties the two.
 */
export const NOMINAL_FALSE_POSITIVE_RATE = 0.05;

/**
 * A proposal, for the project owner to approve or change from screenshots: how
 * far from its centre a share's colour scale may reach, and the values a share
 * can take. Without a cap, the single worst value in view (a coverage of 0)
 * sets the half-width, and every value near the centre looks the same white.
 * With one, a value beyond the cap takes the end colour and its number stays
 * in the cell. These are the only caps and bounds in the file.
 */
const SHARE_BOUNDS = { min: 0, max: 1 } as const;
/** A ratio that is not a sign: it cannot be negative, and has no upper limit. */
const RATIO_BOUNDS = { min: 0 } as const;
const COVERAGE_CAP = 0.15;
const FALSE_POSITIVE_RATE_CAP = 0.15;

/**
 * What a scale needs to say about its two ends. A diverging scale runs from
 * `lowEnd` to `highEnd` about its centre, clipped to the metric's possible
 * values; `lowBeyond` and `highBeyond` say whether any value in view lies past
 * that end, so a legend can print `≤ 0.8` there and the plain number where
 * none does. A sequential scale's ends are the lowest and highest value in
 * view; it names a beyond only if a value lay outside the metric's bounds.
 */
export type Scale =
  | {
      kind: "diverging";
      centre: number;
      halfWidth: number;
      centreLabel: string;
      lowEnd: number;
      highEnd: number;
      lowBeyond: boolean;
      highBeyond: boolean;
    }
  | { kind: "sequential"; low: number; high: number; lowBeyond?: true; highBeyond?: true }
  | { kind: "flat" }; // nothing to compare: one value, or none

/** The values a metric can take; an absent `max` is an open upper end. */
type Bounds = { readonly min: number; readonly max?: number };

type Rule = (
  | { kind: "diverging"; centre: number; centreLabel: string; cap?: number }
  | { kind: "sequential" }
) & { bounds?: Bounds };

const SEQUENTIAL: Rule = { kind: "sequential" };
const SEQUENTIAL_SHARE: Rule = { kind: "sequential", bounds: SHARE_BOUNDS };

const RULES: Record<string, Rule> = {
  bias: { kind: "diverging", centre: 0, centreLabel: "0" },
  relative_bias: { kind: "diverging", centre: 0, centreLabel: "0" },
  se_calibration: { kind: "diverging", centre: 1, centreLabel: "1", bounds: RATIO_BOUNDS },
  type_m: { kind: "diverging", centre: 1, centreLabel: "1", bounds: RATIO_BOUNDS },
  false_positive_rate: {
    kind: "diverging",
    centre: NOMINAL_FALSE_POSITIVE_RATE,
    centreLabel: `${NOMINAL_FALSE_POSITIVE_RATE}, the nominal level`,
    cap: FALSE_POSITIVE_RATE_CAP,
    bounds: SHARE_BOUNDS,
  },
  rmse: SEQUENTIAL,
  mae: SEQUENTIAL,
  mean_ci_width: SEQUENTIAL,
  type_s: SEQUENTIAL_SHARE,
  power: SEQUENTIAL_SHARE,
  runtime_seconds: SEQUENTIAL,
};

const COVERAGE_RULE: Rule = {
  kind: "diverging",
  centre: NOMINAL_COVERAGE,
  centreLabel: `${NOMINAL_COVERAGE}, the nominal level`,
  cap: COVERAGE_CAP,
  bounds: SHARE_BOUNDS,
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
  const bounds = rule.bounds;
  if (rule.kind === "diverging") {
    const { centre } = rule;
    const widest = Math.max(...finite.map((value) => Math.abs(value - centre)));
    if (widest === 0) return { kind: "flat" };
    const halfWidth = rule.cap === undefined ? widest : Math.min(widest, rule.cap);
    // An end the bounds do not clip is judged by the deviation that set the
    // half-width, so no rounding of `centre + halfWidth` can call a value beyond it.
    const max = bounds?.max;
    const clippedHigh = max !== undefined && max < centre + halfWidth;
    const clippedLow = bounds !== undefined && bounds.min > centre - halfWidth;
    return {
      kind: "diverging",
      centre,
      halfWidth,
      centreLabel: rule.centreLabel,
      lowEnd: clippedLow ? bounds.min : centre - halfWidth,
      highEnd: clippedHigh ? max : centre + halfWidth,
      lowBeyond: clippedLow
        ? finite.some((value) => value < bounds.min)
        : finite.some((value) => centre - value > halfWidth),
      highBeyond: clippedHigh
        ? finite.some((value) => value > max)
        : finite.some((value) => value - centre > halfWidth),
    };
  }
  let low = Math.min(...finite);
  let high = Math.max(...finite);
  if (low === high) return { kind: "flat" };
  const lowBeyond = bounds !== undefined && low < bounds.min;
  const max = bounds?.max;
  const highBeyond = max !== undefined && high > max;
  if (lowBeyond) low = bounds.min;
  if (highBeyond) high = max;
  if (low >= high) return { kind: "flat" };
  return {
    kind: "sequential",
    low,
    high,
    ...(lowBeyond ? { lowBeyond: true as const } : {}),
    ...(highBeyond ? { highBeyond: true as const } : {}),
  };
}

/**
 * Diverging: -1 to 1 about the centre. Sequential: 0 to 1. Flat: 0. Clamped,
 * so an infinite value takes the end of the scale like any out-of-range one.
 * `NaN` has no place on a scale and throws: `cellOf` never lets one through.
 */
export function position(value: number, scale: Scale): number {
  if (Number.isNaN(value)) throw new RangeError("a NaN has no position on a scale");
  if (scale.kind === "flat") return 0;
  if (scale.kind === "diverging") {
    return Math.max(-1, Math.min(1, (value - scale.centre) / scale.halfWidth));
  }
  return Math.max(0, Math.min(1, (value - scale.low) / (scale.high - scale.low)));
}

/** `count` positions evenly spaced from `from` to `to`, ends included. */
function evenly(from: number, to: number, count: number): number[] {
  return Array.from({ length: count }, (_, index) => from + ((to - from) * index) / (count - 1));
}

/**
 * The positions a legend's strip draws: only those a value can take, so the
 * colour beside each end label is the colour a cell with that end value gets.
 * The strip runs from the position of the low end to the position of the high
 * end (printed in the legend, clipped to the metric's bounds): -1 to 1 for a
 * scale the bounds do not clip, 0 to 1 for a sequential one. Five steps; a
 * diverging scale whose range holds its centre puts two steps each side of it
 * and one on it (position 0), so the centre colour is always shown. A flat
 * scale has no strip.
 */
export function legendSteps(scale: Scale): number[] {
  if (scale.kind === "flat") return [];
  if (scale.kind === "sequential") return evenly(position(scale.low, scale), position(scale.high, scale), 5);
  // An end the bounds do not clip is, by definition, the end of the scale: -1 or 1 exactly.
  const low = scale.lowEnd <= scale.centre - scale.halfWidth ? -1 : position(scale.lowEnd, scale);
  const high = scale.highEnd >= scale.centre + scale.halfWidth ? 1 : position(scale.highEnd, scale);
  if (low < 0 && high > 0) return [...evenly(low, 0, 3), ...evenly(0, high, 3).slice(1)];
  return evenly(low, high, 5);
}
