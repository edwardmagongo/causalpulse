import { describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));

import { loadStudy } from "./bundle";
import {
  NOMINAL_COVERAGE,
  NOMINAL_FALSE_POSITIVE_RATE,
  hasScaleRule,
  legendSteps,
  position,
  scaleFor,
} from "./scale";
import type { StudyBundle } from "./types";

function bundle(): StudyBundle {
  const loaded = loadStudy();
  if (!loaded.ok) throw new Error("the committed bundle must load");
  return loaded.study;
}

const study = bundle();

describe("scaleFor", () => {
  it("centres coverage at the nominal level", () => {
    const scale = scaleFor("coverage_analytic", [0.8, 0.95, 0.99]);
    expect(scale.kind).toBe("diverging");
    if (scale.kind === "diverging") {
      expect(scale.centre).toBe(NOMINAL_COVERAGE);
      expect(scale.halfWidth).toBeCloseTo(0.15, 12);
    }
    expect(position(0.95, scale)).toBe(0);
    expect(position(0.8, scale)).toBe(-1);
    expect(position(0.99, scale)).toBeCloseTo(0.04 / 0.15, 12);
  });

  it("centres each diverging metric where the study says it is unbiased", () => {
    const centreOf = (metric: string) => {
      const scale = scaleFor(metric, [0, 2]);
      return scale.kind === "diverging" ? scale.centre : null;
    };
    expect(centreOf("bias")).toBe(0);
    expect(centreOf("relative_bias")).toBe(0);
    expect(centreOf("se_calibration")).toBe(1);
    expect(centreOf("type_m")).toBe(1);
    expect(centreOf("false_positive_rate")).toBe(NOMINAL_FALSE_POSITIVE_RATE);
    expect(centreOf("coverage_bootstrap_se")).toBe(NOMINAL_COVERAGE);
  });

  it("ties the false-positive centre to the nominal coverage, to 12 places", () => {
    expect(NOMINAL_COVERAGE + NOMINAL_FALSE_POSITIVE_RATE).toBeCloseTo(1, 12);
  });

  it("is flat when every false-positive rate sits exactly at 0.05, and puts 0.05 at 0 otherwise", () => {
    expect(scaleFor("false_positive_rate", [0.05, 0.05])).toEqual({ kind: "flat" });
    const mixed = scaleFor("false_positive_rate", [0.05, 0.2, 0.01]);
    expect(mixed.kind).toBe("diverging");
    expect(position(0.05, mixed)).toBe(0);
  });

  it("scales the farther side to the half-width, so the other side stays inside", () => {
    const scale = scaleFor("bias", [-1, 3]);
    expect(position(3, scale)).toBe(1);
    expect(position(-1, scale)).toBeCloseTo(-1 / 3, 12);
  });

  it("is flat when every value sits at the centre", () => {
    expect(scaleFor("coverage_analytic", [0.95, 0.95])).toEqual({ kind: "flat" });
    expect(position(0.95, scaleFor("coverage_analytic", [0.95, 0.95]))).toBe(0);
  });

  it("is sequential for an error or a width, low to high", () => {
    for (const metric of ["rmse", "mae", "mean_ci_width", "runtime_seconds"]) {
      expect(scaleFor(metric, [1, 3])).toEqual({ kind: "sequential", low: 1, high: 3 });
    }
    // A share cannot be 3: type_s and power are bounded to [0, 1] and read in that range.
    for (const metric of ["type_s", "power"]) {
      expect(scaleFor(metric, [0.1, 0.3])).toEqual({ kind: "sequential", low: 0.1, high: 0.3 });
    }
    expect(position(2, scaleFor("rmse", [1, 3]))).toBe(0.5);
    expect(position(1, scaleFor("rmse", [1, 3]))).toBe(0);
    expect(position(3, scaleFor("rmse", [1, 3]))).toBe(1);
  });

  it("is flat when one value, or none, leaves nothing to compare", () => {
    expect(scaleFor("rmse", [2, 2])).toEqual({ kind: "flat" });
    expect(scaleFor("rmse", [2])).toEqual({ kind: "flat" });
    expect(scaleFor("rmse", [])).toEqual({ kind: "flat" });
    expect(scaleFor("bias", [])).toEqual({ kind: "flat" });
    expect(position(7, { kind: "flat" })).toBe(0);
  });

  it("reads an unknown metric as sequential", () => {
    expect(scaleFor("not_a_metric", [1, 3])).toEqual({ kind: "sequential", low: 1, high: 3 });
    expect(hasScaleRule("not_a_metric")).toBe(false);
  });

  it("clamps a value outside the scale", () => {
    expect(position(5, scaleFor("rmse", [1, 3]))).toBe(1);
    expect(position(-5, scaleFor("rmse", [1, 3]))).toBe(0);
    expect(position(10, scaleFor("bias", [-1, 1]))).toBe(1);
    expect(position(-10, scaleFor("bias", [-1, 1]))).toBe(-1);
  });

  it("takes an infinite value to the end of the scale, like any value out of range", () => {
    const sequential = scaleFor("rmse", [1, 3]);
    const diverging = scaleFor("bias", [-1, 1]);
    expect(position(Number.POSITIVE_INFINITY, sequential)).toBe(1);
    expect(position(Number.NEGATIVE_INFINITY, sequential)).toBe(0);
    expect(position(Number.POSITIVE_INFINITY, diverging)).toBe(1);
    expect(position(Number.NEGATIVE_INFINITY, diverging)).toBe(-1);
    expect(position(Number.POSITIVE_INFINITY, { kind: "flat" })).toBe(0);
  });

  it("throws on NaN, on every kind of scale, rather than place it anywhere", () => {
    for (const scale of [scaleFor("rmse", [1, 3]), scaleFor("bias", [-1, 1]), { kind: "flat" } as const]) {
      expect(() => position(Number.NaN, scale)).toThrow(RangeError);
    }
  });

  it("ignores a value that is not a finite number", () => {
    expect(scaleFor("rmse", [1, Number.NaN, 3, Number.POSITIVE_INFINITY])).toEqual({
      kind: "sequential",
      low: 1,
      high: 3,
    });
  });
});

describe("a capped scale", () => {
  const coverage = scaleFor("coverage_analytic", [0, 0.93, 0.95, 0.97]);

  it("takes the cap as its half-width when a value lies farther from the centre than that", () => {
    expect(coverage.kind).toBe("diverging");
    if (coverage.kind !== "diverging") return;
    expect(coverage.halfWidth).toBeCloseTo(0.15, 12);
    expect(coverage.centre).toBe(NOMINAL_COVERAGE);
  });

  it("tells 0.93 from 0.97 and from the centre, and gives the end colour to a coverage of 0", () => {
    expect(position(0.93, coverage)).toBeCloseTo(-0.02 / 0.15, 12);
    expect(position(0.97, coverage)).toBeCloseTo(0.02 / 0.15, 12);
    expect(position(0.93, coverage)).toBeLessThan(-0.13);
    expect(position(0.97, coverage)).toBeGreaterThan(0.13);
    expect(position(0.95, coverage)).toBe(0);
    expect(position(0, coverage)).toBe(-1);
  });

  it("names its ends: clipped to the possible values, with whether any value lies beyond", () => {
    expect(coverage.kind).toBe("diverging");
    if (coverage.kind !== "diverging") return;
    expect(coverage.lowEnd).toBeCloseTo(0.8, 12);
    expect(coverage.lowBeyond).toBe(true); // the coverage of 0
    expect(coverage.highEnd).toBe(1); // 1.1 clipped to the largest possible share
    expect(coverage.highBeyond).toBe(false);
  });

  it("never reaches below 0 or above 1 for a share, and flags a value beyond the cap on the high side", () => {
    const high = scaleFor("false_positive_rate", [0.05, 0.9]);
    expect(high.kind).toBe("diverging");
    if (high.kind !== "diverging") return;
    expect(high.halfWidth).toBeCloseTo(0.15, 12);
    expect(high.lowEnd).toBe(0); // -0.1 clipped to 0
    expect(high.lowBeyond).toBe(false);
    expect(high.highEnd).toBeCloseTo(0.2, 12);
    expect(high.highBeyond).toBe(true);
    expect(position(0.9, high)).toBe(1);
  });

  it("uses the largest deviation, not the cap, when every value sits within the cap", () => {
    const near = scaleFor("coverage_analytic", [0.9, 0.95, 0.99]);
    expect(near.kind).toBe("diverging");
    if (near.kind !== "diverging") return;
    expect(near.halfWidth).toBeCloseTo(0.05, 12);
    expect(near.lowBeyond).toBe(false);
    expect(near.highBeyond).toBe(false);
    expect(position(0.9, near)).toBe(-1);
  });

  it("does not call a value beyond an end it sits exactly on", () => {
    const exact = scaleFor("coverage_analytic", [0.9, 1]);
    expect(exact.kind).toBe("diverging");
    if (exact.kind !== "diverging") return;
    expect(exact.highEnd).toBe(1);
    expect(exact.highBeyond).toBe(false);
  });

  it("gives only the coverage metrics and false_positive_rate a cap, and leaves the others as they were", () => {
    const bias = scaleFor("bias", [-3, 0.1]);
    expect(bias.kind).toBe("diverging");
    if (bias.kind !== "diverging") return;
    expect(bias.halfWidth).toBe(3);
    expect(bias.lowEnd).toBe(-3);
    expect(bias.highEnd).toBe(3);
    expect(bias.lowBeyond).toBe(false);
    expect(bias.highBeyond).toBe(false);
    const relative = scaleFor("relative_bias", [-40, 40]);
    expect(relative.kind === "diverging" && relative.halfWidth > 0.15).toBe(true);
    for (const metric of ["se_calibration", "type_m"]) {
      const scale = scaleFor(metric, [1, 41]);
      expect(scale.kind === "diverging" && scale.halfWidth).toBe(40);
    }
  });

  it("bounds a ratio below at 0 and leaves its top open, so its low end is never negative", () => {
    // Centre 1, half-width 3.344: the unclipped low end would be -2.344, for a ratio that cannot be negative.
    for (const metric of ["type_m", "se_calibration"]) {
      const scale = scaleFor(metric, [1, 4.344]);
      expect(scale).toMatchObject({ kind: "diverging", centre: 1, lowEnd: 0, highEnd: 4.344, lowBeyond: false, highBeyond: false });
      const beyond = scaleFor(metric, [-0.5, 4.344]);
      expect(beyond).toMatchObject({ lowEnd: 0, lowBeyond: true, highBeyond: false });
      // No upper bound: a large value sets the half-width and the high end, and is never beyond it.
      const wide = scaleFor(metric, [0.5, 100]);
      expect(wide).toMatchObject({ lowEnd: 0, highEnd: 100, highBeyond: false });
    }
  });

  it("draws a ratio's strip from the colour of a ratio of 0, not from -1", () => {
    const scale = scaleFor("type_m", [1, 4.344]);
    const steps = legendSteps(scale);
    expect(steps).toHaveLength(5);
    expect(steps[0]).toBeCloseTo(position(0, scale), 12);
    expect(steps[0]).toBeCloseTo(-1 / 3.344, 12);
    expect(steps[steps.length - 1]).toBe(1);
    expect(steps).toContain(0);
  });

  it("bounds the sequential shares, and says nothing of it while every value is possible", () => {
    expect(scaleFor("power", [0.2, 0.9])).toEqual({ kind: "sequential", low: 0.2, high: 0.9 });
    expect(scaleFor("type_s", [0, 0.4])).toEqual({ kind: "sequential", low: 0, high: 0.4 });
    expect(scaleFor("power", [-0.1, 0.5])).toEqual({ kind: "sequential", low: 0, high: 0.5, lowBeyond: true });
    expect(scaleFor("power", [0.5, 1.2])).toEqual({ kind: "sequential", low: 0.5, high: 1, highBeyond: true });
  });
});

describe("legendSteps", () => {
  it("runs a coverage scale from the blue end to where a coverage of 1 sits, through the centre", () => {
    const scale = scaleFor("coverage_analytic", [0, 0.93, 0.95, 0.97]);
    const steps = legendSteps(scale);
    expect(steps).toHaveLength(5);
    expect(steps[0]).toBe(-1);
    expect(steps[steps.length - 1]).toBeCloseTo(position(1, scale), 12);
    expect(steps[steps.length - 1]).toBeCloseTo(1 / 3, 12);
    expect(steps).toContain(0);
    expect(Math.max(...steps)).toBeLessThanOrEqual(position(1, scale));
    expect([...steps].sort((a, b) => a - b)).toEqual(steps);
  });

  it("starts a false-positive-rate scale at where a rate of 0 sits, not at -1", () => {
    const scale = scaleFor("false_positive_rate", [0.02, 0.05, 0.3]);
    const steps = legendSteps(scale);
    expect(steps).toHaveLength(5);
    expect(steps[0]).toBeCloseTo(position(0, scale), 12);
    expect(steps[0]).toBeCloseTo(-1 / 3, 12);
    expect(steps[steps.length - 1]).toBe(1);
    expect(steps).toContain(0);
    expect(Math.min(...steps)).toBeGreaterThanOrEqual(position(0, scale));
  });

  it("keeps an unclipped diverging scale at -1 to 1, and a sequential one at 0 to 1", () => {
    expect(legendSteps(scaleFor("bias", [-0.2, 0.1, 0.2]))).toEqual([-1, -0.5, 0, 0.5, 1]);
    expect(legendSteps(scaleFor("bias", [-0.3, 0.1]))).toEqual([-1, -0.5, 0, 0.5, 1]);
    expect(legendSteps(scaleFor("rmse", [0.1, 0.4]))).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(legendSteps(scaleFor("power", [-0.1, 0.5]))).toEqual([0, 0.25, 0.5, 0.75, 1]);
  });

  it("draws nothing for a flat scale", () => {
    expect(legendSteps({ kind: "flat" })).toEqual([]);
  });
});

describe("the scales against the committed bundle", () => {
  it("puts the nominal level where the study's own standard error puts it", () => {
    const { replications, coverage_mcse } = study.overview;
    expect(Math.sqrt((NOMINAL_COVERAGE * (1 - NOMINAL_COVERAGE)) / replications)).toBeCloseTo(
      coverage_mcse,
      12,
    );
  });

  it("gives every metric in family_metrics an explicit rule", () => {
    const metrics = Object.values(study.family_metrics).flat();
    expect(metrics).toHaveLength(16);
    expect(metrics.filter((metric) => !hasScaleRule(metric))).toEqual([]);
  });
});
