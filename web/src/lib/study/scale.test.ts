import { describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));

import { loadStudy } from "./bundle";
import { NOMINAL_COVERAGE, hasScaleRule, position, scaleFor } from "./scale";
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
    expect(centreOf("false_positive_rate")).toBeCloseTo(1 - NOMINAL_COVERAGE, 12);
    expect(centreOf("coverage_bootstrap_se")).toBe(NOMINAL_COVERAGE);
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
    for (const metric of ["rmse", "mae", "mean_ci_width", "type_s", "power", "runtime_seconds"]) {
      expect(scaleFor(metric, [1, 3])).toEqual({ kind: "sequential", low: 1, high: 3 });
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

  it("ignores a value that is not a finite number", () => {
    expect(scaleFor("rmse", [1, Number.NaN, 3, Number.POSITIVE_INFINITY])).toEqual({
      kind: "sequential",
      low: 1,
      high: 3,
    });
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
    const missing = Object.values(study.family_metrics)
      .flat()
      .filter((metric) => !hasScaleRule(metric));
    expect(missing).toEqual([]);
  });
});
