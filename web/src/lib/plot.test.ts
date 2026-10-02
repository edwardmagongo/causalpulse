import { describe, expect, it } from "vitest";

import type { ComparisonGroup, Estimate } from "@/lib/api/types";
import lalonde from "@/test/fixtures/lalonde.analysis.json";
import lalondeSummary from "@/test/fixtures/lalonde.summary.json";
import prop99 from "@/test/fixtures/prop99.analysis.json";
import prop99Summary from "@/test/fixtures/prop99.summary.json";

import { domainOf, panels, position, showsZero, type PlotRow } from "./plot";

const row = (method: string, lower: number, point: number, upper: number): PlotRow => ({
  method,
  lower,
  point,
  upper,
  basis: "analytic",
  level: 0.95,
});

describe("panels", () => {
  it("draws the two LaLonde methods on one axis, because the API compared them", () => {
    const [panel, ...rest] = panels(lalonde.estimates as Estimate[], lalondeSummary.comparisons);
    expect(rest).toEqual([]);
    expect(panel.rows.map((r) => r.method)).toEqual(["difference_in_means", "lin_regression"]);
    expect([panel.estimand, panel.scale]).toEqual(["ATE", "absolute"]);
  });

  it("draws a single method that was not compared", () => {
    const [panel] = panels(prop99.estimates as Estimate[], prop99Summary.comparisons);
    expect(panel.rows.map((r) => r.method)).toEqual(["synthetic_control"]);
    expect(panel.rows[0].basis).toBe("permutation");
  });

  it("puts estimates of different estimands on different axes", () => {
    const estimates = [...lalonde.estimates, ...prop99.estimates] as Estimate[];
    const groups = [...lalondeSummary.comparisons, ...prop99Summary.comparisons];
    const drawn = panels(estimates, groups);
    expect(drawn.map((p) => p.estimand)).toEqual(["ATE", "ATT"]);
    // Each axis spans its own rows only: the ATE's thousands do not stretch the ATT's tens.
    expect(drawn[1].domain[0]).toBeGreaterThan(-200);
    expect(drawn[0].domain[0]).toBeLessThan(-8000);
  });

  it("leaves out a method with no estimate, and a group left with none", () => {
    const failed: Estimate = { method: "lin_regression", band: null, summary: null, error: "boom" };
    const estimates = [lalonde.estimates[0] as Estimate, failed];
    const [panel] = panels(estimates, lalondeSummary.comparisons);
    expect(panel.rows.map((r) => r.method)).toEqual(["difference_in_means"]);

    const group: ComparisonGroup = { estimand: "ATE", scale: "absolute", methods: ["lin_regression"] };
    expect(panels([failed], [group])).toEqual([]);
  });
});

describe("domainOf and position", () => {
  it("pads the range so no interval end sits on the edge", () => {
    const domain = domainOf([row("a", -10, -5, 0), row("b", 0, 5, 10)]);
    expect(domain).toEqual([-11, 11]);
    expect(position(-10, domain)).toBeCloseTo(1 / 22);
    expect(position(0, domain)).toBeCloseTo(0.5);
    expect(position(10, domain)).toBeCloseTo(21 / 22);
  });

  it("never returns an empty range, even when every value coincides", () => {
    const flat = domainOf([row("a", 4, 4, 4)]);
    expect(flat[1]).toBeGreaterThan(flat[0]);
    expect(position(4, flat)).toBeCloseTo(0.5);

    const zero = domainOf([row("a", 0, 0, 0)]);
    expect(zero).toEqual([-1, 1]);
    expect(Number.isFinite(position(0, zero))).toBe(true);
  });

  it("includes a point that lies outside its own interval", () => {
    // A bias-corrected interval need not contain the point estimate.
    const domain = domainOf([row("a", 1, 5, 3)]);
    expect(domain[1]).toBeGreaterThan(5);
  });
});

describe("showsZero", () => {
  it("is true only when zero is strictly inside the axis", () => {
    expect(showsZero([-1, 1])).toBe(true);
    expect(showsZero([1, 5])).toBe(false);
    expect(showsZero([-5, -1])).toBe(false);
  });
});
