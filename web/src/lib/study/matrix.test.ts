import { describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));

import { formatNumber } from "@/lib/format";

import { loadStudy } from "./bundle";
import { cellOf, matrixOf, offeredFamilies } from "./matrix";
import type { FamilyRow, StudyBundle } from "./types";

function bundle(): StudyBundle {
  const loaded = loadStudy();
  if (!loaded.ok) throw new Error("the committed bundle must load");
  return loaded.study;
}

const study = bundle();

function rowOf(family: string, level: "scenario" | "scenario_by_size", method: string, scenario: string) {
  const row = study.families[family][level].find((r) => r.method === method && r.scenario === scenario);
  if (!row) throw new Error(`no ${method} x ${scenario} row in ${family}/${level}`);
  return row;
}

describe("cellOf on the committed bundle", () => {
  it("reads aipw x A bias as a value that prints as the published report does", () => {
    const cell = cellOf(rowOf("accuracy", "scenario", "aipw", "A"), "bias");
    expect(cell.kind).toBe("value");
    if (cell.kind === "value") {
      expect(formatNumber(cell.value)).toBe("0.001247");
      expect(cell.nUsed).toBeGreaterThan(0);
      expect(cell.nFitted).toBeGreaterThan(0);
      expect(cell.mcse).not.toBeNull();
    }
  });

  it("reads synthetic_control x S2 permutation coverage as 0.86", () => {
    const cell = cellOf(rowOf("coverage", "scenario", "synthetic_control", "S2"), "coverage_permutation");
    expect(cell.kind).toBe("value");
    if (cell.kind === "value") expect(formatNumber(cell.value)).toBe("0.86");
  });

  it("calls a scenario the estimator is not eligible for not_eligible", () => {
    expect(cellOf(rowOf("accuracy", "scenario", "aipw", "P1"), "bias")).toEqual({ kind: "not_eligible" });
  });

  it("calls a scenario whose fits were all refused refused", () => {
    const cell = cellOf(rowOf("accuracy", "scenario", "aipw", "H"), "bias");
    expect(cell.kind).toBe("refused");
    if (cell.kind === "refused") expect(cell.nRefused).toBeGreaterThan(0);
  });

  it("calls a fitted estimator whose fits feed no value not_applicable", () => {
    const cell = cellOf(rowOf("coverage", "scenario", "aipw", "A"), "coverage_posterior");
    expect(cell.kind).toBe("not_applicable");
    if (cell.kind === "not_applicable") expect(cell.nFitted).toBeGreaterThan(0);
  });

  it("calls a metric with no value although fits fed it undefined", () => {
    const cell = cellOf(rowOf("accuracy", "scenario", "aipw", "B"), "relative_bias");
    expect(cell.kind).toBe("undefined");
    if (cell.kind === "undefined") expect(cell.nUsed).toBeGreaterThan(0);
  });

  it("keeps a value whose Monte Carlo standard error is null", () => {
    const cell = cellOf(rowOf("sign_and_magnitude", "scenario", "bsts", "S2"), "type_m");
    expect(cell.kind).toBe("value");
    if (cell.kind === "value") expect(cell.mcse).toBeNull();
  });

  it("gives absent for a pair with no row", () => {
    expect(cellOf(undefined, "bias")).toEqual({ kind: "absent" });
  });
});

describe("cellOf on a row of its own", () => {
  const base: FamilyRow = { method: "m", scenario: "s", n_fitted: 3, n_ineligible: 0, n_refused: 0 };

  it("gives absent when the row does not carry the metric at all", () => {
    expect(cellOf(base, "bias")).toEqual({ kind: "absent" });
  });

  it("lets refusal win over ineligibility when nothing was fitted", () => {
    expect(cellOf({ ...base, n_fitted: 0, n_ineligible: 2, n_refused: 1 }, "bias")).toEqual({
      kind: "refused",
      nRefused: 1,
    });
  });

  it("takes a zero value as a value, not as a missing one", () => {
    expect(cellOf({ ...base, bias: 0, bias_mcse: 0.1, bias_n_used: 3 }, "bias")).toEqual({
      kind: "value",
      value: 0,
      mcse: 0.1,
      nUsed: 3,
      nFitted: 3,
    });
  });
});

describe("matrixOf", () => {
  const accuracy = { family: "accuracy", metric: "bias", shape: "all", estimatorFamily: "all" } as const;

  it("lays 12 estimators against 19 scenarios in the bundle's order", () => {
    const matrix = matrixOf(study, { ...accuracy, level: "scenario" });
    expect(matrix.metric).toBe("bias");
    expect(matrix.rows).toHaveLength(12);
    expect(matrix.columns).toHaveLength(19);
    expect(matrix.rows.map((r) => r.estimator)).toEqual(study.estimators.map((e) => e.name));
    expect(matrix.columns.map((c) => c.scenario)).toEqual(study.scenarios.map((s) => s.name));
    for (const row of matrix.rows) expect(row.cells).toHaveLength(19);
  });

  it("puts each cell under its own estimator and scenario", () => {
    const matrix = matrixOf(study, { ...accuracy, level: "scenario" });
    const aipw = matrix.rows.find((r) => r.estimator === "aipw");
    const a = matrix.columns.findIndex((c) => c.scenario === "A");
    const p1 = matrix.columns.findIndex((c) => c.scenario === "P1");
    const h = matrix.columns.findIndex((c) => c.scenario === "H");
    expect(aipw?.family).toBe("cross_sectional");
    expect(aipw?.cells[a].kind).toBe("value");
    expect(aipw?.cells[p1]).toEqual({ kind: "not_eligible" });
    expect(aipw?.cells[h].kind).toBe("refused");
  });

  it("knows the shape of every scenario and the family of every estimator", () => {
    const matrix = matrixOf(study, { ...accuracy, level: "scenario" });
    expect(matrix.columns.map((c) => c.scenario)).toEqual(study.scenarios.map((s) => s.name));
    expect(matrix.rows.every((r) => r.family !== null)).toBe(true);
  });

  it("keeps only the panel scenarios when asked for the panel shape", () => {
    const matrix = matrixOf(study, { ...accuracy, level: "scenario", shape: "panel" });
    expect(matrix.columns).toHaveLength(6);
    expect(matrix.columns.every((c) => c.shape === "panel")).toBe(true);
    expect(matrix.columns.map((c) => c.scenario)).toEqual(["P1", "P2", "P3", "P4", "P5", "P6"]);
    expect(matrix.rows).toHaveLength(12);
    for (const row of matrix.rows) expect(row.cells).toHaveLength(6);
  });

  it("keeps only the estimators of one family", () => {
    const matrix = matrixOf(study, { ...accuracy, level: "scenario", estimatorFamily: "donor_pool" });
    expect(matrix.rows.map((r) => r.estimator)).toEqual(["bsts", "synthetic_control"]);
    expect(matrix.rows.every((r) => r.family === "donor_pool")).toBe(true);
  });

  it("splits scenarios by size at scenario_by_size, ordered by units then periods", () => {
    const matrix = matrixOf(study, { ...accuracy, level: "scenario_by_size" });
    expect(matrix.columns).toHaveLength(32);
    expect(new Set(matrix.columns.map((c) => c.key)).size).toBe(32);
    const s5 = matrix.columns.filter((c) => c.scenario === "S5");
    expect(s5).toHaveLength(1);
    expect(s5[0].nPeriods).toBe(6);
    for (const scenario of study.scenarios.map((s) => s.name)) {
      const own = matrix.columns.filter((c) => c.scenario === scenario);
      const sorted = [...own].sort(
        (x, y) => (x.nUnits ?? 0) - (y.nUnits ?? 0) || (x.nPeriods ?? 0) - (y.nPeriods ?? 0),
      );
      expect(own).toEqual(sorted);
    }
    const order = matrix.columns.map((c) => c.scenario).filter((name, i, all) => all.indexOf(name) === i);
    expect(order).toEqual(study.scenarios.map((s) => s.name));
    const changes = matrix.columns.filter((c, i, all) => i > 0 && all[i - 1].scenario !== c.scenario);
    expect(changes).toHaveLength(order.length - 1);
  });

  it("has no cell for a metric the family does not hold, never a made-up one", () => {
    const matrix = matrixOf(study, { ...accuracy, level: "scenario", metric: "power" });
    for (const row of matrix.rows) for (const cell of row.cells) expect(["absent", "not_eligible", "refused"]).toContain(cell.kind);
  });

  it("gives an empty matrix for a family the bundle does not hold", () => {
    const matrix = matrixOf(study, { ...accuracy, family: "nope", level: "scenario" });
    expect(matrix.rows.every((r) => r.cells.every((c) => c.kind === "absent"))).toBe(true);
  });
});

describe("offeredFamilies", () => {
  it("lists the six metric families, not the false-positive table, each at both levels", () => {
    const offered = offeredFamilies(study);
    expect(offered.map((o) => o.family)).toEqual(Object.keys(study.family_metrics));
    expect(offered).toHaveLength(6);
    expect(offered.some((o) => o.family === "false_positive_rate")).toBe(false);
    for (const family of offered) {
      expect(family.levels).toEqual(["scenario", "scenario_by_size"]);
      expect(family.metrics).toEqual(study.family_metrics[family.family]);
    }
  });

  it("does not offer a family the bundle's tables do not hold", () => {
    const trimmed: StudyBundle = {
      ...study,
      families: Object.fromEntries(Object.entries(study.families).filter(([name]) => name !== "runtime")),
    };
    expect(offeredFamilies(trimmed).map((o) => o.family)).not.toContain("runtime");
  });
});
