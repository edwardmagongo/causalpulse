import { describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));

import { loadStudy } from "./bundle";
import { matrixOf } from "./matrix";
import { defaultView, parseView, viewHref } from "./query";
import type { View } from "./query";
import type { StudyBundle } from "./types";

function bundle(): StudyBundle {
  const loaded = loadStudy();
  if (!loaded.ok) throw new Error("the committed bundle must load");
  return loaded.study;
}

const study = bundle();

/** `/study?a=b` to its params, as a page receives them. */
function paramsOf(href: string): Record<string, string> {
  const query = href.split("?")[1] ?? "";
  return Object.fromEntries(new URLSearchParams(query));
}

describe("defaultView", () => {
  it("is accuracy, bias, by scenario, every shape and family, a heatmap, unsorted", () => {
    expect(defaultView(study)).toEqual({
      family: "accuracy",
      metric: "bias",
      level: "scenario",
      shape: "all",
      estimatorFamily: "all",
      display: "heatmap",
      sort: null,
      direction: "asc",
    });
  });
});

describe("parseView", () => {
  it("reads no parameters as the default view", () => {
    expect(parseView(study, {})).toEqual(defaultView(study));
  });

  it("reads every key", () => {
    const view = parseView(study, {
      family: "coverage",
      metric: "coverage_bootstrap",
      level: "scenario_by_size",
      shape: "panel",
      estimators: "panel",
      display: "table",
      sort: "estimator",
      dir: "desc",
    });
    expect(view).toEqual({
      family: "coverage",
      metric: "coverage_bootstrap",
      level: "scenario_by_size",
      shape: "panel",
      estimatorFamily: "panel",
      display: "table",
      sort: "estimator",
      direction: "desc",
    });
  });

  it("falls back on its own for each unknown value", () => {
    const base = defaultView(study);
    expect(parseView(study, { family: "nope" })).toEqual(base);
    expect(parseView(study, { metric: "nope" })).toEqual(base);
    expect(parseView(study, { level: "nope" })).toEqual(base);
    expect(parseView(study, { shape: "nope" })).toEqual(base);
    expect(parseView(study, { estimators: "nope" })).toEqual(base);
    expect(parseView(study, { display: "nope" })).toEqual(base);
    expect(parseView(study, { dir: "nope" })).toEqual(base);
    // one bad key does not spoil a good one
    expect(parseView(study, { family: "nope", display: "table" })).toEqual({ ...base, display: "table" });
  });

  it("does not offer false_positive_rate as a family, though detection holds it as a metric", () => {
    expect(parseView(study, { family: "false_positive_rate" }).family).toBe("accuracy");
    const view = parseView(study, { family: "detection", metric: "false_positive_rate" });
    expect(view).toMatchObject({ family: "detection", metric: "false_positive_rate" });
  });

  it("gives a metric from another family that family's first metric", () => {
    const view = parseView(study, { family: "coverage", metric: "bias" });
    expect(view.family).toBe("coverage");
    expect(view.metric).toBe("coverage_analytic");
  });

  it("keeps the estimator family and the shape vocabularies apart", () => {
    expect(parseView(study, { shape: "cross_section" }).shape).toBe("cross_section");
    expect(parseView(study, { shape: "cross_sectional" }).shape).toBe("all");
    expect(parseView(study, { estimators: "cross_sectional" }).estimatorFamily).toBe("cross_sectional");
    expect(parseView(study, { estimators: "cross_section" }).estimatorFamily).toBe("all");
  });

  it("uses the first value of a repeated key", () => {
    expect(parseView(study, { display: ["table", "heatmap"] }).display).toBe("table");
    expect(parseView(study, { display: ["nope", "table"] }).display).toBe("heatmap");
    expect(parseView(study, { display: [] }).display).toBe("heatmap");
  });

  it("accepts a sort only when it names the estimator or a column of the selected matrix", () => {
    expect(parseView(study, { sort: "estimator" }).sort).toBe("estimator");
    expect(parseView(study, { sort: "A" }).sort).toBe("A");
    expect(parseView(study, { sort: "nope" }).sort).toBeNull();
    // `A` is a scenario column, not a column of the by-size matrix
    expect(parseView(study, { level: "scenario_by_size", sort: "A" }).sort).toBeNull();
    // a shape filter removes the columns of the other shapes
    expect(parseView(study, { shape: "panel", sort: "A" }).sort).toBeNull();
  });

  it("accepts a by-size column key at the by-size level", () => {
    const view = defaultView(study);
    const columns = matrixOf(study, { ...view, level: "scenario_by_size" }).columns;
    expect(columns.length).toBeGreaterThan(0);
    const key = columns[0].key;
    expect(parseView(study, { level: "scenario_by_size", sort: key }).sort).toBe(key);
    expect(parseView(study, { sort: key }).sort).toBeNull();
  });

  describe("for a family that lacks the by-size level", () => {
    // The committed study, with one family's by-size table taken out: every family in it has both.
    const lacking: StudyBundle = {
      ...study,
      families: { ...study.families, runtime: { scenario: study.families.runtime.scenario } },
    };
    const bySizeKey = matrixOf(study, {
      ...defaultView(study),
      family: "runtime",
      metric: study.family_metrics.runtime[0],
      level: "scenario_by_size",
    }).columns[0].key;

    it("falls back to the level the family offers", () => {
      expect(Object.keys(study.families.runtime).sort()).toEqual(["scenario", "scenario_by_size"]);
      expect(Object.keys(lacking.families.runtime)).toEqual(["scenario"]);
      const view = parseView(lacking, { family: "runtime", level: "scenario_by_size" });
      expect(view.family).toBe("runtime");
      expect(view.level).toBe("scenario");
      // The href of the view it gives names no level, so it parses to the same view.
      const href = viewHref(lacking, view);
      expect(Object.keys(paramsOf(href))).toEqual(["family", "metric"]);
      expect(parseView(lacking, paramsOf(href))).toEqual(view);
    });

    it("keeps the by-size level for a family that has it", () => {
      expect(parseView(lacking, { family: "coverage", level: "scenario_by_size" }).level).toBe("scenario_by_size");
      // and for the same family in the study that holds its by-size table
      expect(parseView(study, { family: "runtime", level: "scenario_by_size" }).level).toBe("scenario_by_size");
    });

    it("reads a sort against the by-scenario view it fell back to", () => {
      expect(bySizeKey).toContain(":");
      // In the whole study the key is a column of the by-size view.
      expect(parseView(study, { family: "runtime", level: "scenario_by_size", sort: bySizeKey }).sort).toBe(bySizeKey);
      expect(parseView(lacking, { family: "runtime", level: "scenario_by_size", sort: bySizeKey }).sort).toBeNull();
      // A by-scenario column is one of the view's columns, so it is kept.
      expect(parseView(lacking, { family: "runtime", level: "scenario_by_size", sort: "A" }).sort).toBe("A");
    });
  });

  it("never carries an unparsed value into the view or its href", () => {
    const hostile = "<script>alert(1)</script>";
    const view = parseView(study, {
      family: hostile,
      metric: hostile,
      level: hostile,
      shape: hostile,
      estimators: hostile,
      display: hostile,
      sort: hostile,
      dir: hostile,
    });
    expect(view).toEqual(defaultView(study));
    expect(viewHref(study, view)).toBe("/study");
    const metricOnly = parseView(study, { metric: "<script>" });
    expect(metricOnly.metric).toBe("bias");
    expect(viewHref(study, metricOnly)).not.toContain("<");
  });
});

describe("viewHref", () => {
  it("is exactly /study for the default view", () => {
    expect(viewHref(study, defaultView(study))).toBe("/study");
  });

  it("holds only the keys that differ from the default", () => {
    const view: View = { ...defaultView(study), display: "table" };
    expect(viewHref(study, view)).toBe("/study?display=table");
    expect(viewHref(study, { ...defaultView(study), direction: "desc" })).toBe("/study?dir=desc");
  });

  it("round-trips every key through parseView", () => {
    const base = defaultView(study);
    const bySize = matrixOf(study, { ...base, level: "scenario_by_size" }).columns[0].key;
    const views: View[] = [
      base,
      { ...base, family: "coverage", metric: "coverage_analytic" },
      { ...base, family: "coverage", metric: "mean_ci_width" },
      { ...base, family: "detection", metric: "false_positive_rate" },
      { ...base, family: "sign_and_magnitude", metric: "type_m" },
      { ...base, metric: "rmse" },
      { ...base, level: "scenario_by_size" },
      { ...base, shape: "cross_section" },
      { ...base, shape: "panel" },
      { ...base, shape: "donor_pool" },
      { ...base, estimatorFamily: "cross_sectional" },
      { ...base, estimatorFamily: "panel" },
      { ...base, estimatorFamily: "donor_pool" },
      { ...base, display: "table" },
      { ...base, sort: "estimator" },
      { ...base, sort: "A", direction: "desc" },
      { ...base, direction: "desc" },
      { ...base, level: "scenario_by_size", sort: bySize, direction: "desc" },
      {
        family: "coverage",
        metric: "coverage_permutation",
        level: "scenario_by_size",
        shape: "donor_pool",
        estimatorFamily: "donor_pool",
        display: "table",
        sort: "estimator",
        direction: "desc",
      },
    ];
    for (const view of views) {
      const href = viewHref(study, view);
      expect(parseView(study, paramsOf(href)), href).toEqual(view);
    }
  });

  it("never emits a character that could open a tag", () => {
    const href = viewHref(study, { ...defaultView(study), sort: "<b>" });
    expect(href).not.toContain("<");
  });
});
