import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));

import StudyPage from "@/app/(app)/study/page";
import { DASH } from "@/lib/format";
import { loadStudy } from "@/lib/study/bundle";
import { matrixOf, offeredFamilies } from "@/lib/study/matrix";
import type { Cell, Matrix } from "@/lib/study/matrix";
import { defaultView, viewHref } from "@/lib/study/query";
import type { View } from "@/lib/study/query";
import { scaleFor } from "@/lib/study/scale";
import type { StudyBundle } from "@/lib/study/types";

import {
  CATE_COLUMNS,
  CateSection,
  FindingsSection,
  LimitationsSection,
  MethodsSection,
  OverviewSection,
  P7Section,
  P7_COLUMNS,
  StudyReport,
  StudyUnavailable,
} from "./StudySections";
import { MatrixHeatmap } from "./MatrixHeatmap";
import { ScaleLegend } from "./ScaleLegend";
import { StudyTable } from "./StudyTable";

afterEach(cleanup);

const loaded = loadStudy();
if (!loaded.ok) throw new Error("the committed bundle must load");
const study: StudyBundle = loaded.study;

describe("StudyReport", () => {
  it("sets the sections in the report's order, each an h2 with its own id", () => {
    render(<StudyReport study={study} />);
    const headings = screen.getAllByRole("heading", { level: 2 });
    expect(headings.map((heading) => heading.id)).toEqual([
      "overview",
      "findings",
      "methods",
      "cate",
      "p7",
      "limitations",
    ]);
  });
});

describe("OverviewSection", () => {
  it("states the study's size and one provenance line per run", () => {
    const { container } = render(<OverviewSection overview={study.overview} provenance={study.provenance} />);
    expect(container.textContent).toContain(`${study.overview.n_cells}`);
    expect(container.textContent).toContain(`${study.overview.n_estimators}`);
    expect(container.textContent).toContain(`${study.overview.n_scenarios}`);
    const lines = screen.getAllByRole("listitem");
    expect(lines).toHaveLength(3);
    study.provenance.forEach((entry, index) => {
      expect(lines[index].textContent).toContain(`Run ${entry.role}`);
      expect(lines[index].textContent).toContain(`${entry.cells_done} of ${entry.cells_expected}`);
    });
    for (const entry of study.provenance) {
      const hash = screen.getByText(entry.git_commit.slice(0, 7));
      expect(hash.getAttribute("title")).toBe(entry.git_commit);
    }
  });
});

describe("FindingsSection", () => {
  it("shows all eight expectations, verbatim, with the tally the published report states", () => {
    // "Met: 7 · Failed: 1 · Inconclusive: 0", under Headline Findings in
    // docs/reports/2026-09-26-benchmark-study/benchmark-report.md.
    expect([study.expectations.n_met, study.expectations.n_failed, study.expectations.n_inconclusive]).toEqual([
      7, 1, 0,
    ]);
    const { container } = render(<FindingsSection expectations={study.expectations} />);
    const tally = screen.getByLabelText("Tally");
    expect(tally.textContent).toContain("Met7");
    expect(tally.textContent).toContain("Failed1");
    expect(tally.textContent).toContain("Inconclusive0");
    expect(study.expectations.rows).toHaveLength(8);
    expect(screen.getAllByRole("listitem")).toHaveLength(8);
    for (const row of study.expectations.rows) {
      expect(screen.getByText(row.id)).toBeTruthy();
      expect(screen.getByText(row.statement)).toBeTruthy();
      expect(screen.getByText(row.narrative)).toBeTruthy();
    }
    expect(container.querySelectorAll('[data-status="failed"]')).toHaveLength(1);
  });
});

describe("the tables' declared columns", () => {
  it("name exactly the committed table's keys, with each metric's _mcse and _n_used", () => {
    for (const [columns, table] of [
      [CATE_COLUMNS, study.cate.table],
      [P7_COLUMNS, study.p7.table],
    ] as const) {
      const declared = [
        ...columns.keys,
        ...columns.metrics.flatMap((metric) => [metric, `${metric}_mcse`, `${metric}_n_used`]),
      ].sort();
      expect(table.length).toBeGreaterThan(0);
      for (const row of table) expect(Object.keys(row).sort()).toEqual(declared);
    }
  });
});

describe("CateSection", () => {
  it("shows the note, the counts and a table of three rows", () => {
    render(<CateSection cate={study.cate} />);
    expect(screen.getByText(study.cate.note)).toBeTruthy();
    const rows = within(screen.getByRole("table")).getAllByRole("row");
    expect(rows).toHaveLength(1 + 3);
    expect(study.cate.table).toHaveLength(3);
    expect(within(rows[0]).getAllByRole("columnheader").map((cell) => cell.textContent)).toEqual([
      "method",
      "scenario",
      "cate_correlation",
      "cate_rmse",
      "cate_calibration_slope",
    ]);
  });

  it("shows the Monte Carlo error beside each metric and a dash where the report has none", () => {
    render(<CateSection cate={study.cate} />);
    const nullCorrelation = study.cate.table.find((row) => row.cate_correlation === null);
    expect(nullCorrelation).toBeDefined();
    const row = screen.getByRole("row", { name: new RegExp(nullCorrelation!.scenario) });
    // The metric is a dash, with no error beside it; the fit count says why: none fed it.
    expect(within(row).getAllByRole("cell")[1].textContent).toBe(`${DASH}0 of 50 fits`);
  });

  it("says the report has no such section, and draws no table, when the bundle holds none", () => {
    const bundle = { ...study, cate: null };
    render(<CateSection cate={bundle.cate} />);
    expect(screen.getByText("The report has no such section.")).toBeTruthy();
    expect(screen.queryByRole("table")).toBeNull();
    expect(screen.getByRole("heading", { level: 2 }).id).toBe("cate");
  });
});

describe("P7Section", () => {
  it("shows the note, the counts and every row's method and interval", () => {
    render(<P7Section p7={study.p7} />);
    expect(screen.getByText(study.p7.note)).toBeTruthy();
    expect(screen.getAllByRole("row")).toHaveLength(1 + study.p7.table.length);
    expect(screen.getAllByRole("columnheader").map((cell) => cell.textContent)).toEqual([
      "method",
      "interval",
      "bias",
      "rmse",
      "coverage",
    ]);
  });

  it("shows the no-section sentence for a bundle whose p7 is null", () => {
    const bundle = { ...study, p7: null };
    render(<P7Section p7={bundle.p7} />);
    expect(screen.getByText("The report has no such section.")).toBeTruthy();
    expect(screen.queryByRole("table")).toBeNull();
  });
});

describe("StudyTable", () => {
  const rows = [
    { method: "m1", scenario: "S1", a: null, a_mcse: null, a_n_used: 0 },
    { method: "m2", scenario: "S2", a: 0, a_mcse: 0, a_n_used: 50 },
    { method: "m3", scenario: "S3", a: 0.123456, a_mcse: 0.01234, a_n_used: 40 },
  ];
  const props = { caption: "Test", rows, keys: ["method", "scenario"], metrics: ["a"], replications: 50 };

  it("renders a null as the dash and never as 0, and a real zero as 0", () => {
    render(<StudyTable {...props} />);
    const cells = (name: RegExp) => within(screen.getByRole("row", { name })).getAllByRole("cell");
    const missing = cells(/m1/)[1].textContent ?? "";
    expect(missing.startsWith(DASH)).toBe(true);
    expect(missing).not.toContain("±");
    expect(missing.replace("0 of 50 fits", "")).toBe(DASH);
    expect(cells(/m2/)[1].textContent).toBe("0± 0");
  });

  it("shows a fit count only where it is below the replications", () => {
    render(<StudyTable {...props} />);
    expect(screen.queryByText(/50 of 50/)).toBeNull();
    expect(screen.getByText("40 of 50 fits")).toBeTruthy();
    expect(screen.getByText("0 of 50 fits")).toBeTruthy();
  });

  it("puts the key columns first, in the order declared", () => {
    render(<StudyTable {...props} keys={["scenario", "method"]} />);
    expect(screen.getAllByRole("columnheader").map((cell) => cell.textContent)).toEqual([
      "scenario",
      "method",
      "a",
    ]);
  });
});

describe("LimitationsSection", () => {
  it("shows each statement verbatim and no list of what is not covered when it is empty", () => {
    expect(study.limitations.not_covered).toEqual([]);
    expect(study.limitations.statements).toHaveLength(2);
    render(<LimitationsSection limitations={study.limitations} />);
    for (const statement of study.limitations.statements) {
      expect(screen.getByText(statement)).toBeTruthy();
    }
    expect(screen.queryByText("Not covered")).toBeNull();
  });

  it("lists what is not covered when the bundle has it", () => {
    render(<LimitationsSection limitations={{ not_covered: ["a gap"], statements: [] }} />);
    expect(screen.getByText("Not covered")).toBeTruthy();
    expect(screen.getByText("a gap")).toBeTruthy();
  });
});

describe("StudyUnavailable", () => {
  it("names the version found and the versions understood, and draws no section", () => {
    render(<StudyUnavailable failure={{ ok: false, reason: "version", found: 7, known: [1, 2] }} />);
    const message = screen.getByRole("status");
    expect(message.textContent).toContain("7");
    expect(message.textContent).toContain("1, 2");
    expect(screen.queryAllByRole("heading", { level: 2 })).toHaveLength(0);
  });

  it("lists the entries the vocabulary check refused, and draws no section", () => {
    render(
      <StudyUnavailable
        failure={{ ok: false, reason: "vocabulary", unknown: ['scenario P9: shape "staggered"'] }}
      />,
    );
    expect(screen.getByRole("status").textContent).toContain('scenario P9: shape "staggered"');
    expect(screen.queryAllByRole("heading", { level: 2 })).toHaveLength(0);
  });
});

describe("MethodsSection", () => {
  const base = defaultView(study);
  const href = (view: View) => viewHref(study, view);
  const methods = (view: Partial<View> = {}) => render(<MethodsSection study={study} view={{ ...base, ...view }} />);

  /** The `td` of an estimator's row under a column head (a scenario, or at the second level a size). */
  function cellAt(estimator: string, head: string): HTMLElement {
    const table = screen.getByRole("table");
    const headRow = within(table)
      .getAllByRole("row")
      .find((row) => within(row).queryByRole("columnheader", { name: "Estimator" }) !== null);
    const index = within(headRow!)
      .getAllByRole("columnheader")
      .slice(1)
      .findIndex((cell) => cell.textContent?.replace(/[\u25B2\u25BC]/g, "").trim() === head);
    expect(index).toBeGreaterThanOrEqual(0);
    const row = within(table)
      .getAllByRole("row")
      .find((candidate) => candidate.querySelector('th[scope="row"]')?.textContent === estimator);
    expect(row).toBeDefined();
    return row!.querySelectorAll("td")[index] as HTMLElement;
  }

  it("shows the default view's number, its Monte Carlo error and the fits behind it", () => {
    methods();
    const cell = cellAt("aipw", "A");
    expect(cell.textContent).toContain("0.001247");
    expect(cell.textContent).toContain("± 0.003131");
    expect(cell.getAttribute("title")).toBe("100 of 100 fits");
    expect(cell.getAttribute("data-kind")).toBe("value");
    expect(cell.style.getPropertyValue("--pos")).not.toBe("");
    expect(cell.getAttribute("data-scale")).toBe("diverging");
  });

  it("leaves an ineligible pair blank, with its reason and no place on the scale", () => {
    methods();
    const cell = cellAt("aipw", "P1");
    expect(cell.getAttribute("title")).toBe("Not eligible for this scenario");
    expect(cell.getAttribute("data-kind")).toBe("not_eligible");
    expect(cell.style.getPropertyValue("--pos")).toBe("");
    expect(cell.getAttribute("data-scale")).toBeNull();
    expect(cell.classList.contains("heat")).toBe(false);
    expect(cell.textContent).not.toMatch(/\d/);
  });

  it("says refused where every fit was refused, and gives the count", () => {
    methods();
    const cell = cellAt("aipw", "H");
    expect(cell.textContent).toBe("refused");
    expect(cell.getAttribute("title")).toBe("Every fit was refused (100)");
    expect(cell.style.getPropertyValue("--pos")).toBe("");
  });

  describe("each state of a cell", () => {
    const columns = matrixOf(study, { ...base, shape: "cross_section" }).columns.slice(0, 1);
    const states: [Cell, string, string][] = [
      [{ kind: "not_eligible" }, "", "Not eligible for this scenario"],
      [{ kind: "refused", nRefused: 7 }, "refused", "Every fit was refused (7)"],
      [{ kind: "not_applicable", nFitted: 9 }, DASH, "No fit feeds this metric (0 of 9)"],
      [{ kind: "undefined", nUsed: 4 }, "undefined", "The metric has no value here"],
      [{ kind: "absent" }, DASH, "No row in the study"],
    ];

    it.each(states)("shows %j as %j, titled %j, with no colour from the scale", (cell, text, title) => {
      const matrix: Matrix = { metric: "bias", columns, rows: [{ estimator: "aipw", family: "cross_sectional", cells: [cell] }] };
      const { container } = render(<MatrixHeatmap matrix={matrix} scale={{ kind: "diverging", centre: 0, halfWidth: 1, centreLabel: "0" }} level="scenario" />);
      const td = container.querySelector("tbody td") as HTMLElement;
      expect(td.getAttribute("title")).toBe(title);
      expect(td.textContent?.replace("not eligible", "")).toBe(text);
      expect(td.getAttribute("data-kind")).toBe(cell.kind);
      expect(td.classList.contains("heat")).toBe(false);
      expect(td.style.getPropertyValue("--pos")).toBe("");
      expect(td.textContent).not.toBe("0");
    });

    it("tells the five apart: no two share the same word and kind", () => {
      const seen = states.map(([cell, text]) => `${cell.kind}|${text}`);
      expect(new Set(seen).size).toBe(5);
    });

    it("shows a null Monte Carlo error as a dash and a fit count only where fewer fits fed it", () => {
      const matrix: Matrix = {
        metric: "bias",
        columns,
        rows: [
          { estimator: "aipw", family: "cross_sectional", cells: [{ kind: "value", value: 0.5, mcse: null, nUsed: 40, nFitted: 50 }] },
        ],
      };
      const { container } = render(<MatrixHeatmap matrix={matrix} scale={{ kind: "flat" }} level="scenario" />);
      const td = container.querySelector("tbody td") as HTMLElement;
      expect(td.textContent).toBe(`0.5± ${DASH}40 of 50 fits`);
      expect(td.getAttribute("title")).toBe("40 of 50 fits");
      expect(td.getAttribute("data-scale")).toBe("flat");
    });
  });

  it("heads each column with its size at the second level: units alone for a cross-section, units by periods for a panel", () => {
    methods({ level: "scenario_by_size" });
    const heads = screen.getAllByRole("columnheader").map((cell) => cell.textContent);
    expect(heads).toContain("500");
    expect(heads).toContain("150 × 7");
    expect(heads).toContain("18 × 24");
    expect(heads.filter((head) => /\.|undefined|null|NaN/.test(head ?? ""))).toEqual([]);
    expect(screen.getByRole("columnheader", { name: "P1" }).getAttribute("colspan")).toBe("2");
  });

  it("heads each column with its scenario at the first level", () => {
    methods();
    const heads = screen.getAllByRole("columnheader").map((cell) => cell.textContent);
    expect(heads).toEqual(["Estimator", ...study.scenarios.map((scenario) => scenario.name)]);
  });

  describe("the switches", () => {
    it("are six groups, each with exactly one current link", () => {
      methods();
      const groups = screen.getAllByRole("group");
      expect(groups.map((group) => group.getAttribute("aria-label"))).toEqual([
        "Metric family",
        "Metric",
        "Level",
        "Scenarios",
        "Estimators",
        "Display",
      ]);
      for (const group of groups) {
        expect(group.querySelectorAll('a[aria-current="true"]')).toHaveLength(1);
        expect(group.querySelectorAll("a[aria-current]")).toHaveLength(1);
      }
      const current = (name: string) =>
        within(screen.getByRole("group", { name })).getByRole("link", { current: true }).textContent;
      expect(current("Metric family")).toBe("accuracy");
      expect(current("Metric")).toBe("bias");
      expect(current("Level")).toBe("By scenario");
      expect(current("Scenarios")).toBe("All");
      expect(current("Estimators")).toBe("All");
      expect(current("Display")).toBe("Heatmap");
    });

    it("lead the family links to the view of each family, with that family's first metric", () => {
      methods();
      const group = screen.getByRole("group", { name: "Metric family" });
      const offered = offeredFamilies(study);
      expect(within(group).getAllByRole("link")).toHaveLength(offered.length);
      for (const entry of offered) {
        const link = within(group).getByRole("link", { name: entry.family });
        expect(link.getAttribute("href")).toBe(href({ ...base, family: entry.family, metric: entry.metrics[0] }));
      }
      expect(within(group).getByRole("link", { name: "coverage" }).getAttribute("href")).toBe(
        "/study?family=coverage&metric=coverage_analytic",
      );
    });

    it("offer the current family's metrics, and keep a link's view in the URL, not in script", () => {
      methods({ family: "coverage", metric: "coverage_bootstrap" });
      const group = screen.getByRole("group", { name: "Metric" });
      const names = within(group).getAllByRole("link").map((link) => link.textContent);
      expect(names).toEqual(study.family_metrics.coverage);
      expect(within(group).getByRole("link", { current: true }).textContent).toBe("coverage_bootstrap");
      expect(within(group).getByRole("link", { name: "mean_ci_width" }).getAttribute("href")).toBe(
        "/study?family=coverage&metric=mean_ci_width",
      );
    });

    it("drop the sort when the columns change, and keep it when they do not", () => {
      methods({ display: "table", sort: "A", direction: "desc" });
      const link = (group: string, name: string) =>
        within(screen.getByRole("group", { name })).getByRole("link", { name: group }).getAttribute("href");
      expect(link("By scenario and size", "Level")).toBe("/study?level=scenario_by_size&display=table");
      expect(link("Panel", "Scenarios")).toBe("/study?shape=panel&display=table");
      expect(link("Panel", "Estimators")).toBe("/study?estimators=panel&display=table&sort=A&dir=desc");
      expect(link("Heatmap", "Display")).toBe("/study?sort=A&dir=desc");
    });
  });

  describe("the table display", () => {
    const tableView = { display: "table" as const };
    const column = (estimator: string) => cellAt(estimator, "A");
    const order = () =>
      within(screen.getByRole("table"))
        .getAllByRole("row")
        .slice(1)
        .map((row) => row.querySelector('th[scope="row"]')?.textContent as string);

    it("shows the same matrix as text with no colour", () => {
      const { container } = methods(tableView);
      expect(order()).toEqual(study.estimators.map((estimator) => estimator.name));
      expect(column("aipw").textContent).toBe("0.001247± 0.003131100 of 100 fits");
      expect(container.querySelectorAll(".heat, [data-scale]")).toHaveLength(0);
      expect(container.querySelector("[style]")).toBeNull();
      expect(column("bsts").textContent).toBe("not eligible");
    });

    it.each([
      ["desc", -1],
      ["asc", 1],
    ] as const)("sorted by A %s puts the rows with a number first, in order, and the rest last", (direction, sign) => {
      methods({ ...tableView, sort: "A", direction });
      const cells = order().map((estimator) => column(estimator));
      const kinds = cells.map((cell) => cell.getAttribute("data-kind"));
      const firstOther = kinds.findIndex((kind) => kind !== "value");
      const valued = firstOther === -1 ? kinds.length : firstOther;
      expect(valued).toBeGreaterThan(0);
      expect(valued).toBeLessThan(kinds.length);
      expect(kinds.slice(valued).every((kind) => kind !== "value")).toBe(true);
      const values = cells.slice(0, valued).map((cell) => Number(cell.textContent?.split("±")[0]));
      expect(values).toEqual([...values].sort((a, b) => sign * (a - b)));
      expect(kinds[0]).toBe("value");
    });

    it("sorts by estimator name, and marks only the sorted column with aria-sort", () => {
      methods({ ...tableView, sort: "estimator", direction: "desc" });
      expect(order()).toEqual(study.estimators.map((estimator) => estimator.name).sort().reverse());
      const sorted = screen.getAllByRole("columnheader").filter((cell) => cell.hasAttribute("aria-sort"));
      expect(sorted).toHaveLength(1);
      expect(sorted[0].getAttribute("aria-sort")).toBe("descending");
    });

    it("heads each column with a link that sets the sort and flips the direction", () => {
      methods({ ...tableView, sort: "A", direction: "asc" });
      const link = (name: string) =>
        within(screen.getByRole("table")).getByRole("link", { name }).getAttribute("href");
      expect(link("A")).toBe("/study?display=table&sort=A&dir=desc");
      expect(link("B")).toBe("/study?display=table&sort=B");
      expect(screen.getByRole("columnheader", { name: /^A/ }).getAttribute("aria-sort")).toBe("ascending");
    });

    it("sorts at the second level by a size's own column", () => {
      methods({ ...tableView, level: "scenario_by_size", sort: "A:500:-", direction: "desc" });
      const heads = screen.getAllByRole("columnheader").filter((cell) => cell.hasAttribute("aria-sort"));
      expect(heads).toHaveLength(1);
      expect(heads[0].textContent).toContain("500");
    });
  });

  describe("the notes and the definition", () => {
    it("show the coverage note for coverage and not for accuracy", () => {
      const { unmount } = methods({ family: "coverage", metric: "coverage_analytic" });
      expect(screen.getByText(study.report_tables.coverage_note)).toBeTruthy();
      unmount();
      methods();
      expect(screen.queryByText(study.report_tables.coverage_note)).toBeNull();
    });

    it("show the exploratory note at the second level only", () => {
      const { unmount } = methods({ level: "scenario_by_size" });
      expect(screen.getByText(study.report_tables.exploratory_note)).toBeTruthy();
      unmount();
      methods();
      expect(screen.queryByText(study.report_tables.exploratory_note)).toBeNull();
    });

    it("show the metric's definition, verbatim, and nothing for a metric without one", () => {
      const { unmount } = methods();
      expect(screen.getByText(study.metric_definitions.bias)).toBeTruthy();
      unmount();
      const bare = { ...study, metric_definitions: {} };
      render(<MethodsSection study={bare} view={base} />);
      expect(screen.queryByText(study.metric_definitions.bias)).toBeNull();
      expect(document.querySelector(".engine")).toBeNull();
    });
  });

  it("never prints NaN or null, in any family at either level", () => {
    let runs = 0;
    for (const entry of offeredFamilies(study)) {
      for (const level of entry.levels) {
        for (const display of ["heatmap", "table"] as const) {
          for (const metric of entry.metrics) {
            const { container, unmount } = methods({ family: entry.family, metric, level, display });
            expect(container.textContent).not.toMatch(/NaN|null|Infinity/);
            expect(container.querySelectorAll("tbody td").length).toBeGreaterThan(0);
            unmount();
          }
        }
        runs += 1;
      }
    }
    expect(runs).toBe(12);
  });

  it("colours every value in every metric between the scale's own ends: positions stay inside -1 to 1", () => {
    for (const entry of offeredFamilies(study)) {
      for (const metric of entry.metrics) {
        const { container, unmount } = methods({ family: entry.family, metric });
        for (const cell of container.querySelectorAll<HTMLElement>("td.heat")) {
          const pos = Number(cell.style.getPropertyValue("--pos"));
          expect(Number.isFinite(pos)).toBe(true);
          expect(Math.abs(pos)).toBeLessThanOrEqual(1);
        }
        unmount();
      }
    }
  });
});

describe("ScaleLegend", () => {
  it("states a diverging scale about its centre, as the nominal level for coverage", () => {
    const { container } = render(
      <ScaleLegend scale={{ kind: "diverging", centre: 0.95, halfWidth: 0.1, centreLabel: "0.95" }} metric="coverage_analytic" />,
    );
    expect(container.textContent).toContain("diverging about 0.95, the nominal level");
    expect(container.querySelectorAll(".scale-step.heat")).toHaveLength(5);
  });

  it("states a diverging scale without a nominal level for a bias", () => {
    const { container } = render(
      <ScaleLegend scale={{ kind: "diverging", centre: 0, halfWidth: 0.2, centreLabel: "0" }} metric="bias" />,
    );
    expect(container.textContent).toContain("diverging about 0");
    expect(container.textContent).not.toContain("nominal");
  });

  it("states a sequential scale from its low to its high", () => {
    const { container } = render(<ScaleLegend scale={{ kind: "sequential", low: 0.012, high: 0.31 }} metric="rmse" />);
    expect(container.textContent).toContain("low to high: 0.012 to 0.31");
    expect(container.querySelectorAll(".scale-step.heat")).toHaveLength(5);
  });

  it("says nothing, and draws no strip, when the scale is flat", () => {
    const { container } = render(<ScaleLegend scale={{ kind: "flat" }} metric="bias" />);
    expect(container.innerHTML).toBe("");
  });

  it("is built from the same scale the matrix colours with", () => {
    const view = defaultView(study);
    const matrix = matrixOf(study, view);
    const values = matrix.rows.flatMap((row) => row.cells.flatMap((cell) => (cell.kind === "value" ? [cell.value] : [])));
    expect(scaleFor(view.metric, values).kind).toBe("diverging");
  });
});

describe("the /study page", () => {
  it("reads the view from the query string and shows it", async () => {
    render(await StudyPage({ searchParams: Promise.resolve({ family: "coverage", display: "table", level: "scenario_by_size" }) }));
    expect(within(screen.getByRole("group", { name: "Metric family" })).getByRole("link", { current: true }).textContent).toBe("coverage");
    expect(within(screen.getByRole("group", { name: "Display" })).getByRole("link", { current: true }).textContent).toBe("Table");
    expect(screen.getByText(study.report_tables.coverage_note)).toBeTruthy();
    expect(screen.getByText(study.report_tables.exploratory_note)).toBeTruthy();
  });

  it("replaces a value it does not recognise with the default, and echoes none of it", async () => {
    const { container } = render(
      await StudyPage({ searchParams: Promise.resolve({ family: "<script>x</script>", sort: "nope", display: ["table", "heatmap"] }) }),
    );
    expect(container.textContent).not.toContain("<script>");
    expect(within(screen.getByRole("group", { name: "Metric family" })).getByRole("link", { current: true }).textContent).toBe("accuracy");
    expect(within(screen.getByRole("group", { name: "Display" })).getByRole("link", { current: true }).textContent).toBe("Table");
  });
});
