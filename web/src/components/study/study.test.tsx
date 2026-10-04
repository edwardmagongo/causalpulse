import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));

import StudyPage from "@/app/(app)/study/page";
import { DASH, formatNumber } from "@/lib/format";
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
import { MatrixTable } from "./MatrixTable";
import { ScaleLegend } from "./ScaleLegend";
import { StudyTable } from "./StudyTable";

afterEach(cleanup);

/** What a screen reader reads of an element: its text without the `aria-hidden` subtrees. */
function withoutHidden(element: Element): string {
  const copy = element.cloneNode(true) as Element;
  for (const hidden of copy.querySelectorAll('[aria-hidden="true"]')) hidden.remove();
  return copy.textContent ?? "";
}

/** What a sighted reader sees of an element: its text without the screen-reader-only spans. */
function visibleText(element: Element): string {
  const copy = element.cloneNode(true) as Element;
  for (const hidden of copy.querySelectorAll(".sr-only")) hidden.remove();
  return copy.textContent ?? "";
}

const loaded = loadStudy();
if (!loaded.ok) throw new Error("the committed bundle must load");
const study: StudyBundle = loaded.study;
// The committed study has both sections; a null here would be the writer dropping one.
const cate = study.cate;
const p7 = study.p7;
if (cate === null || p7 === null) throw new Error("the committed bundle must carry its CATE and P7 sections");

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
    render(<OverviewSection overview={study.overview} provenance={study.provenance} />);
    const facts: [string, string][] = [
      ["Estimators", `${study.overview.n_estimators}`],
      ["Scenarios", `${study.overview.n_scenarios}`],
      ["Cells", `${study.overview.n_cells}`],
      ["Replications", `${study.overview.replications}`],
      ["Coverage Monte Carlo error", formatNumber(study.overview.coverage_mcse)],
    ];
    for (const [label, value] of facts) {
      expect(screen.getByText(label).nextElementSibling?.textContent).toBe(value);
    }
    // The per-label assertions above are what catch a swap; this one is what lets them see it, since two equal
    // values would pass either way. It rests on the committed study's five figures differing: if a regenerated
    // bundle makes two coincide, choose facts that differ rather than dropping the check.
    expect(new Set(facts.map(([, value]) => value)).size).toBe(facts.length);
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
      [CATE_COLUMNS, cate.table],
      [P7_COLUMNS, p7.table],
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
    render(<CateSection cate={cate} />);
    expect(screen.getByText(cate.note)).toBeTruthy();
    const rows = within(screen.getByRole("table")).getAllByRole("row");
    expect(rows).toHaveLength(1 + 3);
    expect(cate.table).toHaveLength(3);
    expect(within(rows[0]).getAllByRole("columnheader").map((cell) => cell.textContent)).toEqual([
      "method",
      "scenario",
      "cate_correlation",
      "cate_rmse",
      "cate_calibration_slope",
    ]);
  });

  it("shows the Monte Carlo error beside each metric and a dash where the report has none", () => {
    render(<CateSection cate={cate} />);
    const nullCorrelation = cate.table.find((row) => row.cate_correlation === null);
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
    render(<P7Section p7={p7} />);
    expect(screen.getByText(p7.note)).toBeTruthy();
    expect(screen.getAllByRole("row")).toHaveLength(1 + p7.table.length);
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
    expect(visibleText(cell)).toBe("refused");
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
      const { container } = render(<MatrixHeatmap matrix={matrix} scale={scaleFor("bias", [-1, 1])} level="scenario" />);
      const td = container.querySelector("tbody td") as HTMLElement;
      expect(td.getAttribute("title")).toBe(title);
      expect(visibleText(td)).toBe(text);
      expect(td.getAttribute("data-kind")).toBe(cell.kind);
      expect(td.classList.contains("heat")).toBe(false);
      expect(td.style.getPropertyValue("--pos")).toBe("");
      expect(td.textContent).not.toBe("0");
    });

    it.each([
      ["a heatmap", (matrix: Matrix) => render(<MatrixHeatmap matrix={matrix} scale={{ kind: "flat" }} level="scenario" />)],
      [
        "a table",
        (matrix: Matrix) => render(<MatrixTable matrix={matrix} view={{ ...base, display: "table" }} href={href} />),
      ],
    ] as const)("puts each neutral kind's reason in the cell's text of %s, for a reader who cannot see the title", (name, draw) => {
      // The table names an ineligible pair; the heatmap leaves it a hatch.
      const visibleMark = (cell: Cell) =>
        cell.kind === "not_eligible" && name === "a table" ? "not eligible" : states.find(([state]) => state.kind === cell.kind)![1];
      for (const [cell, , reason] of states) {
        const matrix: Matrix = { metric: "bias", columns, rows: [{ estimator: "aipw", family: "cross_sectional", cells: [cell] }] };
        const { container, unmount } = draw(matrix);
        const td = container.querySelector("tbody td") as HTMLElement;
        expect(td.textContent).toContain(reason);
        expect(td.getAttribute("title")).toBe(reason);
        const hidden = td.querySelector(".sr-only");
        expect(hidden?.textContent).toBe(reason);
        // The visible mark is hidden from a screen reader, so it is not read as well as the reason.
        const marks = td.querySelectorAll('[aria-hidden="true"]');
        const mark = marks[0];
        if (visibleMark(cell) === "") {
          // An ineligible pair in the heatmap is a hatch: no mark, nothing hidden, only the reason.
          expect(marks).toHaveLength(0);
          expect(td.querySelectorAll("span")).toHaveLength(1);
        } else {
          expect(marks).toHaveLength(1);
          expect(mark.classList.contains("sr-only")).toBe(false);
          expect(mark.textContent).toBe(visibleMark(cell));
        }
        // What a screen reader reads: the cell's text without the hidden subtrees is exactly the reason.
        expect(withoutHidden(td)).toBe(reason);
        unmount();
      }
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

  it.each(["heatmap", "table"] as const)(
    "has one header over the estimator column at the second level of the %s, the corner above it a plain cell",
    (display) => {
      methods({ level: "scenario_by_size", display });
      expect(screen.getAllByRole("columnheader", { name: /Estimator/ })).toHaveLength(1);
      expect(screen.queryByRole("columnheader", { name: "Scenario" })).toBeNull();
      const corner = within(screen.getAllByRole("row")[0]).getAllByRole("cell")[0];
      expect(corner.tagName).toBe("TD");
      expect(corner.textContent).toBe("Scenario");
    },
  );

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
      expect(visibleText(column("bsts"))).toBe("not eligible");
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
  const numbersIn = (text: string) => (text.match(/\d+(?:\.\d+)?/g) ?? []).map(Number);

  it("reads a capped coverage scale as its clipped ends and the nominal centre, and prints no share above 1", () => {
    const { container } = render(<ScaleLegend scale={scaleFor("coverage_analytic", [0, 0.93, 0.95, 0.97])} />);
    const text = container.textContent ?? "";
    expect(text).toContain("diverging about 0.95, the nominal level");
    expect(text).toContain("≤ 0.8");
    expect(text).not.toContain("≥");
    expect(container.querySelectorAll(".value")).toHaveLength(2);
    expect(Array.from(container.querySelectorAll(".value"), (end) => end.textContent)).toEqual(["≤ 0.8", "1"]);
    expect(Math.max(...numbersIn(text))).toBeLessThanOrEqual(1);
    expect(container.querySelectorAll(".scale-step.heat")).toHaveLength(5);
  });

  it("draws the strip only over colours a value can take: its last step is the colour of a coverage of 1", () => {
    const scale = scaleFor("coverage_analytic", [0, 0.93, 0.95, 0.97]);
    const { container } = render(<ScaleLegend scale={scale} />);
    const steps = Array.from(container.querySelectorAll<HTMLElement>(".scale-step"), (step) => step.style.getPropertyValue("--pos"));
    expect(steps).toHaveLength(5);
    expect(steps[0]).toBe("-1.0000");
    expect(steps).toContain("0.0000");
    // A cell holding the value the legend labels at its high end.
    const columns = matrixOf(study, { ...defaultView(study), shape: "cross_section" }).columns.slice(0, 1);
    const cell: Cell = { kind: "value", value: 1, mcse: 0, nUsed: 100, nFitted: 100 };
    const matrix: Matrix = { metric: "coverage_analytic", columns, rows: [{ estimator: "aipw", family: "cross_sectional", cells: [cell] }] };
    const heat = render(<MatrixHeatmap matrix={matrix} scale={scale} level="scenario" />);
    const td = heat.container.querySelector("tbody td") as HTMLElement;
    expect(steps[steps.length - 1]).toBe(td.style.getPropertyValue("--pos"));
    expect(Number(steps[steps.length - 1])).toBeCloseTo(1 / 3, 3);
  });

  it("starts a false-positive-rate strip at the colour of a rate of 0, not at the full blue", () => {
    const { container } = render(<ScaleLegend scale={scaleFor("false_positive_rate", [0.02, 0.05, 0.3])} />);
    const first = container.querySelector<HTMLElement>(".scale-step")!;
    expect(Number(first.style.getPropertyValue("--pos"))).toBeCloseTo(-1 / 3, 3);
  });

  it("prints an end as a plain number when no value in view lies beyond it", () => {
    const { container } = render(<ScaleLegend scale={scaleFor("bias", [-0.2, 0.1, 0.2])} />);
    expect(container.textContent).toContain("diverging about 0");
    expect(container.textContent).not.toContain("nominal");
    expect(Array.from(container.querySelectorAll(".value"), (end) => end.textContent)).toEqual(["-0.2", "0.2"]);
  });

  it("takes every word from the scale: a centre label it was not built for is printed as given", () => {
    const { container } = render(
      <ScaleLegend
        scale={{ kind: "diverging", centre: 3, halfWidth: 2, centreLabel: "3, a label of its own", lowEnd: 1, highEnd: 5, lowBeyond: false, highBeyond: true }}
      />,
    );
    expect(container.textContent).toContain("diverging about 3, a label of its own");
    expect(Array.from(container.querySelectorAll(".value"), (end) => end.textContent)).toEqual(["1", "≥ 5"]);
  });

  it("states a sequential scale from its low to its high", () => {
    const { container } = render(<ScaleLegend scale={{ kind: "sequential", low: 0.012, high: 0.31 }} />);
    expect(container.textContent).toContain("low to high: 0.012 to 0.31");
    expect(container.querySelectorAll(".scale-step.heat")).toHaveLength(5);
  });

  it("says nothing, and draws no strip, when the scale is flat", () => {
    const { container } = render(<ScaleLegend scale={{ kind: "flat" }} />);
    expect(container.innerHTML).toBe("");
  });

  it("is built from the same scale the matrix colours with, and says the default view's centre", () => {
    const view = defaultView(study);
    const matrix = matrixOf(study, view);
    const values = matrix.rows.flatMap((row) => row.cells.flatMap((cell) => (cell.kind === "value" ? [cell.value] : [])));
    const scale = scaleFor(view.metric, values);
    expect(scale.kind).toBe("diverging");
    const { container } = render(<ScaleLegend scale={scale} />);
    expect(container.textContent).toContain("diverging about 0");
  });

  it("reads coverage in the study's own view as the brief does, with the cap in the cells", () => {
    const view = { ...defaultView(study), family: "coverage", metric: "coverage_analytic" };
    const matrix = matrixOf(study, view);
    const values = matrix.rows.flatMap((row) => row.cells.flatMap((cell) => (cell.kind === "value" ? [cell.value] : [])));
    const scale = scaleFor(view.metric, values);
    expect(scale.kind === "diverging" && scale.halfWidth).toBeCloseTo(0.15, 12);
    const { container } = render(<ScaleLegend scale={scale} />);
    expect(Array.from(container.querySelectorAll(".value"), (end) => end.textContent)).toEqual(["≤ 0.8", "1"]);
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
