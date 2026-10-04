import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));

import { DASH } from "@/lib/format";
import { loadStudy } from "@/lib/study/bundle";
import type { StudyBundle } from "@/lib/study/types";

import {
  CATE_COLUMNS,
  CateSection,
  FindingsSection,
  LimitationsSection,
  OverviewSection,
  P7Section,
  P7_COLUMNS,
  StudyReport,
  StudyUnavailable,
} from "./StudySections";
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
