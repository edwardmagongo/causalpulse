import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import AppError from "@/app/(app)/error";
import AnalysesPage from "@/app/(app)/page";
import { listAnalyses } from "@/lib/api/client";
import type { AnalysisDetail, AnalysisPage, Estimate } from "@/lib/api/types";
import { formatInterval, formatLevel } from "@/lib/format";
import { panels } from "@/lib/plot";
import lalonde from "@/test/fixtures/lalonde.analysis.json";
import analyses from "@/test/fixtures/analyses.json";
import lalondeSummary from "@/test/fixtures/lalonde.summary.json";
import partial from "@/test/fixtures/partial.analysis.json";
import prop99 from "@/test/fixtures/prop99.analysis.json";
import prop99Summary from "@/test/fixtures/prop99.summary.json";
import refused from "@/test/fixtures/refused.analysis.json";

import { ComparisonList } from "./ComparisonList";
import { ErrorNotice } from "./ErrorNotice";
import { EstimatesTable } from "./EstimatesTable";
import { IntervalPlot } from "./IntervalPlot";
import { RoutingTable } from "./RoutingTable";
import { StatusHeader } from "./StatusHeader";

// The list is a page that reads through the API client; only that read is replaced.
vi.mock("server-only", () => ({}));
vi.mock("@/lib/api/client", () => ({ listAnalyses: vi.fn() }));

afterEach(cleanup);

describe("EstimatesTable", () => {
  it("shows each method's band and links its report", () => {
    render(<EstimatesTable analysisId="a1" estimates={lalonde.estimates as Estimate[]} />);
    const row = screen.getByRole("row", { name: /lin_regression/ });
    expect(within(row).getByText("MODERATE")).toBeTruthy();
    expect(within(row).getByRole("link", { name: "Evidence report" }).getAttribute("href")).toBe(
      "/analyses/a1/estimates/lin_regression",
    );
    expect(screen.getAllByRole("link", { name: "Evidence report" })).toHaveLength(2);
  });

  it("labels an interval by its own basis and never calls one a confidence interval", () => {
    const { container } = render(
      <EstimatesTable analysisId="a1" estimates={prop99.estimates as Estimate[]} />,
    );
    expect(screen.getByText("95% permutation interval")).toBeTruthy();
    expect(container.textContent?.toLowerCase()).not.toContain("confidence");
  });

  it("labels the bootstrap interval with its own level", () => {
    render(<EstimatesTable analysisId="a1" estimates={lalonde.estimates as Estimate[]} />);
    for (const estimate of lalonde.estimates) {
      const { summary } = estimate;
      const level = summary.bootstrap_ci_level;
      expect(typeof level).toBe("number");
      const cell = within(screen.getByRole("row", { name: new RegExp(estimate.method) })).getAllByRole(
        "cell",
      )[5];
      const interval = formatInterval(summary.bootstrap_ci_lower, summary.bootstrap_ci_upper);
      expect(cell.textContent).toContain(interval);
      expect(within(cell).getByText(`${formatLevel(level)} bootstrap interval`)).toBeTruthy();
    }
  });

  it.each([
    ["absent", (summary: Record<string, unknown>) => void delete summary.bootstrap_ci_level],
    // What the API sends for such an analysis: the model's default, serialised.
    ["null", (summary: Record<string, unknown>) => void (summary.bootstrap_ci_level = null)],
  ])("shows the bootstrap interval alone when its level was not stored (%s)", (_case, strip) => {
    // An analysis stored before the level was kept.
    const stored = lalonde.estimates.map((estimate) => {
      const summary: Record<string, unknown> = { ...estimate.summary };
      strip(summary);
      return { ...estimate, summary };
    }) as Estimate[];
    render(<EstimatesTable analysisId="a1" estimates={stored} />);
    const first = lalonde.estimates[0].summary;
    const cell = within(screen.getByRole("row", { name: /difference_in_means/ })).getAllByRole("cell")[5];
    expect(cell.textContent).toBe(formatInterval(first.bootstrap_ci_lower, first.bootstrap_ci_upper));
    expect(screen.queryByText(/bootstrap interval/)).toBeNull();
  });

  it("shows a dash and no label when the level is stored but the interval is absent", () => {
    // The API does not write this combination (a level comes only with an
    // interval), so this guards the component's own condition: a label belongs
    // to an interval that is shown, not to a level that happens to be present.
    const [first, ...rest] = lalonde.estimates;
    const summary = {
      ...first.summary,
      bootstrap_ci_level: 0.95,
      bootstrap_ci_lower: null,
      bootstrap_ci_upper: null,
    };
    render(<EstimatesTable analysisId="a1" estimates={[{ ...first, summary }, ...rest] as Estimate[]} />);
    const cell = within(screen.getByRole("row", { name: /difference_in_means/ })).getAllByRole("cell")[5];
    expect(cell.textContent).toBe("—");
    expect(within(cell).queryByText(/bootstrap interval/)).toBeNull();
  });

  it("shows a dash and no label when there is no bootstrap interval", () => {
    render(<EstimatesTable analysisId="a1" estimates={prop99.estimates as Estimate[]} />);
    expect(prop99.estimates[0].summary.bootstrap_ci_lower).toBeNull();
    const cell = within(screen.getByRole("row", { name: /synthetic_control/ })).getAllByRole("cell")[5];
    expect(cell.textContent).toBe("—");
    expect(screen.queryByText(/bootstrap interval/)).toBeNull();
  });

  it("renders an absent standard error as a dash, not as zero", () => {
    render(<EstimatesTable analysisId="a1" estimates={prop99.estimates as Estimate[]} />);
    const cells = within(screen.getByRole("row", { name: /synthetic_control/ })).getAllByRole("cell");
    // Columns after the method header: estimand, scale, point, standard error, ...
    expect(prop99.estimates[0].summary.se).toBeNull();
    expect(cells[3].textContent).toBe("—");
    expect(cells[2].textContent).toBe("-19.51");
  });

  it("gives a failed estimator a row with its error and no report link", () => {
    // Derived from a real response: the API stores a failure as a row with
    // only `method` and `error` set (`cp_worker.jobs.store_outcome`).
    const failed: Estimate = { method: "ipw", band: null, summary: null, error: "boom: no overlap" };
    render(
      <EstimatesTable analysisId="a1" estimates={[lalonde.estimates[0] as Estimate, failed]} />,
    );
    const row = screen.getByRole("row", { name: /ipw/ });
    expect(within(row).getByText("boom: no overlap")).toBeTruthy();
    expect(within(row).queryByRole("link")).toBeNull();
    expect(screen.getAllByRole("link", { name: "Evidence report" })).toHaveLength(1);
  });

  it("shows a recorded failed estimate as its method and the engine's error, with no numbers", () => {
    const failed = partial.estimates.find((estimate) => estimate.error !== null)!;
    expect(failed.method).toBe("lin_regression");
    expect(failed.summary).toBeNull();
    render(<EstimatesTable analysisId={partial.id} estimates={partial.estimates as Estimate[]} />);
    const row = screen.getByRole("row", { name: /lin_regression/ });
    const cells = within(row).getAllByRole("cell");
    expect(cells).toHaveLength(1);
    expect(cells[0].textContent).toBe(failed.error);
    // Nothing else in the row: no point, interval, band or runtime.
    expect(row.textContent).toBe(`${failed.method}${failed.error}`);
    expect(within(row).queryByRole("link")).toBeNull();
    // The estimator that succeeded in the same analysis keeps its numbers and its report.
    const succeeded = screen.getByRole("row", { name: /difference_in_means/ });
    expect(within(succeeded).getAllByRole("cell")).toHaveLength(11);
    expect(within(succeeded).getByRole("link", { name: "Evidence report" }).getAttribute("href")).toBe(
      `/analyses/${partial.id}/estimates/difference_in_means`,
    );
  });
});

describe("RoutingTable", () => {
  it("lists every estimator the router considered, eligible ones first, reasons verbatim", () => {
    render(<RoutingTable routing={prop99.routing} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows).toHaveLength(12);
    const eligible = prop99.routing.filter((d) => d.eligible).length;
    expect(eligible).toBe(3);
    rows.slice(0, eligible).forEach((row) => expect(within(row).getByText("eligible")).toBeTruthy());
    rows.slice(eligible).forEach((row) => expect(within(row).getByText("not eligible")).toBeTruthy());
    const refusedOne = prop99.routing.find((d) => !d.eligible)!;
    expect(screen.getAllByText(refusedOne.reasons[0]).length).toBeGreaterThan(0);
  });

  it("renders nothing for an analysis the router refused outright", () => {
    const { container } = render(<RoutingTable routing={refused.routing} />);
    expect(refused.routing).toEqual([]);
    expect(container.innerHTML).toBe("");
  });
});

describe("ComparisonList", () => {
  it("shows a compared group's verdict in the API's words", () => {
    render(<ComparisonList comparisons={lalondeSummary.comparisons} />);
    expect(screen.getByText(lalondeSummary.comparisons[0].verdict)).toBeTruthy();
    expect(screen.getByText("consistent")).toBeTruthy();
    expect(screen.getByText("-8333 to -3597")).toBeTruthy();
  });

  it("shows why a single method was not compared, and no empty verdict", () => {
    render(<ComparisonList comparisons={prop99Summary.comparisons} />);
    expect(screen.getByText(prop99Summary.comparisons[0].not_compared_reason)).toBeTruthy();
    expect(screen.queryByText("Dispersion")).toBeNull();
  });
});

describe("IntervalPlot", () => {
  it("draws one figure per comparison group and one mark per method", () => {
    const drawn = panels(
      [...lalonde.estimates, ...prop99.estimates] as Estimate[],
      [...lalondeSummary.comparisons, ...prop99Summary.comparisons],
    );
    const { container } = render(<IntervalPlot panels={drawn} />);
    expect(screen.getAllByRole("img")).toHaveLength(2);
    expect(container.querySelectorAll("[data-method]")).toHaveLength(3);
    expect(container.querySelector('[data-method="synthetic_control"] title')?.textContent).toContain(
      "95% permutation interval",
    );
  });

  it("labels each interval with its basis in visible text, not only in the hover title", () => {
    const drawn = panels(
      [...lalonde.estimates, ...prop99.estimates] as Estimate[],
      [...lalondeSummary.comparisons, ...prop99Summary.comparisons],
    );
    const { container } = render(<IntervalPlot panels={drawn} />);
    // Select the <text> element: container.textContent also holds the <title> text.
    const synthetic = container.querySelector('[data-method="synthetic_control"] text[data-basis]');
    expect(synthetic?.textContent).toBe("95% permutation interval");
    expect(synthetic?.getAttribute("data-basis")).toBe("permutation");
    for (const estimate of lalonde.estimates) {
      const label = container.querySelector(`[data-method="${estimate.method}"] text[data-basis]`);
      expect(label?.textContent).toBe("95% analytic interval");
      expect(label?.getAttribute("data-basis")).toBe("analytic");
    }
    expect(container.textContent?.toLowerCase()).not.toContain("confidence interval");
  });

  it("draws the zero line only on an axis that crosses zero", () => {
    const drawn = panels(prop99.estimates as Estimate[], prop99Summary.comparisons);
    const { container } = render(<IntervalPlot panels={drawn} />);
    const crosses = drawn[0].domain[0] < 0 && drawn[0].domain[1] > 0;
    expect(container.querySelectorAll("[data-zero]")).toHaveLength(crosses ? 1 : 0);
  });
});

describe("StatusHeader", () => {
  it("shows a refused analysis's stored error verbatim", () => {
    render(<StatusHeader analysis={refused as AnalysisDetail} filename="lalonde.csv" />);
    expect(screen.getByRole("alert").textContent).toBe(refused.error);
    expect(screen.getByText("failed")).toBeTruthy();
  });

  it("flags a recorded partial analysis, which succeeded and has no alert", () => {
    expect(partial.partial).toBe(true);
    const { container } = render(
      <StatusHeader analysis={partial as AnalysisDetail} filename={partial.dataset_filename} />,
    );
    expect(screen.getByRole("heading", { name: "Analysis of lalonde-inf.csv" })).toBeTruthy();
    expect(screen.getByText("succeeded")).toBeTruthy();
    expect(container.querySelector("[data-partial]")?.textContent?.trim()).toBe(
      "partial: some estimators failed",
    );
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("shows no partial mark and no alert for a clean analysis", () => {
    expect(lalonde.partial).toBe(false);
    const { container } = render(
      <StatusHeader analysis={lalonde as AnalysisDetail} filename={lalonde.dataset_filename} />,
    );
    expect(container.querySelector("[data-partial]")).toBeNull();
    expect(screen.queryByText(/partial/)).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

describe("the analyses list", () => {
  it("marks the recorded partial analysis's row, and no clean row", async () => {
    vi.mocked(listAnalyses).mockResolvedValue(analyses as AnalysisPage);
    render(await AnalysesPage({ searchParams: Promise.resolve({}) }));
    const rows = screen.getAllByRole("row").slice(1);
    expect(analyses.items.map((item) => item.partial)).toEqual([true, false, false, false]);
    expect(rows).toHaveLength(analyses.items.length);
    analyses.items.forEach((item, index) => {
      expect(within(rows[index]).getByRole("rowheader").textContent).toBe(item.dataset_filename);
      // The first cell after the dataset's row header is the status.
      expect(within(rows[index]).getAllByRole("cell")[0].textContent).toBe(
        item.partial ? `${item.status} (partial)` : item.status,
      );
    });
    expect(analyses.items[0].id).toBe(partial.id);
  });
});

describe("ErrorNotice", () => {
  it("says what happened in the dashboard's own two sentences", () => {
    render(<ErrorNotice digest={undefined} retry={() => {}} />);
    const notice = screen.getByRole("alert");
    expect(within(notice).getByRole("heading", { name: "Something went wrong on this page." })).toBeTruthy();
    expect(within(notice).getByText("Nothing was changed by this. You can try again.")).toBeTruthy();
  });

  it("calls retry once per click", () => {
    const retry = vi.fn();
    render(<ErrorNotice digest={undefined} retry={retry} />);
    const button = screen.getByRole("button", { name: "Try again" });
    fireEvent.click(button);
    expect(retry).toHaveBeenCalledTimes(1);
    fireEvent.click(button);
    expect(retry).toHaveBeenCalledTimes(2);
  });

  it("shows the digest when there is one", () => {
    const { container } = render(<ErrorNotice digest="1234567890" retry={() => {}} />);
    expect(screen.getByText("Reference: 1234567890")).toBeTruthy();
    expect(container.textContent).toBe(
      "Something went wrong on this page.Nothing was changed by this. You can try again.Try againReference: 1234567890",
    );
  });

  it.each([
    ["undefined", undefined],
    ["empty", ""],
  ])("shows no reference line when the digest is %s", (_case, digest) => {
    const { container } = render(<ErrorNotice digest={digest} retry={() => {}} />);
    expect(container.textContent).not.toContain("Reference");
    expect(container.textContent).toBe(
      "Something went wrong on this page.Nothing was changed by this. You can try again.Try again",
    );
  });

  it("never shows the error's own message", () => {
    const retry = vi.fn();
    const error: Error & { digest?: string } = new Error("secret detail");
    error.digest = "abc123";
    const { container } = render(<AppError error={error} retry={retry} />);
    expect(container.textContent).not.toContain("secret detail");
    expect(screen.getByText("Reference: abc123")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(retry).toHaveBeenCalledTimes(1);
  });
});
