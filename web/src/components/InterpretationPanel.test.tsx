import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { AnalysisDetail, InterpretationDetail } from "@/lib/api/types";
import { DASH, formatTime } from "@/lib/format";
import interpretations from "@/test/fixtures/interpretations.json";

const ask = vi.fn();
const refresh = vi.fn();
vi.mock("@/app/(app)/analyses/[id]/actions", () => ({ askForInterpretation: (id: string) => ask(id) }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ refresh }) }));

import { InterpretationPanel } from "./InterpretationPanel";

const BASE: InterpretationDetail = {
  id: "7a9d3c52-1f0e-4b8a-9c6d-2e5f4a3b1c0d",
  status: "queued",
  model: "claude-opus-5-5",
  returned_model: null,
  prompt_version: null,
  interpreter_version: null,
  out_of_date: false,
  created_at: "2026-10-08T10:00:00+00:00",
  started_at: null,
  finished_at: null,
  sections: null,
  limits: null,
  notice: null,
  attempts: null,
  reason: null,
  error: null,
};

const VERIFIED: InterpretationDetail = {
  ...BASE,
  status: "verified",
  returned_model: "claude-opus-5-5-20261001",
  prompt_version: "abc123def456",
  interpreter_version: "35d8275efecb",
  out_of_date: false,
  sections: [
    { name: "summary", heading: "Summary", statements: [] },
    { name: "estimates", heading: "Estimates", statements: ["The estimate for ipw is 0.07."] },
  ],
  limits: "The validity band reflects the checks it could run.",
  notice: "A verified interpretation means: every value shown is the system's own.",
};

function panel(props: Partial<Parameters<typeof InterpretationPanel>[0]> = {}) {
  return render(
    <InterpretationPanel
      analysisId="a1"
      interpretation={null}
      earlierInterpretations={[]}
      interpretable={false}
      interpretableReason={null}
      {...props}
    />,
  );
}

beforeEach(() => {
  ask.mockReset();
  refresh.mockReset();
});
afterEach(cleanup);

describe("InterpretationPanel", () => {
  it("shows nothing when the server writes no interpretations and none is on record", () => {
    const { container } = panel();
    expect(container.innerHTML).toBe("");
  });

  it("shows the API's reason when this analysis cannot be interpreted", () => {
    panel({ interpretableReason: "Only an analysis that has succeeded can be interpreted." });
    expect(screen.getByText("Only an analysis that has succeeded can be interpreted.")).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("offers to write one when the analysis can be interpreted", () => {
    panel({ interpretable: true });
    expect(screen.getByRole("button", { name: "Write an interpretation" })).toBeTruthy();
  });

  it("says one is being written while it is queued or running", () => {
    panel({ interpretation: BASE });
    expect(screen.getByText("Writing the interpretation…")).toBeTruthy();
    panel({ interpretation: { ...BASE, status: "running" } });
    expect(screen.getAllByText("Writing the interpretation…")).toHaveLength(2);
  });

  it("shows a verified one under its heading, with the limits, the notice and the model", () => {
    panel({ interpretation: VERIFIED, interpretable: true });
    expect(
      screen.getByRole("heading", { name: "Interpretation (model-written, verified against the facts)" }),
    ).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Estimates" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Summary" })).toBeNull();
    expect(screen.getByText("The estimate for ipw is 0.07.")).toBeTruthy();
    expect(screen.getByText(VERIFIED.limits!)).toBeTruthy();
    expect(screen.getByText(VERIFIED.notice!)).toBeTruthy();
    expect(screen.getByText(/Written by claude-opus-5-5-20261001, prompt abc123def456\./)).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("says a current refused one is not shown, with its attempts and reason, and offers no retry", () => {
    panel({
      interpretation: {
        ...BASE,
        status: "refused",
        interpreter_version: "35d8275efecb",
        out_of_date: false,
        attempts: 2,
        reason: "The last attempt was refused because it wrote a number in its own words instead of citing it.",
      },
      interpretable: true,
    });
    expect(screen.getByText(/could not be verified against this analysis's facts, so it is not shown/)).toBeTruthy();
    expect(screen.getByText("Attempts: 2. The last attempt was refused because it wrote a number in its own words instead of citing it.")).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("shows a failed one's error and lets it be tried again", () => {
    panel({ interpretation: { ...BASE, status: "failed", error: "the model API call failed" }, interpretable: true });
    expect(screen.getByRole("alert").textContent).toBe("the model API call failed");
    expect(screen.getByRole("button", { name: "Try again" })).toBeTruthy();
  });

  it("offers no retry for a failed one when the server no longer writes interpretations", () => {
    panel({ interpretation: { ...BASE, status: "failed", error: "worker lost" } });
    expect(screen.queryByRole("button")).toBeNull();
  });
});

const OUT_OF_DATE = "Written under an earlier version of the prompt and rules.";
const REFUSED_SENTENCE =
  "The model's narrative could not be verified against this analysis's facts, so it is not shown.";

type Recorded = Pick<
  AnalysisDetail,
  "interpretation" | "earlier_interpretations" | "interpretable" | "interpretable_reason"
>;

/** The panel as the page draws it for one of the recorded cases. */
function recorded(name: keyof typeof interpretations) {
  const analysis = interpretations[name] as Recorded;
  return panel({
    interpretation: analysis.interpretation,
    earlierInterpretations: analysis.earlier_interpretations,
    interpretable: analysis.interpretable,
    interpretableReason: analysis.interpretable_reason,
  });
}

function earlierList(): HTMLDetailsElement {
  const heading = screen.getByRole("heading", { name: "Earlier interpretations" });
  const list = heading.closest("section")!.querySelector<HTMLDetailsElement>(":scope > details");
  expect(list).not.toBeNull();
  return list!;
}

/** The list's entries, one per earlier interpretation. */
function earlierEntries(): HTMLLIElement[] {
  return Array.from(earlierList().querySelectorAll<HTMLLIElement>(":scope > ol > li"));
}

describe("InterpretationPanel, when the prompt and rules have changed", () => {
  it("marks a verified one written under an earlier version and offers a new one", () => {
    recorded("verified_out_of_date");
    expect(screen.getByRole("heading", { name: "Estimates" })).toBeTruthy();
    expect(screen.getByText("The estimate for ipw is 1794 [-63.52, 3652].")).toBeTruthy();
    expect(screen.getByText(OUT_OF_DATE)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Write a new interpretation" })).toBeTruthy();
  });

  it("adds nothing under a verified one written under the current version", () => {
    recorded("verified_current");
    expect(screen.getByRole("heading", { name: "Estimates" })).toBeTruthy();
    expect(screen.queryByText(OUT_OF_DATE)).toBeNull();
    expect(screen.queryByRole("button")).toBeNull();
    // The API's reason for refusing a request is not repeated under the interpretation.
    expect(screen.queryByText(interpretations.verified_current.interpretable_reason)).toBeNull();
  });

  it("marks a refused one written under an earlier version and offers a new one", () => {
    recorded("refused_out_of_date");
    expect(screen.getByText(REFUSED_SENTENCE)).toBeTruthy();
    expect(
      screen.getByText(
        "Attempts: 2. The last attempt was refused because it quoted an estimate rated INSUFFICIENT without saying so.",
      ),
    ).toBeTruthy();
    expect(screen.getByText(OUT_OF_DATE)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Write a new interpretation" })).toBeTruthy();
  });

  it("adds nothing under a refused one written under the current version", () => {
    recorded("refused_current");
    expect(screen.getByText(REFUSED_SENTENCE)).toBeTruthy();
    expect(screen.queryByText(OUT_OF_DATE)).toBeNull();
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.queryByText(interpretations.refused_current.interpretable_reason)).toBeNull();
  });

  it("still offers to try a failed one again, written under an earlier version", () => {
    recorded("failed");
    expect(screen.getByRole("alert").textContent).toBe("the model API call failed");
    expect(screen.getByRole("button", { name: "Try again" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Write a new interpretation" })).toBeNull();
    expect(screen.queryByText(OUT_OF_DATE)).toBeNull();
  });

  it("still offers to try a failed one again when it was written under the current version", () => {
    recorded("failed_current");
    expect(screen.getByRole("alert").textContent).toBe("the model API call failed");
    expect(screen.getByRole("button", { name: "Try again" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Write a new interpretation" })).toBeNull();
    expect(screen.queryByText(OUT_OF_DATE)).toBeNull();
  });

  it.each(["failed", "pending"] as const)(
    "marks only a verdict: a %s one flagged out of date gets no line and no new button",
    (name) => {
      // The API flags only a verified or refused interpretation as out of date,
      // but the panel holds the rule itself, so it is tested against a flag the
      // API does not send for these.
      const analysis = interpretations[name] as Recorded;
      panel({
        interpretation: { ...analysis.interpretation!, out_of_date: true },
        earlierInterpretations: analysis.earlier_interpretations,
        interpretable: analysis.interpretable,
        interpretableReason: analysis.interpretable_reason,
      });
      expect(screen.queryByText(OUT_OF_DATE)).toBeNull();
      expect(screen.queryByRole("button", { name: "Write a new interpretation" })).toBeNull();
    },
  );

  it("does not mark one that is still being written, which has no version yet", () => {
    recorded("pending");
    expect(screen.getByText("Writing the interpretation…")).toBeTruthy();
    expect(screen.queryByText(OUT_OF_DATE)).toBeNull();
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.queryByText(interpretations.pending.interpretable_reason)).toBeNull();
    expect(earlierEntries()).toHaveLength(1);
  });
});

describe("InterpretationPanel's earlier interpretations", () => {
  it("lists them, closed, newest first, each with its status, date and interpreter version", () => {
    recorded("with_earlier");
    const list = earlierList();
    expect(list.open).toBe(false);
    const entries = earlierEntries();
    const earlier = interpretations.with_earlier.earlier_interpretations;
    expect(entries).toHaveLength(earlier.length);
    entries.forEach((entry, index) => {
      expect(entry.querySelector("details")!.open).toBe(false);
      expect(entry.querySelector("summary")!.textContent).toBe(
        `${earlier[index].status}, ${formatTime(earlier[index].created_at)}, version ${earlier[index].interpreter_version ?? DASH}`,
      );
    });
    expect(entries.map((entry) => entry.querySelector("summary .status")!.textContent)).toEqual([
      "refused",
      "failed",
      "verified",
    ]);
  });

  it("shows a verified one's sections, its limits and the notice when it is opened", () => {
    recorded("with_earlier");
    const item = earlierEntries()[2];
    const earlier = interpretations.with_earlier.earlier_interpretations[2];
    const entry = within(item);
    const heading = entry.getByRole("heading", { name: "Agreement across methods" });
    // Outside the summary, so only opening the entry shows it.
    expect(heading.closest("summary")).toBeNull();
    const details = item.querySelector("details")!;
    expect(details.open).toBe(false);
    fireEvent.click(details.querySelector("summary")!);
    expect(details.open).toBe(true);
    expect(entry.getByText("The methods agree.")).toBeTruthy();
    expect(entry.queryByRole("heading", { name: "Summary" })).toBeNull();
    expect(entry.getByText(earlier.limits!)).toBeTruthy();
    expect(entry.getByText(earlier.notice!)).toBeTruthy();
    expect(entry.getByText(/Written by claude-opus-5-5-20261001, prompt a28f0812170d\./)).toBeTruthy();
  });

  it.each([
    [
      "with_earlier",
      [
        "H2 Interpretation (model-written, verified against the facts)",
        "H3 Estimates",
        "H3 Earlier interpretations",
        "H4 Agreement across methods",
      ],
    ],
    ["failed_current", ["H2 Interpretation", "H3 Earlier interpretations", "H4 Estimates"]],
    ["pending", ["H2 Interpretation", "H3 Earlier interpretations", "H4 Estimates"]],
  ] as const)("files an earlier one's headings under their own heading (%s)", (name, outline) => {
    const { container } = recorded(name);
    const headings = Array.from(container.querySelectorAll("h1, h2, h3, h4, h5, h6"));
    expect(headings.map((h) => `${h.tagName} ${h.textContent}`)).toEqual(outline);
  });

  it("shows a refused one's refusal sentence only, and a failed one's error", () => {
    recorded("with_earlier");
    const [refusedEntry, failedEntry] = earlierEntries();
    expect(within(refusedEntry).getByText(REFUSED_SENTENCE)).toBeTruthy();
    expect(within(refusedEntry).queryByText(/Attempts/)).toBeNull();
    expect(within(failedEntry).getByText("the model API call failed")).toBeTruthy();
  });

  it("keeps its heading outside the disclosure, which says how many it holds", () => {
    recorded("with_earlier");
    const heading = screen.getByRole("heading", { name: "Earlier interpretations" });
    // A heading inside a <summary> is not announced as one by some screen readers.
    expect(heading.closest("summary")).toBeNull();
    expect(heading.closest("details")).toBeNull();
    const count = interpretations.with_earlier.earlier_interpretations.length;
    expect(earlierList().querySelector(":scope > summary")!.textContent).toBe(
      `Show the ${count} earlier interpretations`,
    );
  });

  it("says so when it holds one", () => {
    const analysis = interpretations.with_earlier as Recorded;
    panel({
      interpretation: analysis.interpretation,
      earlierInterpretations: analysis.earlier_interpretations.slice(0, 1),
    });
    expect(earlierList().querySelector(":scope > summary")!.textContent).toBe(
      "Show the earlier interpretation",
    );
  });

  it("keeps an opened entry open when a newer one is added, even with the same date", () => {
    const analysis = interpretations.with_earlier as Recorded;
    const [first, second] = analysis.earlier_interpretations;
    // Two interpretations asked for in the same instant, told apart only by their ids.
    const twins = [first, { ...second, created_at: first.created_at }];
    const props = {
      interpretation: analysis.interpretation,
      interpretable: false,
      interpretableReason: null,
    };
    const { rerender } = panel({ ...props, earlierInterpretations: twins });
    const opened = earlierEntries()[1].querySelector("details")!;
    fireEvent.click(opened.querySelector("summary")!);
    expect(opened.open).toBe(true);

    const newer = { ...analysis.interpretation!, id: "0b6f2a4e-3c1d-4e5f-8a9b-7c6d5e4f3a2b" };
    rerender(
      <InterpretationPanel analysisId="a1" {...props} earlierInterpretations={[newer, ...twins]} />,
    );
    const entries = earlierEntries();
    expect(entries).toHaveLength(3);
    expect(entries.map((entry) => entry.querySelector("details")!.open)).toEqual([
      false,
      false,
      true,
    ]);
  });

  it("is absent when there are none", () => {
    recorded("verified_out_of_date");
    expect(screen.queryByText("Earlier interpretations")).toBeNull();
    expect(document.querySelector("details")).toBeNull();
  });
});

describe("InterpretButton", () => {
  it("shows the API's sentence when the request is refused, and does not refresh", async () => {
    ask.mockResolvedValue({ error: "Too many interpretations today for this account. Try again in 3 hours." });
    panel({ interpretable: true });
    fireEvent.click(screen.getByRole("button", { name: "Write an interpretation" }));
    await waitFor(() =>
      expect(screen.getByRole("alert").textContent).toBe(
        "Too many interpretations today for this account. Try again in 3 hours.",
      ),
    );
    expect(ask).toHaveBeenCalledWith("a1");
    expect(refresh).not.toHaveBeenCalled();
  });

  it("refreshes the page when the request is accepted", async () => {
    ask.mockResolvedValue({ ok: true });
    panel({ interpretable: true });
    fireEvent.click(screen.getByRole("button", { name: "Write an interpretation" }));
    await waitFor(() => expect(refresh).toHaveBeenCalledTimes(1));
  });
});
