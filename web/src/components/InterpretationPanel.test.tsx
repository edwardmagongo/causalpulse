import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { AnalysisDetail, InterpretationDetail, InterpretationOption } from "@/lib/api/types";
import { DASH, formatTime } from "@/lib/format";
import interpretations from "@/test/fixtures/interpretations.json";

const ask = vi.fn();
const refresh = vi.fn();
vi.mock("@/app/(app)/analyses/[id]/actions", () => ({
  askForInterpretation: (id: string, choice: unknown) => ask(id, choice),
}));
vi.mock("next/navigation", () => ({ useRouter: () => ({ refresh }) }));

import { InterpretationPanel } from "./InterpretationPanel";

const BASE: InterpretationDetail = {
  id: "7a9d3c52-1f0e-4b8a-9c6d-2e5f4a3b1c0d",
  status: "queued",
  model: "claude-opus-5-5",
  effort: "medium",
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

/** The server's default entry, available. */
const OPUS: InterpretationOption = {
  model: "claude-opus-5-5",
  effort: "medium",
  default: true,
  available: true,
  reason: null,
};

function panel(props: Partial<Parameters<typeof InterpretationPanel>[0]> = {}) {
  return render(
    <InterpretationPanel
      analysisId="a1"
      interpretation={null}
      earlierInterpretations={[]}
      options={[]}
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
    panel({ options: [OPUS] });
    expect(screen.getByRole("button", { name: "Write an interpretation" })).toBeTruthy();
  });

  it("says one is being written while it is queued or running", () => {
    panel({ interpretation: BASE });
    expect(screen.getByText("Writing the interpretation…")).toBeTruthy();
    panel({ interpretation: { ...BASE, status: "running" } });
    expect(screen.getAllByText("Writing the interpretation…")).toHaveLength(2);
  });

  it("shows a verified one under its heading, with the limits, the notice and the model", () => {
    panel({ interpretation: VERIFIED, options: [{ ...OPUS, available: false }] });
    expect(
      screen.getByRole("heading", { name: "Interpretation (model-written, verified against the facts)" }),
    ).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Estimates" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Summary" })).toBeNull();
    expect(screen.getByText("The estimate for ipw is 0.07.")).toBeTruthy();
    expect(screen.getByText(VERIFIED.limits!)).toBeTruthy();
    expect(screen.getByText(VERIFIED.notice!)).toBeTruthy();
    expect(
      screen.getByText("Written by claude-opus-5-5-20261001, medium effort, prompt abc123def456."),
    ).toBeTruthy();
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
      options: [{ ...OPUS, available: false }],
    });
    expect(screen.getByText(/could not be verified against this analysis's facts, so it is not shown/)).toBeTruthy();
    expect(screen.getByText("Attempts: 2. The last attempt was refused because it wrote a number in its own words instead of citing it.")).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("shows a failed one's error and lets it be tried again", () => {
    panel({ interpretation: { ...BASE, status: "failed", error: "the model API call failed" }, options: [OPUS] });
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
  | "interpretation"
  | "earlier_interpretations"
  | "interpretation_options"
  | "interpretable"
  | "interpretable_reason"
>;

/** The panel as the page draws it for one of the recorded cases. */
function recorded(name: keyof typeof interpretations) {
  const analysis = interpretations[name] as Recorded;
  return panel({
    interpretation: analysis.interpretation,
    earlierInterpretations: analysis.earlier_interpretations,
    options: analysis.interpretation_options,
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
        options: analysis.interpretation_options,
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
  it("lists them, closed, newest first, each with its status, entry, date and interpreter version", () => {
    recorded("with_earlier");
    const list = earlierList();
    expect(list.open).toBe(false);
    const entries = earlierEntries();
    const earlier = interpretations.with_earlier.earlier_interpretations;
    expect(entries).toHaveLength(earlier.length);
    entries.forEach((entry, index) => {
      expect(entry.querySelector("details")!.open).toBe(false);
      expect(entry.querySelector("summary")!.textContent).toBe(
        `${earlier[index].status}, ${earlier[index].model}, ${earlier[index].effort}, ${formatTime(earlier[index].created_at)}, version ${earlier[index].interpreter_version ?? DASH}`,
      );
    });
    // Each names its own entry, "model, effort", as the choice does.
    expect(entries.map((entry) => entry.querySelector("summary .status + .value")!.textContent)).toEqual([
      "claude-sonnet-5-5, high",
      "claude-opus-5-5, medium",
      "claude-opus-5-5, medium",
    ]);
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
    expect(
      entry.getByText("Written by claude-opus-5-5-20261001, medium effort, prompt a28f0812170d."),
    ).toBeTruthy();
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

  it("shows a refused one's refusal sentence without its attempts, and a failed one's error", () => {
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
      options: [],
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

const SONNET_LOW: InterpretationOption = {
  model: "claude-sonnet-5-5",
  effort: "low",
  default: false,
  available: true,
  reason: null,
};

function choice(): HTMLSelectElement {
  return screen.getByRole<HTMLSelectElement>("combobox", { name: "Model and effort" });
}

function offered(): string[] {
  return Array.from(choice().options).map((option) => option.textContent ?? "");
}

function selected(): string {
  return choice().selectedOptions[0].textContent ?? "";
}

describe("InterpretationPanel's choice of model and effort", () => {
  it("offers only the available entries, each as 'model, effort', in the server's order", () => {
    recorded("choices_after_refused");
    expect(offered()).toEqual(["claude-sonnet-5-5, low", "claude-sonnet-5-5, high"]);
    expect(screen.queryByText("claude-opus-5-5, medium")).toBeNull();
  });

  it("keeps the choice in the same form as the button", () => {
    recorded("choices_after_refused");
    const button = screen.getByRole<HTMLButtonElement>("button", { name: "Write a new interpretation" });
    expect(choice().form).not.toBeNull();
    expect(choice().form).toBe(button.form);
  });

  it("selects the default when it is available", () => {
    recorded("choices_none_yet");
    expect(offered()).toEqual(["claude-opus-5-5, medium", "claude-sonnet-5-5, low"]);
    expect(selected()).toBe("claude-opus-5-5, medium");
    expect(screen.getByRole("button", { name: "Write an interpretation" })).toBeTruthy();
  });

  it("selects the default wherever it stands in the list", () => {
    // The API lists the default first, but the panel holds the rule itself.
    panel({ options: [SONNET_LOW, OPUS] });
    expect(selected()).toBe("claude-opus-5-5, medium");
  });

  it("selects the first available entry when the default is not available", () => {
    recorded("choices_after_refused");
    expect(selected()).toBe("claude-sonnet-5-5, low");
  });

  it("asks for the entry chosen", async () => {
    ask.mockResolvedValue({ ok: true });
    recorded("choices_after_refused");
    fireEvent.change(choice(), { target: { value: choice().options[1].value } });
    expect(selected()).toBe("claude-sonnet-5-5, high");
    fireEvent.click(screen.getByRole("button", { name: "Write a new interpretation" }));
    await waitFor(() => expect(refresh).toHaveBeenCalledTimes(1));
    expect(ask).toHaveBeenCalledTimes(1);
    expect(ask).toHaveBeenCalledWith("a1", { model: "claude-sonnet-5-5", effort: "high" });
  });

  it("asks for the entry selected at first when none is chosen", async () => {
    ask.mockResolvedValue({ ok: true });
    recorded("choices_after_refused");
    fireEvent.click(screen.getByRole("button", { name: "Write a new interpretation" }));
    await waitFor(() => expect(refresh).toHaveBeenCalledTimes(1));
    expect(ask).toHaveBeenCalledWith("a1", { model: "claude-sonnet-5-5", effort: "low" });
  });

  it("offers a new one under a current refused one when another entry is available", () => {
    recorded("choices_after_refused");
    expect(screen.getByText(REFUSED_SENTENCE)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Write a new interpretation" })).toBeTruthy();
    expect(screen.queryByText(OUT_OF_DATE)).toBeNull();
  });

  it("offers a new one under a current verified one, naming the one entry beside the button", async () => {
    ask.mockResolvedValue({ ok: true });
    recorded("another_after_verified");
    expect(screen.getByRole("heading", { name: "Estimates" })).toBeTruthy();
    expect(screen.queryByRole("combobox")).toBeNull();
    const button = screen.getByRole("button", { name: "Write a new interpretation" });
    const label = screen.getByText("claude-sonnet-5-5, low");
    expect(label.closest("form")).toBe(button.closest("form"));
    expect(screen.queryByText("claude-opus-5-5, medium")).toBeNull();
    expect(screen.queryByText(OUT_OF_DATE)).toBeNull();
    fireEvent.click(button);
    await waitFor(() => expect(refresh).toHaveBeenCalledTimes(1));
    expect(ask).toHaveBeenCalledWith("a1", { model: "claude-sonnet-5-5", effort: "low" });
  });

  it("names the one entry beside the button for a first interpretation and a retry", () => {
    panel({ options: [OPUS] });
    expect(screen.queryByRole("combobox")).toBeNull();
    expect(screen.getByText("claude-opus-5-5, medium")).toBeTruthy();
    cleanup();
    recorded("failed");
    expect(screen.queryByRole("combobox")).toBeNull();
    expect(screen.getByRole("button", { name: "Try again" })).toBeTruthy();
    expect(screen.getByText("claude-opus-5-5, medium")).toBeTruthy();
  });

  it("offers the choice for trying a failed one again", () => {
    panel({
      interpretation: { ...BASE, status: "failed", error: "the model API call failed" },
      options: [OPUS, SONNET_LOW],
    });
    expect(offered()).toEqual(["claude-opus-5-5, medium", "claude-sonnet-5-5, low"]);
    expect(screen.getByRole("button", { name: "Try again" })).toBeTruthy();
  });

  it("still marks an out-of-date one, with the choice beneath the line", () => {
    const analysis = interpretations.verified_out_of_date as Recorded;
    panel({
      interpretation: analysis.interpretation,
      options: [OPUS, SONNET_LOW],
    });
    const line = screen.getByText(OUT_OF_DATE);
    expect(offered()).toEqual(["claude-opus-5-5, medium", "claude-sonnet-5-5, low"]);
    expect(screen.getByRole("button", { name: "Write a new interpretation" })).toBeTruthy();
    expect(line.compareDocumentPosition(choice()) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("offers no button while the newest is being written, even with an entry marked available", () => {
    // The API makes no entry available while one is pending, but the panel holds the rule itself.
    panel({ interpretation: { ...BASE, status: "running" }, options: [OPUS, SONNET_LOW] });
    expect(screen.getByText("Writing the interpretation…")).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.queryByRole("combobox")).toBeNull();
    cleanup();
    panel({ interpretation: BASE, options: [OPUS] });
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.queryByText("claude-opus-5-5, medium")).toBeNull();
  });

  it("describes the button by the one entry it asks with", () => {
    recorded("another_after_verified");
    const button = screen.getByRole("button", { name: "Write a new interpretation" });
    const label = screen.getByText("claude-sonnet-5-5, low");
    expect(label.id).not.toBe("");
    expect(button.getAttribute("aria-describedby")).toBe(label.id);
  });

  it("clears a refusal when another entry is chosen", async () => {
    const sentence = "That model and effort are not offered on this server.";
    ask.mockResolvedValue({ error: sentence });
    recorded("choices_after_refused");
    fireEvent.click(screen.getByRole("button", { name: "Write a new interpretation" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toBe(sentence));
    fireEvent.change(choice(), { target: { value: choice().options[1].value } });
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByText(sentence)).toBeNull();
  });

  it.each(["verified_current", "refused_current", "pending", "with_earlier"] as const)(
    "offers no choice and no button when no entry is available (%s)",
    (name) => {
      recorded(name);
      expect(screen.queryByRole("combobox")).toBeNull();
      expect(screen.queryByRole("button")).toBeNull();
      // An earlier interpretation's summary names its entry; nothing else may.
      const named = screen.queryAllByText("claude-opus-5-5, medium");
      expect(named.filter((element) => element.closest("summary") === null)).toEqual([]);
    },
  );

  it("offers nothing that is not available, even with none written", () => {
    panel({
      options: [{ ...OPUS, available: false, reason: "Only an analysis that has succeeded can be interpreted." }],
      interpretableReason: "Only an analysis that has succeeded can be interpreted.",
    });
    expect(screen.getByText("Only an analysis that has succeeded can be interpreted.")).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.queryByText("claude-opus-5-5, medium")).toBeNull();
  });
});

const EARLIER_ID = "4c2e8f1a-6b3d-5e7f-9a0c-1d2b3e4f5a6b";

describe("InterpretationPanel's line on what wrote each interpretation", () => {
  it("names the model, the effort and the prompt of a newest refused one", () => {
    recorded("choices_after_refused");
    expect(
      screen.getByText("Written by claude-opus-5-5-20261001, medium effort, prompt 5b08e57994d4."),
    ).toBeTruthy();
  });

  it("says what a newest failed one was asked of, and what an earlier failed one was", () => {
    recorded("failed");
    expect(screen.getByText("Asked of claude-opus-5-5, medium effort.")).toBeTruthy();
    expect(screen.queryByText(/Written by/)).toBeNull();
    cleanup();
    recorded("choices_after_refused");
    const [failedEntry] = earlierEntries();
    expect(within(failedEntry).getByText("Asked of claude-sonnet-5-5, high effort.")).toBeTruthy();
    expect(within(failedEntry).queryByText(/Written by/)).toBeNull();
  });

  it.each(["queued", "running"] as const)(
    "says what a newest %s one is being written by, and not that it was written",
    (status) => {
      panel({ interpretation: { ...BASE, status } });
      expect(screen.getByText("Being written by claude-opus-5-5, medium effort.")).toBeTruthy();
      expect(screen.queryByText(/Written by/)).toBeNull();
    },
  );

  it.each(["queued", "running"] as const)(
    "says what an earlier %s one is being written by, and not that it was written",
    (status) => {
      // The API has at most one pending, the newest, but the panel holds the rule itself.
      panel({
        interpretation: VERIFIED,
        earlierInterpretations: [
          { ...BASE, id: EARLIER_ID, status, model: "claude-sonnet-5-5", effort: "xhigh" },
        ],
      });
      const [entry] = earlierEntries();
      expect(within(entry).getByText("Being written by claude-sonnet-5-5, xhigh effort.")).toBeTruthy();
      expect(within(entry).queryByText(/Written by/)).toBeNull();
    },
  );

  it("names them for every earlier one", () => {
    recorded("with_earlier");
    const [refusedEntry, failedEntry, verifiedEntry] = earlierEntries();
    expect(
      within(refusedEntry).getByText(
        "Written by claude-sonnet-5-5-20260915, high effort, prompt 5b08e57994d4.",
      ),
    ).toBeTruthy();
    expect(within(failedEntry).getByText("Asked of claude-opus-5-5, medium effort.")).toBeTruthy();
    expect(
      within(verifiedEntry).getByText(
        "Written by claude-opus-5-5-20261001, medium effort, prompt a28f0812170d.",
      ),
    ).toBeTruthy();
  });

  it("names an unmeasured entry, and shows the API's notice about it", () => {
    recorded("verified_unmeasured");
    expect(
      screen.getByText("Written by claude-sonnet-5-5-20260915, low effort, prompt 5b08e57994d4."),
    ).toBeTruthy();
    expect(screen.getByText(interpretations.verified_unmeasured.interpretation.notice)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Write a new interpretation" })).toBeTruthy();
    expect(screen.getByText("claude-opus-5-5, medium")).toBeTruthy();
  });
});

describe("InterpretButton", () => {
  it("shows the API's sentence when the request is refused, and does not refresh", async () => {
    ask.mockResolvedValue({ error: "Too many interpretations today for this account. Try again in 3 hours." });
    panel({ options: [OPUS] });
    fireEvent.click(screen.getByRole("button", { name: "Write an interpretation" }));
    await waitFor(() =>
      expect(screen.getByRole("alert").textContent).toBe(
        "Too many interpretations today for this account. Try again in 3 hours.",
      ),
    );
    expect(ask).toHaveBeenCalledWith("a1", { model: "claude-opus-5-5", effort: "medium" });
    expect(refresh).not.toHaveBeenCalled();
  });

  it("refreshes the page when the request is accepted", async () => {
    ask.mockResolvedValue({ ok: true });
    panel({ options: [OPUS] });
    fireEvent.click(screen.getByRole("button", { name: "Write an interpretation" }));
    await waitFor(() => expect(refresh).toHaveBeenCalledTimes(1));
  });
});

const VERIFIED_HEADING = "Interpretation (model-written, verified against the facts)";

function newer(): HTMLElement {
  const heading = screen.getByRole("heading", { name: "Newer interpretation" });
  return heading.closest("section")!;
}

describe("InterpretationPanel, when the newest is not verified but a current one is", () => {
  it("leads with the current verified one, and shows the newer failed one beneath it", () => {
    recorded("failed_after_current_verified");

    expect(screen.getByRole("heading", { level: 2 }).textContent).toBe(VERIFIED_HEADING);
    const lead = screen.getByText("The estimate for ipw is 1794 [-63.52, 3652].");
    expect(lead.closest("details")).toBeNull();
    const section = newer();
    expect(within(section).getByRole("alert").textContent).toBe(
      "the model API call failed: APIConnectionError",
    );
    expect(within(section).getByText("Asked of claude-sonnet-5-5, low effort.")).toBeTruthy();
    expect(within(section).getByRole("button", { name: "Try again" })).toBeTruthy();
    expect(screen.getAllByRole("button")).toHaveLength(1);
  });

  it("lists neither the leading one nor the newer one among the earlier ones", () => {
    recorded("failed_after_current_verified");

    const entries = earlierEntries();
    expect(entries).toHaveLength(1);
    expect(entries[0].querySelector("summary")!.textContent).toContain("failed, claude-opus-5-5, medium");
  });

  it("shows a newer refused one beneath it, with a new one offered", () => {
    const analysis = interpretations.failed_after_current_verified as Recorded;
    panel({
      interpretation: {
        ...analysis.interpretation!,
        status: "refused",
        error: null,
        attempts: 2,
        returned_model: "claude-sonnet-5-5-20261001",
        prompt_version: "5b08e57994d4",
      },
      earlierInterpretations: analysis.earlier_interpretations,
      options: analysis.interpretation_options,
    });

    expect(screen.getByRole("heading", { level: 2 }).textContent).toBe(VERIFIED_HEADING);
    const section = newer();
    expect(within(section).getByText(REFUSED_SENTENCE)).toBeTruthy();
    expect(within(section).getByRole("button", { name: "Write a new interpretation" })).toBeTruthy();
  });

  it("shows a newer one still being written beneath it, with no button", () => {
    const analysis = interpretations.failed_after_current_verified as Recorded;
    panel({
      interpretation: { ...analysis.interpretation!, status: "running", error: null },
      earlierInterpretations: analysis.earlier_interpretations,
      options: analysis.interpretation_options,
    });

    expect(screen.getByRole("heading", { level: 2 }).textContent).toBe(VERIFIED_HEADING);
    expect(within(newer()).getByText("Writing the interpretation…")).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("leads with the newest of two current verified ones, and lists the other", () => {
    const analysis = interpretations.failed_after_current_verified as Recorded;
    const [latest, olderFailed] = analysis.earlier_interpretations;
    const older = {
      ...latest,
      id: "0d9c8b7a-6e5f-5a4b-9c3d-2e1f0a9b8c7d",
      model: "claude-fable-5-1",
      created_at: "2026-10-18T08:00:00+00:00",
      sections: [{ name: "summary", heading: "Summary", statements: ["The older narrative."] }],
    };
    panel({
      interpretation: analysis.interpretation,
      earlierInterpretations: [latest, olderFailed, older],
      options: analysis.interpretation_options,
    });

    expect(screen.getByText("The estimate for ipw is 1794 [-63.52, 3652].").closest("details")).toBeNull();
    const entries = earlierEntries();
    expect(entries).toHaveLength(2);
    expect(entries[1].querySelector("summary")!.textContent).toContain("verified, claude-fable-5-1");
  });

  it("does not lead with a verified one written under an earlier version", () => {
    recorded("failed_current");

    expect(screen.getByRole("heading", { level: 2 }).textContent).toBe("Interpretation");
    expect(screen.queryByRole("heading", { name: "Newer interpretation" })).toBeNull();
    expect(earlierEntries()).toHaveLength(1);
  });
});
