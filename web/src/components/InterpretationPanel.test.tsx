import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { InterpretationDetail } from "@/lib/api/types";

const ask = vi.fn();
const refresh = vi.fn();
vi.mock("@/app/(app)/analyses/[id]/actions", () => ({ askForInterpretation: (id: string) => ask(id) }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ refresh }) }));

import { InterpretationPanel } from "./InterpretationPanel";

const BASE: InterpretationDetail = {
  status: "queued",
  model: "claude-opus-5-5",
  returned_model: null,
  prompt_version: null,
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

  it("says a refused one is not shown, with its attempts and reason, and offers no retry", () => {
    panel({
      interpretation: { ...BASE, status: "refused", attempts: 2, reason: "The last attempt was refused because it wrote a number in its own words instead of citing it." },
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
