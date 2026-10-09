import { describe, expect, it } from "vitest";

import type { AnalysisDetail } from "@/lib/api/types";

import { FIRST_DELAY_MS, MAX_DELAY_MS, inFlight, nextDelay, stillChanging } from "./polling";

it("backs off by half again each time and stops at the ceiling", () => {
  const delays = [FIRST_DELAY_MS];
  for (let i = 0; i < 7; i += 1) delays.push(nextDelay(delays[delays.length - 1]));
  expect(delays).toEqual([2000, 3000, 4500, 6750, 10125, 15000, 15000, 15000]);
  expect(nextDelay(MAX_DELAY_MS)).toBe(MAX_DELAY_MS);
});

it("polls only an analysis that can still change", () => {
  expect(inFlight("queued")).toBe(true);
  expect(inFlight("running")).toBe(true);
  expect(inFlight("succeeded")).toBe(false);
  expect(inFlight("failed")).toBe(false);
});

describe("stillChanging", () => {
  const none = { interpretation: null };
  const pending = (status: "queued" | "running" | "verified" | "refused" | "failed") => ({
    interpretation: { status } as AnalysisDetail["interpretation"],
  });

  it("is true while the analysis is in flight", () => {
    expect(stillChanging({ status: "queued", ...none })).toBe(true);
    expect(stillChanging({ status: "running", ...none })).toBe(true);
  });

  it("is true while a finished analysis's interpretation is queued or running", () => {
    expect(stillChanging({ status: "succeeded", ...pending("queued") })).toBe(true);
    expect(stillChanging({ status: "succeeded", ...pending("running") })).toBe(true);
  });

  it("is false once both have finished", () => {
    expect(stillChanging({ status: "succeeded", ...none })).toBe(false);
    for (const done of ["verified", "refused", "failed"] as const) {
      expect(stillChanging({ status: "succeeded", ...pending(done) })).toBe(false);
    }
  });
});
