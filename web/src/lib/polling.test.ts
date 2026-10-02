import { expect, it } from "vitest";

import { FIRST_DELAY_MS, MAX_DELAY_MS, inFlight, nextDelay } from "./polling";

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
