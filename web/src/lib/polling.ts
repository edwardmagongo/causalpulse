import type { Status } from "@/lib/api/types";

export const FIRST_DELAY_MS = 2_000;
export const MAX_DELAY_MS = 15_000;

/** The wait before the next poll: half again as long each time, up to the ceiling. */
export function nextDelay(current: number): number {
  return Math.min(Math.round(current * 1.5), MAX_DELAY_MS);
}

/** Whether an analysis can still change, and so is worth polling. */
export function inFlight(status: Status): boolean {
  return status === "queued" || status === "running";
}
