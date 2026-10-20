/** One place numbers and timestamps become text. */

import type { InterpretationOption } from "@/lib/api/types";

export const DASH = "—";

/** Four significant figures; `null` is a dash, never a zero. */
export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return DASH;
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e6 || magnitude < 1e-3) return value.toExponential(3);
  return Number(value.toPrecision(4)).toString();
}

/** A p-value: small ones as a bound, the rest to three decimals. */
export function formatP(value: number | null | undefined): string {
  if (value === null || value === undefined) return DASH;
  return value < 0.001 ? "< 0.001" : value.toFixed(3);
}

/** `[lower, upper]`, or a dash when either end is absent. */
export function formatInterval(
  lower: number | null | undefined,
  upper: number | null | undefined,
): string {
  if (lower === null || lower === undefined || upper === null || upper === undefined) return DASH;
  return `[${formatNumber(lower)}, ${formatNumber(upper)}]`;
}

/** An interval's level as a percentage: 0.95 is "95%". */
export function formatLevel(level: number): string {
  return `${Number((level * 100).toPrecision(4))}%`;
}

/** An ISO timestamp in UTC to the second, so a page reads the same on every machine. */
export function formatTime(iso: string | null | undefined): string {
  if (!iso) return DASH;
  const moment = new Date(iso);
  if (Number.isNaN(moment.getTime())) return DASH;
  return `${moment.toISOString().slice(0, 19).replace("T", " ")} UTC`;
}

/** Seconds as "0.4 s" or "2 min 5 s". The total is rounded before it is split into minutes. */
export function formatSeconds(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return DASH;
  if (seconds < 60) {
    const rounded = Number(seconds.toPrecision(2));
    if (rounded < 60) return `${rounded} s`;
  }
  const total = Math.round(seconds);
  return `${Math.floor(total / 60)} min ${total % 60} s`;
}

/**
 * A byte count in B, KB or MB. Below 1024 bytes it is shown exactly in B. From
 * there it is shown in KB, unless it would round to 1000 KB or more at three
 * significant figures, in which case it is shown in MB: 1,048,575 bytes is
 * "1 MB", not "1020 KB", and 1000 to 1023 KB read as 0.977 to 0.999 MB.
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kilobytes = bytes / 1024;
  if (Number(kilobytes.toPrecision(3)) < 1000) return `${Number(kilobytes.toPrecision(3))} KB`;
  return `${Number((bytes / (1024 * 1024)).toPrecision(3))} MB`;
}

/**
 * How an entry of the server's interpretation list is named: "model, effort".
 * Here, not beside the button that offers the entries, because a server
 * component cannot call a function exported from a client module.
 */
export function optionLabel(option: Pick<InterpretationOption, "model" | "effort">): string {
  return `${option.model}, ${option.effort}`;
}
