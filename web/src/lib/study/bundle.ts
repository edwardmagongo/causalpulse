import "server-only";

import bundle from "@/data/benchmark-study.json";

import type { StudyBundle } from "./types";

/** The schema versions this build of the dashboard can read. */
export const KNOWN_SCHEMA_VERSIONS = [1] as const;

export type VersionCheck = { ok: true } | { ok: false; found: number; known: readonly number[] };

export function checkVersion(found: number): VersionCheck {
  const known: readonly number[] = KNOWN_SCHEMA_VERSIONS;
  return known.includes(found) ? { ok: true } : { ok: false, found, known };
}

/**
 * The drift guard: this plain typed assignment fails `tsc` when the committed
 * bundle stops matching `StudyBundle`. It is the only import of the file.
 */
const study: StudyBundle = bundle;

export type StudyLoad =
  | { ok: true; study: StudyBundle }
  | { ok: false; found: number; known: readonly number[] };

export function loadStudy(): StudyLoad {
  const checked = checkVersion(study.schema_version);
  if (!checked.ok) return checked;
  return { ok: true, study };
}
