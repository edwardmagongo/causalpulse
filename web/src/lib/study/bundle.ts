import "server-only";

import bundle from "@/data/benchmark-study.json";

import { isEstimatorFamily, isShape } from "./types";
import type { StudyBundle, StudyLoad, VersionCheck, VocabularyCheck } from "./types";

/** The schema versions this build of the dashboard can read. */
export const KNOWN_SCHEMA_VERSIONS = [1] as const;

export function checkVersion(found: number): VersionCheck {
  const known: readonly number[] = KNOWN_SCHEMA_VERSIONS;
  return known.includes(found) ? { ok: true } : { ok: false, reason: "version", found, known };
}

/**
 * Every scenario's shape and every estimator's family must be one this build
 * knows: `matrixOf` has no column or family for any other. `unknown` names each
 * offender, for example `scenario P9: shape "staggered"`.
 */
export function checkVocabulary(study: StudyBundle): VocabularyCheck {
  const unknown = [
    ...study.scenarios
      .filter((scenario) => !isShape(scenario.shape))
      .map((scenario) => `scenario ${scenario.name}: shape "${scenario.shape}"`),
    ...study.estimators
      .filter((estimator) => !isEstimatorFamily(estimator.family))
      .map((estimator) => `estimator ${estimator.name}: family "${estimator.family}"`),
  ];
  return unknown.length === 0 ? { ok: true } : { ok: false, reason: "vocabulary", unknown };
}

/**
 * The drift guard: this plain typed assignment fails `tsc` when the committed
 * bundle stops matching `StudyBundle`. It is the only import of the file.
 */
const study: StudyBundle = bundle;

/** Version first, then vocabulary: a study of another version is not read for its names. */
export function loadFrom(candidate: StudyBundle): StudyLoad {
  const version = checkVersion(candidate.schema_version);
  if (!version.ok) return version;
  const vocabulary = checkVocabulary(candidate);
  if (!vocabulary.ok) return vocabulary;
  return { ok: true, study: candidate };
}

export function loadStudy(): StudyLoad {
  return loadFrom(study);
}
