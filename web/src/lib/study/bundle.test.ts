import { describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));

import { KNOWN_SCHEMA_VERSIONS, checkVersion, checkVocabulary, loadFrom, loadStudy } from "./bundle";
import type { StudyBundle } from "./types";

describe("loadStudy", () => {
  it("reads the committed bundle", () => {
    const loaded = loadStudy();
    expect(loaded.ok).toBe(true);
    if (loaded.ok) {
      expect(loaded.study.schema_version).toBe(1);
      expect(loaded.study.estimators).toHaveLength(12);
      expect(loaded.study.scenarios).toHaveLength(19);
    }
  });
});

describe("checkVersion", () => {
  it("accepts every known version", () => {
    for (const version of KNOWN_SCHEMA_VERSIONS) {
      expect(checkVersion(version)).toEqual({ ok: true });
    }
  });

  it("refuses a version it does not know, naming the one found and the ones known", () => {
    expect(checkVersion(2)).toEqual({
      ok: false,
      reason: "version",
      found: 2,
      known: KNOWN_SCHEMA_VERSIONS,
    });
  });
});

describe("checkVocabulary", () => {
  const loaded = loadStudy();
  if (!loaded.ok) throw new Error("the committed bundle must load");
  const study = loaded.study;

  it("accepts every shape and family of the committed bundle", () => {
    expect(checkVocabulary(study)).toEqual({ ok: true });
  });

  it("refuses a scenario whose shape it does not know, naming the scenario", () => {
    const altered: StudyBundle = {
      ...study,
      scenarios: study.scenarios.map((s) => (s.name === "P1" ? { ...s, shape: "staggered" } : s)),
    };
    expect(checkVocabulary(altered)).toEqual({
      ok: false,
      reason: "vocabulary",
      unknown: ['scenario P1: shape "staggered"'],
    });
  });

  it("refuses an estimator whose family it does not know, naming the estimator", () => {
    const altered: StudyBundle = {
      ...study,
      estimators: study.estimators.map((e) => (e.name === "aipw" ? { ...e, family: "bayesian" } : e)),
    };
    expect(checkVocabulary(altered)).toEqual({
      ok: false,
      reason: "vocabulary",
      unknown: ['estimator aipw: family "bayesian"'],
    });
  });

  it("lists every offender, scenarios first", () => {
    const altered: StudyBundle = {
      ...study,
      scenarios: study.scenarios.map((s) => (s.name === "A" || s.name === "S0" ? { ...s, shape: "x" } : s)),
      estimators: study.estimators.map((e) => (e.name === "ipw" ? { ...e, family: "y" } : e)),
    };
    expect(checkVocabulary(altered)).toEqual({
      ok: false,
      reason: "vocabulary",
      unknown: ['scenario A: shape "x"', 'scenario S0: shape "x"', 'estimator ipw: family "y"'],
    });
  });
});

describe("loadFrom", () => {
  const loaded = loadStudy();
  if (!loaded.ok) throw new Error("the committed bundle must load");
  const study = loaded.study;
  const staggered = study.scenarios.map((s) => (s.name === "P1" ? { ...s, shape: "staggered" } : s));

  it("loads the committed study", () => {
    expect(loadFrom(study)).toEqual({ ok: true, study });
  });

  it("refuses a study of a version it does not know, naming the version", () => {
    expect(loadFrom({ ...study, schema_version: 2 })).toEqual({
      ok: false,
      reason: "version",
      found: 2,
      known: KNOWN_SCHEMA_VERSIONS,
    });
  });

  it("refuses a study with a shape it does not know, naming the scenario", () => {
    expect(loadFrom({ ...study, scenarios: staggered })).toEqual({
      ok: false,
      reason: "vocabulary",
      unknown: ['scenario P1: shape "staggered"'],
    });
  });

  it("reports the version first when the study fails both checks", () => {
    expect(loadFrom({ ...study, schema_version: 2, scenarios: staggered })).toMatchObject({
      ok: false,
      reason: "version",
      found: 2,
    });
  });
});
