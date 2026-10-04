import { describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));

import { KNOWN_SCHEMA_VERSIONS, checkVersion, loadStudy } from "./bundle";

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
    expect(checkVersion(2)).toEqual({ ok: false, found: 2, known: KNOWN_SCHEMA_VERSIONS });
  });
});
