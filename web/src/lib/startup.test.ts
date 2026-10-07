// @vitest-environment node
import { readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DEMO_SESSION_SECRETS, isProduction } from "@/lib/env";
import { requireSettings } from "@/lib/startup";

const STRONG = "Zq7-vN2xLw9_TfB4hKc8RsJd-m3Xp6Yg";

/** What `process.exit` becomes: a throw, so the check stops where an exit would. */
class Exited extends Error {
  constructor(readonly code: number | string | null | undefined) {
    super(`exit ${code}`);
  }
}

let printed: ReturnType<typeof vi.spyOn>;

/** Everything the check wrote to the log, as one string. */
function log(): string {
  return printed.mock.calls.flat().map(String).join("\n");
}

beforeEach(() => {
  vi.spyOn(process, "exit").mockImplementation((code) => {
    throw new Exited(code);
  });
  printed = vi.spyOn(console, "error").mockImplementation(() => {});
  // A complete, safe production environment; each test spoils one thing.
  vi.stubEnv("CP_API_URL", "http://api.test");
  vi.stubEnv("CP_SESSION_SECRET", STRONG);
  vi.stubEnv("CP_COOKIE_SECURE", "true");
  vi.stubEnv("CP_ENV", "production");
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("isProduction", () => {
  it("is true for CP_ENV=production and for nothing else", () => {
    expect(isProduction()).toBe(true);
    for (const value of ["", "development", "Production", "prod", "1", "true"]) {
      vi.stubEnv("CP_ENV", value);
      expect(isProduction(), value).toBe(false);
    }
    vi.stubEnv("CP_ENV", undefined);
    expect(isProduction()).toBe(false);
  });
});

describe("requireSettings", () => {
  it("accepts a production environment with its own secret and a secure cookie", () => {
    expect(() => requireSettings()).not.toThrow();
    expect(printed).not.toHaveBeenCalled();
  });

  it("still exits without the settings it always needed", () => {
    vi.stubEnv("CP_ENV", undefined);
    vi.stubEnv("CP_SESSION_SECRET", undefined);
    expect(() => requireSettings()).toThrow(new Exited(1));
    expect(log()).toContain("CP_SESSION_SECRET");
  });

  it.each([...DEMO_SESSION_SECRETS])("refuses a demo session secret in production", (secret) => {
    vi.stubEnv("CP_SESSION_SECRET", secret);
    expect(() => requireSettings()).toThrow(new Exited(1));
    expect(log()).toContain("CP_SESSION_SECRET");
    expect(log()).not.toContain(secret);
  });

  it.each([...DEMO_SESSION_SECRETS])("accepts a demo session secret without CP_ENV", (secret) => {
    vi.stubEnv("CP_ENV", undefined);
    vi.stubEnv("CP_COOKIE_SECURE", undefined);
    vi.stubEnv("CP_SESSION_SECRET", secret);
    expect(() => requireSettings()).not.toThrow();
    expect(printed).not.toHaveBeenCalled();
  });

  it.each([undefined, "", "false", "TRUE", "1", "yes-please"])(
    "refuses CP_COOKIE_SECURE=%s in production",
    (value) => {
      vi.stubEnv("CP_COOKIE_SECURE", value);
      expect(() => requireSettings()).toThrow(new Exited(1));
      expect(log()).toContain("CP_COOKIE_SECURE");
      if (value === "yes-please") expect(log()).not.toContain(value);
    },
  );

  it("does not ask for a secure cookie without CP_ENV", () => {
    vi.stubEnv("CP_ENV", undefined);
    vi.stubEnv("CP_COOKIE_SECURE", undefined);
    expect(() => requireSettings()).not.toThrow();
  });
});

describe("DEMO_SESSION_SECRETS", () => {
  const web = resolve(__dirname, "../..");
  const repository = resolve(web, "..");

  /** Every file under `web/` that is the project's own: not installed, not built. */
  function ownFiles(directory: string): string[] {
    return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
      if (["node_modules", ".next", "test-results", "playwright-report"].includes(entry.name)) {
        return [];
      }
      const path = join(directory, entry.name);
      return entry.isDirectory() ? ownFiles(path) : [path];
    });
  }

  // `CP_SESSION_SECRET: value` in a compose file, `CP_SESSION_SECRET=value`
  // in a Makefile, a script or an environment file.
  const assignment = /CP_SESSION_SECRET\s*[:=]\s*["']?([^\s"']+)/g;

  function secretsIn(path: string): string[] {
    let text: string;
    try {
      text = readFileSync(path, "utf-8");
    } catch {
      return [];
    }
    return (
      [...text.matchAll(assignment)]
        .map((match) => match[1])
        // `$(...)` and `${...}` are filled in when the file is used: a value
        // generated or read from the environment is not one the repository holds.
        .filter((value) => !value.startsWith("$"))
    );
  }

  it("holds every session secret that configures a stack in this repository", () => {
    // The demo compose file, the Makefile's gates and the dashboard's own
    // files. The README and `docs/` are not read: prose that quotes a value
    // starts no server with it.
    const compose = secretsIn(join(repository, "docker-compose.yml"));
    // The demo file is known to set one; finding none means the pattern is stale.
    expect(compose).not.toHaveLength(0);

    const found = [
      ...compose,
      ...secretsIn(join(repository, "Makefile")),
      // Not this file: it spells out the patterns it looks for.
      ...ownFiles(web)
        .filter((path) => path !== __filename)
        .flatMap(secretsIn),
    ];
    expect(found.filter((secret) => !DEMO_SESSION_SECRETS.has(secret))).toEqual([]);
  });
});
