// @vitest-environment node
import { readdirSync, readFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";

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

// --- the list is complete ---------------------------------------------------
//
// The scanner reads the files that configure a stack someone runs: every
// `docker-compose*.yml` at the root, the Makefile's gates, and the dashboard's
// own files. The README, `docs/` and Markdown under `web/` are not read: prose
// that quotes a value starts no server with it.

type Kind = "compose" | "makefile" | "code" | "plain";

function kindOf(path: string): Kind {
  const name = basename(path);
  if (/\.ya?ml$/.test(name)) return "compose";
  if (name === "Makefile" || name.endsWith(".mk")) return "makefile";
  if (/\.(?:[cm]?[jt]sx?|json)$/.test(name)) return "code";
  return "plain";
}

// One piece of a value: `${...}`, `$(...)`, `$$(...)`.
const PIECE = String.raw`\$\{[^}]*\}|\$\$?\([^)]*\)`;

// Every form that can set the variable in a file:
//   CP_SESSION_SECRET: value             a compose mapping, an object literal
//   "CP_SESSION_SECRET": "value"         JSON
//   - CP_SESSION_SECRET=value            a compose list, a script, an env file
//   process.env.CP_SESSION_SECRET = "v"  and `??=`, `||=`, `env["..."] = "v"`
//   stubEnv("CP_SESSION_SECRET", "v")    and any other `name, value` pair
// but not a comparison (`===`), nor the name inside `${CP_SESSION_SECRET:-v}`,
// which is read as the value of whatever it is assigned to.
const ASSIGNMENT = new RegExp(
  String.raw`(?<![$\{\w])CP_SESSION_SECRET["'\`]?\]?[ \t]*(\?\?=|\|\|=|:|=(?!=)|,)[ \t]*` +
    String.raw`("[^"\n]*"|'[^'\n]*'|\`[^\`\n]*\`|(?:${PIECE}|[^\s"'\`,;)#])*)`,
  "g",
);

function makeVariables(text: string): Map<string, string> {
  const found = [...text.matchAll(/^(\w+)[ \t]*(?:::?=|\?=|=)[ \t]*(.*?)[ \t]*$/gm)];
  return new Map(found.map((match) => [match[1], match[2]]));
}

/**
 * The literal a value holds, or null if it holds none.
 *
 * Null is a reference the file does not fill in: `${NAME}`, `${NAME:?why}`,
 * `$NAME`, a command's output. `${NAME:-x}` and `${NAME-x}` hold `x`: that is
 * what the stack runs with when nobody sets `NAME`, which is how a new demo
 * value would arrive. Anything else with a `$` in it is an error: the scanner
 * cannot tell, and says so.
 */
function literal(token: string, variables?: Map<string, string>): string | null {
  if (!token.includes("$")) return token;
  if (variables) {
    if (token.startsWith("$$")) return null; // the recipe's shell fills it in
    const reference = /^\$[({](\w+)[)}]$/.exec(token);
    if (reference) {
      const value = variables.get(reference[1]);
      if (value === undefined || value.includes("$")) {
        throw new Error(`cannot tell: ${token} is not defined as a plain NAME := value`);
      }
      return value;
    }
    if (/^\$\(\w+\s[^)]*\)$/.test(token)) return null; // `$(shell ...)`: its output
    throw new Error(`cannot tell what ${token} holds`);
  }
  const braced = /^\$\{(\w+)(?:(:?[-?+])([\s\S]*))?\}$/.exec(token);
  if (braced) {
    const [, , operator, rest] = braced;
    if (operator === undefined || operator === "?" || operator === ":?") return null;
    if (operator === "-" || operator === ":-") return literal(rest);
    throw new Error(`cannot tell what ${token} holds`);
  }
  if (/^\$\w+$/.test(token) || /^\$\$?\([^)]*\)$/.test(token)) return null;
  throw new Error(`cannot tell what ${token} holds`);
}

/** Every literal session secret `text` sets, in the order it sets them. */
function sessionSecrets(text: string, kind: Kind): string[] {
  const variables = kind === "makefile" ? makeVariables(text) : undefined;
  const found: string[] = [];
  for (const [, operator, raw] of text.matchAll(ASSIGNMENT)) {
    const quoted = raw.length >= 2 && "\"'`".includes(raw[0]) && raw.endsWith(raw[0]);
    const inner = quoted ? raw.slice(1, -1) : raw;
    if (kind === "code") {
      // In code only a string is a literal: a bare value is an expression or
      // a type, and a template with `${` in it is computed.
      if (quoted && !(raw[0] === "`" && inner.includes("${"))) found.push(inner);
      continue;
    }
    // Outside code, `NAME, something` is a list in prose or a comment.
    if (operator === ",") continue;
    const value = literal(inner, variables);
    if (value !== null) found.push(value);
  }
  return found;
}

describe("the scanner for session secrets", () => {
  const S = "CP_SESSION_SECRET";

  it.each<[string, string[]]>([
    // A compose file's mapping form, and its list form.
    [`      ${S}: plain-value`, ["plain-value"]],
    [`      ${S}: "quoted value"  # a comment`, ["quoted value"]],
    [`      - ${S}=listed`, ["listed"]],
    [`      - "${S}=listed"`, ["listed"]],
    // A default is a literal; a reference with none is not.
    [`      ${S}: \${${S}:-another-demo-secret}`, ["another-demo-secret"]],
    [`      ${S}: \${${S}-another-demo-secret}`, ["another-demo-secret"]],
    [`      ${S}: "\${${S}:-another-demo-secret}"`, ["another-demo-secret"]],
    [`      ${S}: \${OUTER:-\${INNER:-nested}}`, ["nested"]],
    [`      ${S}: \${${S}}`, []],
    [`      ${S}: $${S}`, []],
    [`      ${S}: \${${S}:?set a session secret first}`, []],
    [`      ${S}: \${${S}?set a session secret first}`, []],
    [`      - ${S}`, []],
    [`      # ${S}, CP_API_URL and the limits come from the environment file`, []],
  ])("reads the compose line %s", (line, expected) => {
    expect(sessionSecrets(line, "compose")).toEqual(expected);
  });

  it.each([
    `      ${S}: prefix\${SUFFIX}`,
    `      ${S}: \${${S}:+alternative}`,
  ])("fails on the compose value %s, which it cannot read", (line) => {
    expect(() => sessionSecrets(line, "compose")).toThrow(/cannot tell/);
  });

  it.each<[string, string[]]>([
    [`\t${S}=plain-value pnpm run e2e`, ["plain-value"]],
    [`SECRET := followed\nSTACK = ${S}=$(SECRET) docker compose`, ["followed"]],
    [`SECRET ?= followed\nSTACK = ${S}=\${SECRET} docker compose`, ["followed"]],
    // The recipe's shell fills these in when the gate runs.
    [`\t${S}=$$(openssl rand -hex 32) docker compose up`, []],
    [`\t${S}=$$SECRET docker compose up`, []],
    [`STACK = ${S}=$(shell openssl rand -hex 32) docker compose`, []],
  ])("reads the Makefile text %s", (text, expected) => {
    expect(sessionSecrets(text, "makefile")).toEqual(expected);
  });

  it.each([
    `STACK = ${S}=$(SECRET) docker compose`, // defined nowhere
    `SECRET := $(shell openssl rand -hex 32)\nSTACK = ${S}=$(SECRET) docker compose`,
  ])("fails on the Makefile variable in %s, which it cannot follow", (text) => {
    expect(() => sessionSecrets(text, "makefile")).toThrow(/cannot tell/);
  });

  it.each<[string, string[]]>([
    [`vi.stubEnv("${S}", "stubbed-value");`, ["stubbed-value"]],
    [`vi.stubEnv('${S}', 'stubbed-value');`, ["stubbed-value"]],
    [`process.env.${S} = "assigned-value";`, ["assigned-value"]],
    [`process.env.${S} ??= "defaulted-value";`, ["defaulted-value"]],
    [`process.env.${S} ||= "defaulted-value";`, ["defaulted-value"]],
    [`process.env["${S}"] = "indexed-value";`, ["indexed-value"]],
    [`  env: { ${S}: "object-value", CP_API_URL: url },`, ["object-value"]],
    [`  env: { "${S}": 'object-value' },`, ["object-value"]],
    [`  "${S}": "json-value"`, ["json-value"]],
    ["  " + S + ": `template-value`,", ["template-value"]],
    // Not literals: an expression, a type, a computed template, a comparison.
    [`vi.stubEnv("${S}", STRONG);`, []],
    [`  env: { ${S}: secret },`, []],
    [`  ${S}: string;`, []],
    ["  " + S + ": `${prefix}-computed`,", []],
    [`if (process.env.${S} === "compared-value") {}`, []],
    [`const secret = process.env.${S};`, []],
    [`throw new Error("${S} must be set to at least 32 characters");`, []],
  ])("reads the line of code %s", (line, expected) => {
    expect(sessionSecrets(line, "code")).toEqual(expected);
  });

  it.each<[string, string[]]>([
    [`${S}=env-file-value`, ["env-file-value"]],
    [`export ${S}="script-value"`, ["script-value"]],
    [`ENV ${S}=dockerfile-value PORT=3000`, ["dockerfile-value"]],
    [`${S}=`, [""]],
    [`${S}="\${${S}:-shell-default}"`, ["shell-default"]],
    [`${S}=$(openssl rand -hex 32)`, []],
  ])("reads the script or environment line %s", (line, expected) => {
    expect(sessionSecrets(line, "plain")).toEqual(expected);
  });

  it("tells the kinds of file apart", () => {
    expect(kindOf("/r/docker-compose.prod.yml")).toBe("compose");
    expect(kindOf("/r/Makefile")).toBe("makefile");
    expect(kindOf("/r/web/playwright.config.ts")).toBe("code");
    expect(kindOf("/r/web/eslint.config.mjs")).toBe("code");
    expect(kindOf("/r/web/package.json")).toBe("code");
    expect(kindOf("/r/web/.env.example")).toBe("plain");
    expect(kindOf("/r/web/Dockerfile")).toBe("plain");
  });
});

describe("DEMO_SESSION_SECRETS", () => {
  const web = resolve(__dirname, "../..");
  const repository = resolve(web, "..");

  /** Every file under `web/` that is the project's own: not installed, not built, not prose. */
  function ownFiles(directory: string): string[] {
    return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
      if (["node_modules", ".next", "test-results", "playwright-report"].includes(entry.name)) {
        return [];
      }
      const path = join(directory, entry.name);
      if (entry.isDirectory()) return ownFiles(path);
      return entry.name.endsWith(".md") ? [] : [path];
    });
  }

  function secretsIn(path: string): string[] {
    let text: string;
    try {
      text = readFileSync(path, "utf-8");
    } catch {
      return [];
    }
    return sessionSecrets(text, kindOf(path));
  }

  const composeFiles = readdirSync(repository)
    .filter((name) => /^docker-compose.*\.ya?ml$/.test(name))
    .sort();

  it("is not empty, and holds the demo compose file's secret", () => {
    // The tests above that run once per listed secret run zero times for an
    // empty list; this one does not.
    const demo = secretsIn(join(repository, "docker-compose.yml"));
    expect(demo).toHaveLength(1);
    expect(DEMO_SESSION_SECRETS.size).toBeGreaterThan(0);
    expect(DEMO_SESSION_SECRETS.has(demo[0])).toBe(true);
  });

  it("holds every session secret that configures a stack in this repository", () => {
    expect(composeFiles).toContain("docker-compose.yml");
    const found = [
      ...composeFiles.flatMap((name) => secretsIn(join(repository, name))),
      ...secretsIn(join(repository, "Makefile")),
      // Not this file: it spells out the forms it looks for.
      ...ownFiles(web)
        .filter((path) => path !== __filename)
        .flatMap(secretsIn),
    ];
    expect(found.filter((secret) => !DEMO_SESSION_SECRETS.has(secret))).toEqual([]);
  });
});
