import { describe, expect, it } from "vitest";

import { contentSecurityPolicy } from "./csp";

const NONCE = "bm9uY2UtZm9yLXRoZS10ZXN0";

function directives(policy: string): Map<string, string[]> {
  const found = new Map<string, string[]>();
  for (const directive of policy.split("; ")) {
    const [name, ...sources] = directive.split(" ");
    expect(found.has(name), `${name} appears twice`).toBe(false);
    found.set(name, sources);
  }
  return found;
}

describe("the Content-Security-Policy", () => {
  it("holds each directive of the design exactly once", () => {
    expect(contentSecurityPolicy(NONCE, false, false)).toBe(
      [
        "default-src 'self'",
        `script-src 'self' 'nonce-${NONCE}' 'strict-dynamic'`,
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data:",
        "font-src 'self'",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
      ].join("; "),
    );
    expect([...directives(contentSecurityPolicy(NONCE, false, false)).keys()]).toEqual([
      "default-src",
      "script-src",
      "style-src",
      "img-src",
      "font-src",
      "connect-src",
      "object-src",
      "base-uri",
      "form-action",
      "frame-ancestors",
    ]);
  });

  it("puts the nonce in script-src and nowhere else", () => {
    for (const development of [false, true]) {
      for (const production of [false, true]) {
        const policy = contentSecurityPolicy(NONCE, development, production);
        expect(policy.split(NONCE)).toHaveLength(2);
        for (const [name, sources] of directives(policy)) {
          expect(sources.includes(`'nonce-${NONCE}'`), name).toBe(name === "script-src");
        }
      }
    }
  });

  it("allows eval only in development", () => {
    expect(contentSecurityPolicy(NONCE, false, false)).not.toContain("'unsafe-eval'");
    for (const [name, sources] of directives(contentSecurityPolicy(NONCE, true, false))) {
      expect(sources.includes("'unsafe-eval'"), name).toBe(name === "script-src");
    }
  });

  it("upgrades insecure requests only in production, once, and changes nothing else", () => {
    for (const development of [false, true]) {
      const without = contentSecurityPolicy(NONCE, development, false);
      const withIt = contentSecurityPolicy(NONCE, development, true);
      expect(directives(without).has("upgrade-insecure-requests")).toBe(false);
      // `directives` fails on a name that appears twice; the directive takes no value.
      expect(directives(withIt).get("upgrade-insecure-requests")).toEqual([]);
      expect(withIt).toBe(`${without}; upgrade-insecure-requests`);
      expect(withIt.split("upgrade-insecure-requests")).toHaveLength(2);
    }
  });

  it("never allows inline scripts", () => {
    for (const development of [false, true]) {
      for (const production of [false, true]) {
        expect(directives(contentSecurityPolicy(NONCE, development, production)).get("script-src")).not.toContain(
          "'unsafe-inline'",
        );
      }
    }
  });

  it("gives two policies for two nonces", () => {
    const one = contentSecurityPolicy("b25l", false, false);
    const other = contentSecurityPolicy("dHdv", false, false);
    expect(one).not.toBe(other);
    expect(one.replace("b25l", "dHdv")).toBe(other);
  });
});
