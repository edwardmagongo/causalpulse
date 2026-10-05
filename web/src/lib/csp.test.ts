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
    expect(contentSecurityPolicy(NONCE, false)).toBe(
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
    expect([...directives(contentSecurityPolicy(NONCE, false)).keys()]).toEqual([
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
      const policy = contentSecurityPolicy(NONCE, development);
      expect(policy.split(NONCE)).toHaveLength(2);
      for (const [name, sources] of directives(policy)) {
        expect(sources.includes(`'nonce-${NONCE}'`), name).toBe(name === "script-src");
      }
    }
  });

  it("allows eval only in development", () => {
    expect(contentSecurityPolicy(NONCE, false)).not.toContain("'unsafe-eval'");
    for (const [name, sources] of directives(contentSecurityPolicy(NONCE, true))) {
      expect(sources.includes("'unsafe-eval'"), name).toBe(name === "script-src");
    }
  });

  it("never allows inline scripts", () => {
    for (const development of [false, true]) {
      expect(directives(contentSecurityPolicy(NONCE, development)).get("script-src")).not.toContain("'unsafe-inline'");
    }
  });

  it("gives two policies for two nonces", () => {
    const one = contentSecurityPolicy("b25l", false);
    const other = contentSecurityPolicy("dHdv", false);
    expect(one).not.toBe(other);
    expect(one.replace("b25l", "dHdv")).toBe(other);
  });
});
