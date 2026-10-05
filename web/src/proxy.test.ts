import { NextRequest } from "next/server";
import { describe, expect, it } from "vitest";

import { contentSecurityPolicy } from "@/lib/csp";
import { SESSION_COOKIE } from "@/lib/session-cookie";

import { config, proxy } from "./proxy";

// Next.js anchors a matcher to the whole path; do the same here.
const matched = new RegExp(`^${config.matcher[0]}$`);

const ORIGIN = "http://localhost:3000";

function request(path: string, signedIn: boolean): NextRequest {
  return new NextRequest(`${ORIGIN}${path}`, signedIn ? { headers: { cookie: `${SESSION_COOKIE}=sealed` } } : undefined);
}

/** The response's policy and the nonce in its `script-src`, checked to be the whole policy for that nonce. */
function policyOf(response: Response): { policy: string; nonce: string } {
  const policy = response.headers.get("content-security-policy") ?? "";
  const nonce = /script-src [^;]*'nonce-([^']+)'/.exec(policy)?.[1] ?? "";
  expect(nonce).not.toBe("");
  expect(policy).toBe(contentSecurityPolicy(nonce, false));
  return { policy, nonce };
}

describe("the proxy's matcher", () => {
  it("matches every page, sign-in included, and paths that only start like an exclusion", () => {
    for (const path of [
      "/",
      "/analyses/new",
      "/analyses/abc",
      "/study",
      "/sign-in",
      "/sign-in/",
      "/sign-in/again",
      "/sign-in.rsc",
      "/api-keys",
      "/apixyz",
      "/sign-insider",
      "/sign-outside",
    ]) {
      expect(matched.test(path), path).toBe(true);
    }
  });

  it("leaves out the relay handlers, the sign-out route, and static files", () => {
    for (const path of [
      "/api",
      "/api/datasets",
      "/api/analyses/abc/report",
      "/sign-out",
      "/sign-out/now",
      "/_next/static/x.js",
      "/_next/image",
      "/favicon.ico",
    ]) {
      expect(matched.test(path), path).toBe(false);
    }
  });
});

describe("the proxy", () => {
  it("redirects a page without a session cookie to sign-in, under the policy", () => {
    for (const path of ["/", "/analyses/new", "/study", "/sign-insider"]) {
      const response = proxy(request(path, false));
      expect(response.status, path).toBe(307);
      expect(response.headers.get("location"), path).toBe(`${ORIGIN}/sign-in`);
      policyOf(response);
    }
  });

  it("lets sign-in through without a session cookie, under the policy", () => {
    for (const path of ["/sign-in", "/sign-in/again"]) {
      const response = proxy(request(path, false));
      expect(response.status, path).toBe(200);
      expect(response.headers.get("location"), path).toBeNull();
      expect(response.headers.get("x-middleware-next"), path).toBe("1");
      policyOf(response);
    }
  });

  it("lets a page through with a session cookie, under the policy", () => {
    const response = proxy(request("/analyses/new", true));
    expect(response.status).toBe(200);
    expect(response.headers.get("location")).toBeNull();
    expect(response.headers.get("x-middleware-next")).toBe("1");
    policyOf(response);
  });

  it("hands the page the same policy and nonce it answers with", () => {
    const response = proxy(request("/", true));
    const { policy, nonce } = policyOf(response);
    expect(response.headers.get("x-middleware-request-content-security-policy")).toBe(policy);
    expect(response.headers.get("x-middleware-request-x-nonce")).toBe(nonce);
  });

  it("makes a new nonce for every request", () => {
    const first = policyOf(proxy(request("/", true)));
    const second = policyOf(proxy(request("/", true)));
    expect(first.nonce).not.toBe(second.nonce);
    expect(first.policy).not.toBe(second.policy);
  });
});
