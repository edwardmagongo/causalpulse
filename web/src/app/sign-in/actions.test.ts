// @vitest-environment node
import { afterEach, describe, expect, it, vi } from "vitest";

const saved = vi.hoisted(() => ({ session: { save: vi.fn(), destroy: vi.fn() } as Record<string, unknown> }));

vi.mock("server-only", () => ({}));
vi.mock("next/navigation", () => ({
  redirect: (to: string) => {
    throw new Error(`REDIRECT ${to}`);
  },
  unstable_rethrow: () => {},
}));
vi.mock("next/headers", () => ({ headers: async () => new Headers({ "x-forwarded-for": "203.0.113.7" }) }));
vi.mock("@/lib/env", () => ({ apiUrl: () => "http://api.test" }));
vi.mock("@/lib/session", () => ({ getSession: async () => saved.session }));

import { signIn } from "./actions";

type Fetch = (url: string, init?: RequestInit) => Promise<Response>;

function answers(status: number, body: unknown, headers: Record<string, string> = {}) {
  const stub = vi.fn<Fetch>(
    async () =>
      new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } }),
  );
  vi.stubGlobal("fetch", stub);
  return stub;
}

function form(key: string): FormData {
  const data = new FormData();
  data.set("key", key);
  return data;
}

afterEach(() => {
  vi.unstubAllGlobals();
  saved.session = { save: vi.fn(), destroy: vi.fn() };
});

// The real client runs under the action: only `fetch` is replaced.
describe("signing in", () => {
  it("shows the API's own sentence when the address is over the limit, and starts no session", async () => {
    const sentence = "Too many failed sign-ins from this address. Try again in 14 minutes.";
    const stub = answers(429, { detail: sentence }, { "retry-after": "840" });
    expect(await signIn({ error: null }, form("cpk_AAAAAAAAAAAA_right"))).toEqual({ error: sentence });
    expect(stub).toHaveBeenCalledTimes(1);
    expect(saved.session.key).toBeUndefined();
    expect(saved.session.save).not.toHaveBeenCalled();
  });

  it("shows the API's message for a wrong key", async () => {
    answers(401, { detail: "missing or invalid API key" });
    expect(await signIn({ error: null }, form("cpk_AAAAAAAAAAAA_wrong"))).toEqual({
      error: "missing or invalid API key",
    });
    expect(saved.session.save).not.toHaveBeenCalled();
  });

  it("checks the key the form holds, from the address the proxy saw", async () => {
    const stub = answers(200, { name: "Ada" });
    await expect(signIn({ error: null }, form(" cpk_AAAAAAAAAAAA_right "))).rejects.toThrow("REDIRECT /");
    const [url, init] = stub.mock.calls[0];
    expect(url).toBe("http://api.test/v1/me");
    const headers = new Headers(init?.headers);
    expect(headers.get("authorization")).toBe("Bearer cpk_AAAAAAAAAAAA_right");
    expect(headers.get("x-cp-client-address")).toBe("203.0.113.7");
    expect(saved.session).toMatchObject({ key: "cpk_AAAAAAAAAAAA_right", name: "Ada" });
    expect(saved.session.save).toHaveBeenCalledTimes(1);
  });

  it("sends nothing for an empty key", async () => {
    const stub = answers(200, { name: "Ada" });
    expect(await signIn({ error: null }, form("  "))).toEqual({ error: "Enter an API key." });
    expect(stub).not.toHaveBeenCalled();
  });
});
