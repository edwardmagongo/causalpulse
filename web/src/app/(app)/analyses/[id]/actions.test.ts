// @vitest-environment node
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));
vi.mock("next/navigation", () => ({
  redirect: (to: string) => {
    throw new Error(`REDIRECT ${to}`);
  },
  unstable_rethrow: () => {},
}));
vi.mock("next/headers", () => ({ headers: async () => new Headers() }));
vi.mock("@/lib/env", () => ({ apiUrl: () => "http://api.test" }));
vi.mock("@/lib/session", () => ({ getSession: async () => ({ key: "cp_test" }) }));

import { askForInterpretation } from "./actions";

type Fetch = (url: string, init?: RequestInit) => Promise<Response>;

function answers(status: number, body: unknown) {
  const stub = vi.fn<Fetch>(
    async () =>
      new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }),
  );
  vi.stubGlobal("fetch", stub);
  return stub;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

// The real client runs under the action: only `fetch` is replaced.
describe("asking for an interpretation", () => {
  it("posts to the analysis's interpretation and answers ok when it is queued", async () => {
    const stub = answers(202, { status: "queued" });
    expect(await askForInterpretation("a1")).toEqual({ ok: true });
    expect(stub.mock.calls[0][0]).toBe("http://api.test/v1/analyses/a1/interpretation");
    expect(stub.mock.calls[0][1]?.method).toBe("POST");
  });

  it("answers with the API's sentence for a 409", async () => {
    const sentence = "This analysis already has an interpretation.";
    answers(409, { detail: sentence });
    expect(await askForInterpretation("a1")).toEqual({ error: sentence });
  });

  it("answers with the API's sentence for a 429", async () => {
    const sentence = "Too many interpretations today for this account. Try again in 3 hours.";
    answers(429, { detail: sentence });
    expect(await askForInterpretation("a1")).toEqual({ error: sentence });
  });

  it("leaves for sign-out on a 401", async () => {
    answers(401, { detail: "invalid key" });
    await expect(askForInterpretation("a1")).rejects.toThrow("REDIRECT /sign-out");
  });
});
