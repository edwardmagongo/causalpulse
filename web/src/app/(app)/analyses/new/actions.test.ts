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

import type { SchemaJson } from "@/lib/schema-form";

import { submitAnalysis } from "./actions";

type Fetch = (url: string, init?: RequestInit) => Promise<Response>;

function answers(status: number, body: unknown, headers: Record<string, string> = {}) {
  const stub = vi.fn<Fetch>(
    async () =>
      new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } }),
  );
  vi.stubGlobal("fetch", stub);
  return stub;
}

const REQUEST = { datasetId: "d1", schema: {} as SchemaJson, estimators: null, seed: 0 };

afterEach(() => {
  vi.unstubAllGlobals();
});

// The real client runs under the action: only `fetch` is replaced.
describe("submitting an analysis", () => {
  it("answers with the API's own sentence when the user is over a limit", async () => {
    const sentence = "3 analyses are already queued or running; wait for one to finish";
    const stub = answers(429, { detail: sentence }, { "retry-after": "60" });
    expect(await submitAnalysis(REQUEST)).toEqual({ error: sentence });
    expect(stub).toHaveBeenCalledTimes(1);
    expect(stub.mock.calls[0][0]).toBe("http://api.test/v1/analyses");
  });

  it("goes to the analysis when it is accepted", async () => {
    answers(202, { id: "a1" });
    await expect(submitAnalysis(REQUEST)).rejects.toThrow("REDIRECT /analyses/a1");
  });

  it("leaves for sign-out on a 401", async () => {
    answers(401, { detail: "missing or invalid API key" });
    await expect(submitAnalysis(REQUEST)).rejects.toThrow("REDIRECT /sign-out");
  });
});
