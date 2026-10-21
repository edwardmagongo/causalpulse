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
    // No entry named: no body, and the API takes the default.
    expect(stub.mock.calls[0][1]?.body).toBeUndefined();
  });

  it("posts the chosen model and effort as the request's JSON body", async () => {
    const stub = answers(202, { status: "queued" });
    const choice = { model: "claude-sonnet-5-5", effort: "low" };
    expect(await askForInterpretation("a1", choice)).toEqual({ ok: true });
    const init = stub.mock.calls[0][1]!;
    expect(init.method).toBe("POST");
    expect(new Headers(init.headers).get("content-type")).toBe("application/json");
    expect(JSON.parse(init.body as string)).toEqual(choice);
  });

  it("sends only the model and the effort of what it is given", async () => {
    const stub = answers(202, { status: "queued" });
    const choice = { model: "claude-sonnet-5-5", effort: "low", default: false, available: true };
    await askForInterpretation("a1", choice);
    expect(JSON.parse(stub.mock.calls[0][1]!.body as string)).toEqual({
      model: "claude-sonnet-5-5",
      effort: "low",
    });
  });

  it.each([
    "This analysis already has an interpretation from this model and effort, written under the current prompt and rules.",
    "This analysis's interpretation from this model and effort could not be verified under the current prompt and rules, and asking again would not change them.",
    "An interpretation of this analysis is already being written.",
  ])("answers with the API's sentence for a 409: %s", async (sentence) => {
    answers(409, { detail: sentence });
    expect(await askForInterpretation("a1", { model: "claude-opus-5-5", effort: "medium" })).toEqual({
      error: sentence,
    });
  });

  it.each([
    "Name both the model and the effort, or neither.",
    "That model and effort are not offered on this server.",
  ])("answers with the API's sentence for a 422: %s", async (sentence) => {
    answers(422, { detail: sentence });
    expect(await askForInterpretation("a1", { model: "claude-haiku-5-5", effort: "max" })).toEqual({
      error: sentence,
    });
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
