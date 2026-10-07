// @vitest-environment node
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));
vi.mock("next/navigation", () => ({
  notFound: () => {
    throw new Error("NOT_FOUND");
  },
  redirect: (to: string) => {
    throw new Error(`REDIRECT ${to}`);
  },
  // As Next.js does: an error of its own carries a digest, and is thrown again.
  unstable_rethrow: (error: unknown) => {
    if (typeof error === "object" && error !== null && "digest" in error) throw error;
  },
}));
// The incoming request's headers; `null` is a call made where there is no request,
// and an `Error` is what reading them throws.
const incoming = vi.hoisted(() => ({ headers: null as Record<string, string> | Error | null }));
vi.mock("next/headers", () => ({
  headers: async () => {
    if (incoming.headers === null) throw new Error("`headers` was called outside a request scope.");
    if (incoming.headers instanceof Error) throw incoming.headers;
    return new Headers(incoming.headers);
  },
}));
vi.mock("@/lib/env", () => ({ apiUrl: () => "http://api.test" }));
vi.mock("@/lib/session", () => ({ getSession: async () => ({ key: "cp_test" }) }));

import { GET as getAnalysisRoute } from "@/app/api/analyses/[id]/route";
import { GET as getManifestRoute } from "@/app/api/analyses/[id]/manifest/route";
import { GET as getReportRoute } from "@/app/api/analyses/[id]/estimates/[method]/report.md/route";
import { POST as postDatasetRoute } from "@/app/api/datasets/route";

import {
  READ_DEADLINE_MS,
  RELAY_DEADLINE_MS,
  UPLOAD_DEADLINE_MS,
  forward,
  getReportHtml,
  listAnalyses,
  me,
} from "./client";
import { ApiError, ApiUnavailable } from "./errors";
import { load } from "./load";
import { relay } from "./relay";

/** A test that waits on a deadline fails fast if the deadline never comes. */
const FAST = 1_000;

type Fetch = (url: string, init?: RequestInit) => Promise<Response>;

/** An API that accepts the call and never answers: it settles only when its signal aborts. */
function neverAnswers() {
  const stub = vi.fn<Fetch>(
    (_url, init) =>
      new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal;
        if (!signal) return;
        if (signal.aborted) return reject(signal.reason);
        signal.addEventListener("abort", () => reject(signal.reason), { once: true });
      }),
  );
  vi.stubGlobal("fetch", stub);
  return stub;
}

/** An API that sends its status and the start of a body, then nothing more until its signal aborts. */
function stallsMidBody(status = 200) {
  const stub = vi.fn<Fetch>(async (_url, init) => {
    const signal = init?.signal;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('{"items": ['));
        signal?.addEventListener("abort", () => controller.error(signal.reason), { once: true });
      },
    });
    return new Response(body, { status, headers: { "content-type": "application/json" } });
  });
  vi.stubGlobal("fetch", stub);
  return stub;
}

function answers(make: () => Response) {
  const stub = vi.fn<Fetch>(async () => make());
  vi.stubGlobal("fetch", stub);
  return stub;
}

/**
 * Record the deadline each call asks for, and hand back a signal that fires
 * after `ms` instead, so a 30 s deadline is reached without waiting 30 s.
 */
function shortenDeadlines(ms = 5) {
  const real = AbortSignal.timeout.bind(AbortSignal);
  return vi.spyOn(AbortSignal, "timeout").mockImplementation(() => real(ms));
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  incoming.headers = null;
});

describe("a call that outlives its deadline", () => {
  it("throws ApiUnavailable from a relayed call", { timeout: FAST }, async () => {
    neverAnswers();
    await expect(forward("/v1/analyses/a", {}, 5)).rejects.toBeInstanceOf(ApiUnavailable);
  });

  it("is aborted with the time-out's own reason", { timeout: FAST }, async () => {
    const stub = neverAnswers();
    await expect(forward("/v1/analyses/a", {}, 5)).rejects.toBeInstanceOf(ApiUnavailable);
    const signal = stub.mock.calls[0][1]?.signal;
    expect(signal?.aborted).toBe(true);
    expect((signal?.reason as Error).name).toBe("TimeoutError");
  });

  it("throws ApiUnavailable from a typed call", { timeout: FAST }, async () => {
    neverAnswers();
    shortenDeadlines();
    await expect(listAnalyses()).rejects.toBeInstanceOf(ApiUnavailable);
    await expect(me("cp_test")).rejects.toBeInstanceOf(ApiUnavailable);
  });

  it("reaches a page as the unavailable state", { timeout: FAST }, async () => {
    neverAnswers();
    shortenDeadlines();
    expect(await load(() => listAnalyses())).toEqual({ ok: false, reason: "unavailable" });
  });

  it("is answered 503 by a relay", { timeout: FAST }, async () => {
    neverAnswers();
    const response = await relay("/v1/analyses/a", {}, {}, 5);
    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ detail: "the service is unavailable" });
  });
});

describe("a typed call whose body stalls past its deadline", () => {
  it("throws ApiUnavailable from a JSON read", { timeout: FAST }, async () => {
    stallsMidBody();
    shortenDeadlines();
    await expect(listAnalyses()).rejects.toBeInstanceOf(ApiUnavailable);
  });

  it("reaches a page as the unavailable state", { timeout: FAST }, async () => {
    stallsMidBody();
    shortenDeadlines();
    expect(await load(() => listAnalyses())).toEqual({ ok: false, reason: "unavailable" });
  });

  it("throws ApiUnavailable from the HTML report's text read", { timeout: FAST }, async () => {
    stallsMidBody();
    shortenDeadlines();
    await expect(getReportHtml("a", "ipw")).rejects.toBeInstanceOf(ApiUnavailable);
    expect(await load(() => getReportHtml("a", "ipw"))).toEqual({ ok: false, reason: "unavailable" });
  });

  it("throws ApiUnavailable when it is a refusal's body that stalls", { timeout: FAST }, async () => {
    stallsMidBody(500);
    shortenDeadlines();
    await expect(listAnalyses()).rejects.toBeInstanceOf(ApiUnavailable);
  });
});

describe("a body that arrives in time but is not JSON", () => {
  const malformed = () => new Response('{"items": [', { status: 200 });

  it("is not called unavailable: the parse error propagates", async () => {
    answers(malformed);
    const error: unknown = await listAnalyses().catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(SyntaxError);
    expect(error).not.toBeInstanceOf(ApiUnavailable);
  });

  it("is not turned into the unavailable state for a page", async () => {
    answers(malformed);
    await expect(load(() => listAnalyses())).rejects.toBeInstanceOf(SyntaxError);
  });

  it("leaves a refusal with an unreadable body as the refusal it is", async () => {
    answers(() => new Response("<html>oops", { status: 500 }));
    const error: unknown = await listAnalyses().catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(500);
  });
});

describe("a call that answers in time", () => {
  it("returns the API's response untouched, having been given an AbortSignal", async () => {
    const upstream = Response.json({ id: "a" });
    const stub = answers(() => upstream);
    expect(await forward("/v1/analyses/a")).toBe(upstream);
    expect(stub.mock.calls[0][1]?.signal).toBeInstanceOf(AbortSignal);
    expect(stub.mock.calls[0][1]?.signal?.aborted).toBe(false);
  });

  it("returns a typed call's data, having been given an AbortSignal", async () => {
    const stub = answers(() => Response.json({ items: [], next_cursor: null }));
    expect(await listAnalyses()).toEqual({ items: [], next_cursor: null });
    expect(stub.mock.calls[0][1]?.signal).toBeInstanceOf(AbortSignal);
  });
});

describe("each kind of call has its own deadline", () => {
  const sameSite = { origin: "http://localhost:3000", host: "localhost:3000" };

  it("orders them: a read, then a relay, then the upload", () => {
    expect(READ_DEADLINE_MS).toBeLessThan(RELAY_DEADLINE_MS);
    expect(RELAY_DEADLINE_MS).toBeLessThan(UPLOAD_DEADLINE_MS);
  });

  it("gives a typed call the read deadline", async () => {
    answers(() => Response.json({ items: [], next_cursor: null }));
    const timeout = vi.spyOn(AbortSignal, "timeout");
    await listAnalyses();
    await me("cp_test");
    expect(timeout.mock.calls).toEqual([[READ_DEADLINE_MS], [READ_DEADLINE_MS]]);
  });

  it("gives a relayed call the relay deadline unless told otherwise", async () => {
    answers(() => Response.json({}));
    const timeout = vi.spyOn(AbortSignal, "timeout");
    await forward("/v1/analyses/a");
    await relay("/v1/analyses/a");
    expect(timeout.mock.calls).toEqual([[RELAY_DEADLINE_MS], [RELAY_DEADLINE_MS]]);
  });

  it("gives the three download and polling routes the relay deadline", async () => {
    answers(() => Response.json({}));
    const timeout = vi.spyOn(AbortSignal, "timeout");
    const request = new Request("http://localhost:3000/");
    await getAnalysisRoute(request, { params: Promise.resolve({ id: "a" }) });
    await getManifestRoute(request, { params: Promise.resolve({ id: "a" }) });
    await getReportRoute(request, { params: Promise.resolve({ id: "a", method: "ipw" }) });
    expect(timeout.mock.calls).toEqual([[RELAY_DEADLINE_MS], [RELAY_DEADLINE_MS], [RELAY_DEADLINE_MS]]);
  });

  it("gives the upload route the upload deadline", async () => {
    const stub = answers(() => Response.json({ id: "d" }, { status: 201 }));
    const timeout = vi.spyOn(AbortSignal, "timeout");
    const response = await postDatasetRoute(
      new Request("http://localhost:3000/api/datasets", { method: "POST", headers: sameSite, body: "x" }),
    );
    expect(response.status).toBe(201);
    expect(stub.mock.calls[0][0]).toBe("http://api.test/v1/datasets");
    expect(timeout.mock.calls).toEqual([[UPLOAD_DEADLINE_MS]]);
  });
});

describe("the client's address", () => {
  const sameSite = { origin: "http://localhost:3000", host: "localhost:3000" };
  // What a client can send in its own `X-Forwarded-For`, and what the proxy adds after it.
  const DECOY = "198.51.100.99";
  const PEER = "203.0.113.7";

  /** The address each call to the API carried, in order; `null` where it carried none. */
  function sent(stub: ReturnType<typeof answers>): (string | null)[] {
    return stub.mock.calls.map(([, init]) => new Headers(init?.headers).get("x-cp-client-address"));
  }

  it("goes on every kind of call, and is the entry the proxy added, not the one before it", async () => {
    incoming.headers = { "x-forwarded-for": `${DECOY}, ${PEER}` };
    const stub = answers(() => Response.json({ items: [], next_cursor: null, name: "a" }));
    await listAnalyses();
    await me("cp_other");
    await forward("/v1/analyses/a");
    await relay("/v1/analyses/a");
    await getAnalysisRoute(new Request("http://localhost:3000/"), { params: Promise.resolve({ id: "a" }) });
    await postDatasetRoute(
      new Request("http://localhost:3000/api/datasets", { method: "POST", headers: sameSite, body: "x" }),
    );
    expect(sent(stub)).toEqual([PEER, PEER, PEER, PEER, PEER, PEER]);
  });

  it("is a lone entry as it is, without its spaces", async () => {
    incoming.headers = { "x-forwarded-for": ` ${PEER} ` };
    const stub = answers(() => Response.json({ items: [], next_cursor: null }));
    await listAnalyses();
    expect(sent(stub)).toEqual([PEER]);
  });

  it("is absent when the request has no forwarded address, or an empty last one", async () => {
    const stub = answers(() => Response.json({ items: [], next_cursor: null, name: "a" }));
    const without: Record<string, string>[] = [{}, { "x-forwarded-for": "" }, { "x-forwarded-for": `${DECOY}, ` }];
    for (const headers of without) {
      incoming.headers = headers;
      await listAnalyses();
      await me("cp_other");
      await forward("/v1/analyses/a");
    }
    expect(sent(stub)).toEqual(Array(9).fill(null));
  });

  it("is absent, and the call still made, where there is no request to read", async () => {
    incoming.headers = null;
    const stub = answers(() => Response.json({ items: [], next_cursor: null }));
    expect(await listAnalyses()).toEqual({ items: [], next_cursor: null });
    expect(sent(stub)).toEqual([null]);
  });

  it("lets a signal of Next.js's own through as it is, and makes no call", async () => {
    const signal = Object.assign(new Error("dynamic server usage"), { digest: "DYNAMIC_SERVER_USAGE" });
    incoming.headers = signal;
    const stub = answers(() => Response.json({ items: [], next_cursor: null }));
    await expect(listAnalyses()).rejects.toBe(signal);
    await expect(forward("/v1/analyses/a")).rejects.toBe(signal);
    expect(stub).not.toHaveBeenCalled();
  });

  it("cannot be set by a caller, in a header of any case or in the body", async () => {
    const stub = answers(() => Response.json({}));
    const claims: RequestInit[] = [
      { headers: { "X-CP-Client-Address": DECOY } },
      { headers: { "x-cp-client-address": DECOY } },
      { headers: new Headers({ "X-Cp-Client-Address": DECOY }) },
      { method: "POST", body: JSON.stringify({ client_address: DECOY, "x-cp-client-address": DECOY }) },
    ];
    for (const init of claims) await forward("/v1/analyses", init);
    expect(sent(stub)).toEqual([null, null, null, null]);

    incoming.headers = { "x-forwarded-for": PEER };
    for (const init of claims) await forward("/v1/analyses", init);
    expect(sent(stub).slice(4)).toEqual([PEER, PEER, PEER, PEER]);
  });

  it("is not taken from what the browser sends the upload route", async () => {
    incoming.headers = { "x-forwarded-for": `${DECOY}, ${PEER}` };
    const stub = answers(() => Response.json({ id: "d" }, { status: 201 }));
    await postDatasetRoute(
      new Request("http://localhost:3000/api/datasets", {
        method: "POST",
        headers: { ...sameSite, "x-cp-client-address": DECOY },
        body: DECOY,
      }),
    );
    expect(sent(stub)).toEqual([PEER]);
  });

  it("leaves the key and the caller's other headers as they were", async () => {
    incoming.headers = { "x-forwarded-for": PEER };
    const stub = answers(() => Response.json({}));
    await forward("/v1/analyses", { method: "POST", headers: { "content-type": "application/json" } });
    const headers = new Headers(stub.mock.calls[0][1]?.headers);
    expect(headers.get("authorization")).toBe("Bearer cp_test");
    expect(headers.get("content-type")).toBe("application/json");
  });
});

describe("a call the API refuses with 429", () => {
  const SENTENCE = "too many failed sign-ins from this address; try again later";
  const limited = () =>
    new Response(JSON.stringify({ detail: SENTENCE }), {
      status: 429,
      headers: { "content-type": "application/json", "retry-after": "840" },
    });

  it("is an ApiError holding the API's sentence, from a typed call", async () => {
    answers(limited);
    const error: unknown = await me("cp_other").catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(429);
    expect((error as ApiError).detail).toBe(SENTENCE);
  });

  it("reaches the browser from the upload route with its status and its sentence", async () => {
    answers(limited);
    const response = await postDatasetRoute(
      new Request("http://localhost:3000/api/datasets", {
        method: "POST",
        headers: { origin: "http://localhost:3000", host: "localhost:3000" },
        body: "x",
      }),
    );
    expect(response.status).toBe(429);
    expect(await response.json()).toEqual({ detail: SENTENCE });
  });
});
