import "server-only";

import { headers } from "next/headers";
import { unstable_rethrow } from "next/navigation";

import { apiUrl } from "@/lib/env";
import { getSession } from "@/lib/session";

import { ApiUnavailable, Unauthorized, errorFor } from "./errors";
import type {
  AnalysisCreated,
  AnalysisDetail,
  AnalysisPage,
  AnalysisRequest,
  DatasetDetail,
  DatasetPage,
  EstimatorEntry,
  InterpretationChoice,
  InterpretationDetail,
  Me,
  Summary,
} from "./types";

/**
 * The one module that calls cp-api.
 *
 * `forward` returns the API's response as it is, for the relay handlers,
 * which pass a refusal on to the browser. Every typed function below goes
 * through `ask`, which turns a refusal into one of the three errors in
 * `./errors`.
 *
 * Every call has a deadline, which keeps running while the body arrives. A
 * typed call that outlives it, waiting for the response or reading it, fails
 * as a refused connection does, with `ApiUnavailable`. So does a relayed call
 * still waiting for the response; one whose download stalls is cut off.
 *
 * Every call carries the address of the browser it is made for, which the
 * API counts failed sign-ins against (it believes the header in production
 * only). `send` sets it, and nothing a caller passes can.
 */

const CLIENT_ADDRESS_HEADER = "x-cp-client-address";

/**
 * The browser's address, as the reverse proxy in front of this server saw it.
 *
 * A proxy puts the peer it accepted the connection from at the end of
 * `X-Forwarded-For`, after anything the client sent there that it kept. With
 * one proxy in front, the last entry is the only one the client cannot have
 * chosen, so it is the one taken: going by the first would let a client name
 * a new address for each attempt and never reach the limit. With no proxy,
 * Next.js fills an absent header in with the connection's own peer.
 *
 * `null` when there is no such entry, or no request to read one from.
 */
async function clientAddress(): Promise<string | null> {
  let forwarded: string | null;
  try {
    forwarded = (await headers()).get("x-forwarded-for");
  } catch (error) {
    // Next.js's own signals (a page being rendered ahead of any request) are not ours to swallow.
    unstable_rethrow(error);
    return null;
  }
  const entries = (forwarded ?? "").split(",");
  return entries[entries.length - 1].trim() || null;
}

/** A typed call: a small JSON request, and none streams a file. */
export const READ_DEADLINE_MS = 30_000;

/** A relayed call: a manifest or a Markdown report. */
export const RELAY_DEADLINE_MS = 60_000;

/**
 * The upload. Its body is streamed from the browser, so the deadline covers
 * the time the browser takes to send the file: 50 MB at 1 Mbit/s is about
 * seven minutes.
 */
export const UPLOAD_DEADLINE_MS = 600_000;

async function sessionKey(): Promise<string> {
  const { key } = await getSession();
  if (!key) throw new Unauthorized();
  return key;
}

// A caller's own `init.signal` is replaced by the deadline's; no caller passes one.
async function send(
  path: string,
  init: RequestInit,
  key: string,
  deadline: AbortSignal,
): Promise<Response> {
  const sent = new Headers(init.headers);
  // Removed first: a name in another case, or a request with no address, must not let a caller's value through.
  sent.delete(CLIENT_ADDRESS_HEADER);
  const address = await clientAddress();
  if (address) sent.set(CLIENT_ADDRESS_HEADER, address);
  sent.set("authorization", `Bearer ${key}`);
  try {
    return await fetch(`${apiUrl()}${path}`, {
      ...init,
      cache: "no-store",
      signal: deadline,
      headers: sent,
    });
  } catch {
    throw new ApiUnavailable();
  }
}

/**
 * The API's own response, 4xx included. Throws only for 401, 503, no
 * connection and no answer within the deadline.
 */
export async function forward(
  path: string,
  init: RequestInit = {},
  deadlineMs: number = RELAY_DEADLINE_MS,
): Promise<Response> {
  const response = await send(path, init, await sessionKey(), AbortSignal.timeout(deadlineMs));
  if (response.status === 401) throw new Unauthorized();
  if (response.status === 503) throw new ApiUnavailable();
  return response;
}

/**
 * Read a response's body, which the deadline can still cut off.
 *
 * A read that fails after the deadline has passed is the API not answering,
 * so it is `ApiUnavailable`. A read that fails before it (a body that arrived
 * and is not JSON) is something else and is rethrown as it is. The signal's
 * `aborted` state tells the two apart; the error thrown does not, reliably.
 */
async function received<T>(deadline: AbortSignal, read: () => Promise<T>): Promise<T> {
  try {
    return await read();
  } catch (error) {
    if (deadline.aborted) throw new ApiUnavailable();
    throw error;
  }
}

/** One typed call: the body of a 2xx as `read` takes it, or the error a refusal stands for. */
async function ask<T>(
  path: string,
  read: (response: Response) => Promise<T>,
  init: RequestInit = {},
  key?: string,
): Promise<T> {
  const deadline = AbortSignal.timeout(READ_DEADLINE_MS);
  const response = await send(path, init, key ?? (await sessionKey()), deadline);
  if (response.ok) return received(deadline, () => read(response));
  // A refusal whose body cannot be parsed is still the refusal; one cut off is not.
  const body: unknown = await received(deadline, () => response.json()).catch((error: unknown) => {
    if (error instanceof ApiUnavailable) throw error;
    return null;
  });
  throw errorFor(response.status, body);
}

function json<T>(path: string, init: RequestInit = {}, key?: string): Promise<T> {
  return ask(path, (response) => response.json() as Promise<T>, init, key);
}

function paged(path: string, cursor: string | undefined): string {
  return cursor ? `${path}?cursor=${encodeURIComponent(cursor)}` : path;
}

/** Whose key this is. Takes the key itself: sign-in calls it before any session exists. */
export function me(key: string): Promise<Me> {
  return json<Me>("/v1/me", {}, key);
}

export async function listEstimators(): Promise<EstimatorEntry[]> {
  return (await json<{ estimators: EstimatorEntry[] }>("/v1/estimators")).estimators;
}

export function listAnalyses(cursor?: string): Promise<AnalysisPage> {
  return json<AnalysisPage>(paged("/v1/analyses", cursor));
}

export function listDatasets(cursor?: string): Promise<DatasetPage> {
  return json<DatasetPage>(paged("/v1/datasets", cursor));
}

export function getDataset(id: string): Promise<DatasetDetail> {
  return json<DatasetDetail>(`/v1/datasets/${encodeURIComponent(id)}`);
}

export function createAnalysis(body: AnalysisRequest): Promise<AnalysisCreated> {
  return json<AnalysisCreated>("/v1/analyses", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getAnalysis(id: string): Promise<AnalysisDetail> {
  return json<AnalysisDetail>(`/v1/analyses/${encodeURIComponent(id)}`);
}

/**
 * Ask for the analysis's interpretation; the API answers with it, queued.
 * With a choice, from that entry of the server's list; without one, from the
 * list's first entry, the default.
 */
export function requestInterpretation(
  id: string,
  choice?: InterpretationChoice,
): Promise<InterpretationDetail> {
  const path = `/v1/analyses/${encodeURIComponent(id)}/interpretation`;
  if (!choice) return json<InterpretationDetail>(path, { method: "POST" });
  return json<InterpretationDetail>(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: choice.model, effort: choice.effort }),
  });
}

export function getSummary(id: string): Promise<Summary> {
  return json<Summary>(`/v1/analyses/${encodeURIComponent(id)}/summary`);
}

export function getReportHtml(id: string, method: string): Promise<string> {
  const path = `/v1/analyses/${encodeURIComponent(id)}/estimates/${encodeURIComponent(method)}`;
  return ask(`${path}/report.html`, (response) => response.text());
}
