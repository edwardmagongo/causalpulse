import "server-only";

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
 */

async function sessionKey(): Promise<string> {
  const { key } = await getSession();
  if (!key) throw new Unauthorized();
  return key;
}

async function send(path: string, init: RequestInit, key: string): Promise<Response> {
  try {
    return await fetch(`${apiUrl()}${path}`, {
      ...init,
      cache: "no-store",
      headers: { ...init.headers, authorization: `Bearer ${key}` },
    });
  } catch {
    throw new ApiUnavailable();
  }
}

/** The API's own response, 4xx included. Throws only for 401, 503 and no connection. */
export async function forward(path: string, init: RequestInit = {}): Promise<Response> {
  const response = await send(path, init, await sessionKey());
  if (response.status === 401) throw new Unauthorized();
  if (response.status === 503) throw new ApiUnavailable();
  return response;
}

async function ask(path: string, init: RequestInit = {}, key?: string): Promise<Response> {
  const response = await send(path, init, key ?? (await sessionKey()));
  if (!response.ok) {
    throw errorFor(response.status, await response.json().catch(() => null));
  }
  return response;
}

async function json<T>(path: string, init: RequestInit = {}, key?: string): Promise<T> {
  return (await (await ask(path, init, key)).json()) as T;
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

export function getSummary(id: string): Promise<Summary> {
  return json<Summary>(`/v1/analyses/${encodeURIComponent(id)}/summary`);
}

export async function getReportHtml(id: string, method: string): Promise<string> {
  const path = `/v1/analyses/${encodeURIComponent(id)}/estimates/${encodeURIComponent(method)}`;
  return (await ask(`${path}/report.html`)).text();
}
