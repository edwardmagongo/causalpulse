import "server-only";

import { ApiUnavailable, Unauthorized } from "./errors";
import { forward } from "./client";

/**
 * Pass one API response on to the browser, body and status unchanged.
 *
 * `headers` adds to the response, for a download's `content-disposition`.
 */
export async function relay(
  path: string,
  init: RequestInit = {},
  headers: Record<string, string> = {},
): Promise<Response> {
  try {
    const upstream = await forward(path, init);
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") ?? "application/json",
        ...(upstream.ok ? headers : {}),
      },
    });
  } catch (error) {
    if (error instanceof Unauthorized) {
      return Response.json({ detail: "signed out" }, { status: 401 });
    }
    if (error instanceof ApiUnavailable) {
      return Response.json({ detail: "the service is unavailable" }, { status: 503 });
    }
    throw error;
  }
}
