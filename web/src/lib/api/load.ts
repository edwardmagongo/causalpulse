import "server-only";

import { notFound, redirect } from "next/navigation";

import { ApiError, ApiUnavailable, Unauthorized } from "./errors";

/** Why a page has nothing to show. `limited` carries the API's own sentence. */
export type NotLoaded = { ok: false; reason: "unavailable" } | { ok: false; reason: "limited"; detail: string };

export type Loaded<T> = { ok: true; data: T } | NotLoaded;

/**
 * Run a read for a page, turning the API's refusals into what the page does.
 *
 * - 401 goes to `/sign-out`: a render cannot clear a cookie, a route handler can.
 * - 404 is the not-found page, the same for missing and for not-yours.
 * - 422 is the not-found page too: a read sends no body, so the API can only
 *   be refusing a malformed id or cursor in the URL.
 * - Unavailable comes back as a value, so the page can say so in place.
 * - 429 comes back as a value too, with the API's sentence, which says what
 *   was too many and when to try again. An address over the limit on failed
 *   sign-ins is answered 429 on every route, and the service is not down.
 */
export async function load<T>(read: () => Promise<T>): Promise<Loaded<T>> {
  try {
    return { ok: true, data: await read() };
  } catch (error) {
    if (error instanceof Unauthorized) redirect("/sign-out");
    if (error instanceof ApiError && (error.status === 404 || error.status === 422)) notFound();
    if (error instanceof ApiUnavailable) return { ok: false, reason: "unavailable" };
    if (error instanceof ApiError && error.status === 429) {
      return { ok: false, reason: "limited", detail: error.detail };
    }
    throw error;
  }
}
