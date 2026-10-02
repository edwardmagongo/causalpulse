import "server-only";

import { notFound, redirect } from "next/navigation";

import { ApiError, ApiUnavailable, Unauthorized } from "./errors";

export type Loaded<T> = { ok: true; data: T } | { ok: false; reason: "unavailable" };

/**
 * Run a read for a page, turning the API's refusals into what the page does.
 *
 * - 401 goes to `/sign-out`: a render cannot clear a cookie, a route handler can.
 * - 404 is the not-found page, the same for missing and for not-yours.
 * - Unavailable comes back as a value, so the page can say so in place.
 */
export async function load<T>(read: () => Promise<T>): Promise<Loaded<T>> {
  try {
    return { ok: true, data: await read() };
  } catch (error) {
    if (error instanceof Unauthorized) redirect("/sign-out");
    if (error instanceof ApiError && error.status === 404) notFound();
    if (error instanceof ApiUnavailable) return { ok: false, reason: "unavailable" };
    throw error;
  }
}
