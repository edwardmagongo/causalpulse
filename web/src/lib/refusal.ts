import "server-only";

import { redirect } from "next/navigation";

import { ApiError, ApiUnavailable, Unauthorized } from "@/lib/api/errors";

export const UNAVAILABLE = "The service is unavailable. Try again in a moment.";

/** What a server action answers for a refusal; a 401 leaves for sign-out instead. */
export function refusal(error: unknown): { error: string } {
  if (error instanceof Unauthorized) redirect("/sign-out");
  if (error instanceof ApiUnavailable) return { error: UNAVAILABLE };
  if (error instanceof ApiError) return { error: error.detail };
  throw error;
}
