"use server";

import { redirect } from "next/navigation";

import { me } from "@/lib/api/client";
import { ApiError, ApiUnavailable, Unauthorized } from "@/lib/api/errors";
import { getSession } from "@/lib/session";

export type SignInState = { error: string | null };

// The API's own words for a key it does not accept (`cp_api.auth.UNAUTHORIZED`).
const REFUSED = "missing or invalid API key";

export async function signIn(_previous: SignInState, form: FormData): Promise<SignInState> {
  const key = String(form.get("key") ?? "").trim();
  if (!key) return { error: REFUSED };
  let name: string;
  try {
    name = (await me(key)).name;
  } catch (error) {
    if (error instanceof Unauthorized) return { error: REFUSED };
    if (error instanceof ApiUnavailable) {
      return { error: "The service is unavailable. Try again in a moment." };
    }
    if (error instanceof ApiError) return { error: error.detail };
    throw error;
  }
  const session = await getSession();
  session.key = key;
  session.name = name;
  await session.save();
  redirect("/");
}

export async function signOut(): Promise<void> {
  (await getSession()).destroy();
  redirect("/sign-in");
}
