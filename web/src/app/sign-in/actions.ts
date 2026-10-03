"use server";

import { redirect } from "next/navigation";

import { me } from "@/lib/api/client";
import { ApiError, ApiUnavailable, Unauthorized } from "@/lib/api/errors";
import { getSession } from "@/lib/session";

export type SignInState = { error: string | null };

export async function signIn(_previous: SignInState, form: FormData): Promise<SignInState> {
  const key = String(form.get("key") ?? "").trim();
  // The dashboard's own words: nothing was sent, so there is no API message to show.
  if (!key) return { error: "Enter an API key." };
  let name: string;
  try {
    name = (await me(key)).name;
  } catch (error) {
    if (error instanceof Unauthorized) return { error: error.detail };
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
