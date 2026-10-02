import "server-only";

import { getIronSession, type IronSession, type SessionOptions } from "iron-session";
import { cookies } from "next/headers";

import { sessionSecret } from "@/lib/env";

export const SESSION_COOKIE = "cp_session";
const SEVEN_DAYS = 7 * 24 * 60 * 60;

export type SessionData = { key?: string; name?: string };

export function sessionOptions(): SessionOptions {
  return {
    password: sessionSecret(),
    cookieName: SESSION_COOKIE,
    ttl: SEVEN_DAYS,
    cookieOptions: {
      httpOnly: true,
      sameSite: "lax",
      path: "/",
      secure: process.env.CP_COOKIE_SECURE === "true",
    },
  };
}

export async function getSession(): Promise<IronSession<SessionData>> {
  return getIronSession<SessionData>(await cookies(), sessionOptions());
}
