import { NextResponse, type NextRequest } from "next/server";

import { contentSecurityPolicy } from "@/lib/csp";
import { SESSION_COOKIE } from "@/lib/session-cookie";

const POLICY_HEADER = "Content-Security-Policy";

function isSignIn(pathname: string): boolean {
  return pathname === "/sign-in" || pathname.startsWith("/sign-in/");
}

// Presence only. Whether the key inside still works is the API's to say, on every call.
export function proxy(request: NextRequest) {
  const nonce = Buffer.from(crypto.randomUUID()).toString("base64");
  const policy = contentSecurityPolicy(nonce, process.env.NODE_ENV === "development");
  if (!isSignIn(request.nextUrl.pathname) && !request.cookies.has(SESSION_COOKIE)) {
    const redirect = NextResponse.redirect(new URL("/sign-in", request.url));
    redirect.headers.set(POLICY_HEADER, policy);
    return redirect;
  }
  // Next.js reads the nonce from the policy on the request and puts it on its own scripts.
  const headers = new Headers(request.headers);
  headers.set("x-nonce", nonce);
  headers.set(POLICY_HEADER, policy);
  const response = NextResponse.next({ request: { headers } });
  response.headers.set(POLICY_HEADER, policy);
  return response;
}

export const config = {
  // Never `/api`: when the proxy runs on a request, Next.js buffers its body
  // and truncates it at 10 MB, which would silently cut an upload short.
  // `/sign-in` is matched, so that it is served under the policy; the proxy
  // lets it through without a session. `api` and `sign-out` are each excluded
  // as that path and its sub-paths only, so a page named `/api-keys` or
  // `/sign-outside` is still matched. `_next/static/` is excluded as
  // sub-paths only; `_next/image` and `favicon.ico` are exact.
  // Next.js reads this as a literal at build time: do not build it from an import.
  matcher: [
    "/((?!api/|api$|sign-out/|sign-out$|_next/static/|_next/image$|favicon\\.ico$).*)",
  ],
};
