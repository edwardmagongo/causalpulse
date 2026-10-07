import { NextResponse, type NextRequest } from "next/server";

import { contentSecurityPolicy } from "@/lib/csp";
import { isProduction } from "@/lib/env";
import { SESSION_COOKIE } from "@/lib/session-cookie";

const POLICY_HEADER = "Content-Security-Policy";

// The pages served without a session, each as that exact path: sign-in, and
// the benchmark study, whose page calls no API and reads no session. A
// sub-path of either is gated like any other page.
const OPEN_PAGES: ReadonlySet<string> = new Set(["/sign-in", "/study"]);

// Presence only. Whether the key inside still works is the API's to say, on every call.
export function proxy(request: NextRequest) {
  const nonce = Buffer.from(crypto.randomUUID()).toString("base64");
  const policy = contentSecurityPolicy(nonce, process.env.NODE_ENV === "development", isProduction());
  if (!OPEN_PAGES.has(request.nextUrl.pathname) && !request.cookies.has(SESSION_COOKIE)) {
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
  // `/sign-in` and `/study` are matched, so that they are served under the
  // policy; the proxy lets those two paths through without a session, and a
  // sub-path of either is gated like any other page. `api` and `sign-out`
  // are each excluded as that path and its sub-paths only, so a page named
  // `/api-keys` or `/sign-outside` is still matched. `_next/static/` is
  // excluded as sub-paths only; `_next/image` and `favicon.ico` are exact.
  // Next.js reads this as a literal at build time: do not build it from an import.
  matcher: [
    "/((?!api/|api$|sign-out/|sign-out$|_next/static/|_next/image$|favicon\\.ico$).*)",
  ],
};
