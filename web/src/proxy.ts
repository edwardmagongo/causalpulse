import { NextResponse, type NextRequest } from "next/server";

import { SESSION_COOKIE } from "@/lib/session-cookie";

// Presence only. Whether the key inside still works is the API's to say, on every call.
export function proxy(request: NextRequest) {
  if (request.cookies.has(SESSION_COOKIE)) return NextResponse.next();
  return NextResponse.redirect(new URL("/sign-in", request.url));
}

export const config = {
  // Never `/api`: when the proxy runs on a request, Next.js buffers its body
  // and truncates it at 10 MB, which would silently cut an upload short.
  // Each exclusion is that path and its sub-paths only, so a page named
  // `/api-keys` or `/sign-insider` is still gated. Next.js reads this as a
  // literal at build time: do not build it from an import.
  matcher: [
    "/((?!api/|api$|sign-in/|sign-in$|sign-out/|sign-out$|_next/static/|_next/image$|favicon\\.ico$).*)",
  ],
};
