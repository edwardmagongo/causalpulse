import { NextResponse, type NextRequest } from "next/server";

// Presence only. Whether the key inside still works is the API's to say, on every call.
export function proxy(request: NextRequest) {
  if (request.cookies.has("cp_session")) return NextResponse.next();
  return NextResponse.redirect(new URL("/sign-in", request.url));
}

export const config = {
  // Never `/api`: when the proxy runs on a request, Next.js buffers its body
  // and truncates it at 10 MB, which would silently cut an upload short.
  matcher: ["/((?!api|sign-in|sign-out|_next/static|_next/image|favicon.ico).*)"],
};
