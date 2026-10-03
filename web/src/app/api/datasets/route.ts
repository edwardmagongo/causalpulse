import { relay } from "@/lib/api/relay";

/**
 * The upload, streamed to cp-api.
 *
 * A route handler, not a server action: an action caps its body at 1 MB. The
 * body is passed on as a stream, so a 50 MB file is never held in memory
 * here. This path must stay out of `proxy.ts`'s matcher: when the proxy runs
 * on a request, Next.js buffers the body and cuts it off at 10 MB.
 */
export async function POST(request: Request) {
  const origin = request.headers.get("origin");
  const host = request.headers.get("x-forwarded-host") ?? request.headers.get("host");
  if (!origin || !host || new URL(origin).host !== host) {
    return Response.json({ detail: "the upload must come from this site" }, { status: 403 });
  }
  return relay("/v1/datasets", {
    method: "POST",
    headers: { "content-type": request.headers.get("content-type") ?? "" },
    body: request.body,
    // Node's fetch requires this to send a stream as a request body; the DOM types omit it.
    duplex: "half",
  } as RequestInit);
}
