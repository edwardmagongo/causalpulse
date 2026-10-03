import { relay } from "@/lib/api/relay";

/** One estimate's Markdown report, as a download. */
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string; method: string }> },
) {
  const { id, method } = await params;
  const path = `/v1/analyses/${encodeURIComponent(id)}/estimates/${encodeURIComponent(method)}`;
  return relay(`${path}/report.md`, {}, {
    // The method is a path segment the API already matched, but it is quoted
    // into a header here, so anything outside the estimator-name alphabet goes.
    "content-disposition": `attachment; filename="${method.replace(/[^A-Za-z0-9_-]/g, "")}-report.md"`,
  });
}
