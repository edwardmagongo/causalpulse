import Link from "next/link";

import { Unavailable } from "@/components/Unavailable";
import { getAnalysis, getReportHtml } from "@/lib/api/client";
import { ApiError } from "@/lib/api/errors";
import { load } from "@/lib/api/load";

export default async function ReportPage({
  params,
}: {
  params: Promise<{ id: string; method: string }>;
}) {
  const { id, method: encoded } = await params;
  const method = decodeURIComponent(encoded);
  const loaded = await load(async () => {
    // First, so an analysis that is missing or not the caller's is a 404 page.
    await getAnalysis(id);
    try {
      return { html: await getReportHtml(id, method), reason: null };
    } catch (error) {
      // The estimator ran and produced no report: the API says why.
      if (error instanceof ApiError && error.status === 404) {
        return { html: null, reason: error.detail };
      }
      throw error;
    }
  });
  if (!loaded.ok) return <Unavailable failure={loaded} />;
  const { html, reason } = loaded.data;
  return (
    <>
      <div className="flex flex-col gap-4">
        <p className="text-note">
          <Link href={`/analyses/${id}`} className="link">
            Back to the analysis
          </Link>
        </p>
        <h1 className="title-page">Evidence report: {method}</h1>
        {html === null ? null : (
          <p className="text-note">
            <a
              href={`/api/analyses/${id}/estimates/${encodeURIComponent(method)}/report.md`}
              className="link"
            >
              Download as Markdown
            </a>
          </p>
        )}
      </div>
      {html === null ? (
        <p role="alert" className="engine engine-alert">
          {reason}
        </p>
      ) : (
        /* sandbox="" grants nothing: no script, no same-origin access, no forms. */
        <iframe
          title={`Evidence report for ${method}`}
          sandbox=""
          srcDoc={html}
          className="block h-[calc(100dvh-16rem)] min-h-[24rem] w-full rounded-control border border-rule bg-surface"
        />
      )}
    </>
  );
}
