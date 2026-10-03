import type { AnalysisDetail } from "@/lib/api/types";
import { formatTime } from "@/lib/format";

/** Status, the partial flag, timestamps, and the stored error when the analysis failed. */
export function StatusHeader({ analysis, filename }: { analysis: AnalysisDetail; filename: string }) {
  return (
    <header>
      <h1 className="title-page">Analysis of {filename}</h1>
      <div className="mt-5 flex flex-col items-start gap-x-12 gap-y-4 sm:flex-row sm:flex-wrap sm:items-end">
        <p>
          <span data-status={analysis.status} className="status">
            {analysis.status}
          </span>
          {analysis.partial ? (
            <span data-partial="true" className="text-note font-medium text-caution">
              {" "}
              partial: some estimators failed
            </span>
          ) : null}
        </p>
        {/* Label over value in columns; label beside value where the screen is narrow. */}
        <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 sm:grid-flow-col sm:grid-cols-none sm:grid-rows-[auto_auto] sm:gap-x-8 sm:gap-y-0">
          <dt className="label">Created</dt>
          <dd className="value">{formatTime(analysis.created_at)}</dd>
          <dt className="label">Started</dt>
          <dd className="value">{formatTime(analysis.started_at)}</dd>
          <dt className="label">Finished</dt>
          <dd className="value">{formatTime(analysis.finished_at)}</dd>
        </dl>
      </div>
      {analysis.error ? (
        <p role="alert" className="engine engine-alert mt-8">
          {analysis.error}
        </p>
      ) : null}
    </header>
  );
}
