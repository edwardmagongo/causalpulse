import type { AnalysisDetail } from "@/lib/api/types";
import { formatTime } from "@/lib/format";

/** Status, the partial flag, timestamps, and the stored error when the analysis failed. */
export function StatusHeader({ analysis, filename }: { analysis: AnalysisDetail; filename: string }) {
  return (
    <header>
      <h1>Analysis of {filename}</h1>
      <p>
        <span data-status={analysis.status}>{analysis.status}</span>
        {analysis.partial ? <span data-partial="true"> partial: some estimators failed</span> : null}
      </p>
      <dl>
        <dt>Created</dt>
        <dd>{formatTime(analysis.created_at)}</dd>
        <dt>Started</dt>
        <dd>{formatTime(analysis.started_at)}</dd>
        <dt>Finished</dt>
        <dd>{formatTime(analysis.finished_at)}</dd>
      </dl>
      {analysis.error ? <p role="alert">{analysis.error}</p> : null}
    </header>
  );
}
