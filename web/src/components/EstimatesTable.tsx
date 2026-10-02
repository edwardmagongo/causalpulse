import Link from "next/link";

import type { Estimate } from "@/lib/api/types";
import {
  DASH,
  formatInterval,
  formatLevel,
  formatNumber,
  formatP,
  formatSeconds,
} from "@/lib/format";

/**
 * One row per requested estimator: its numbers and band, or its error.
 *
 * The interval is labelled with its own basis and level. It is never called
 * a confidence interval here, because for some estimators it is not one.
 */
export function EstimatesTable({
  analysisId,
  estimates,
}: {
  analysisId: string;
  estimates: Estimate[];
}) {
  if (estimates.length === 0) return null;
  return (
    <section aria-labelledby="estimates-heading">
      <h2 id="estimates-heading">Estimates</h2>
      <table>
        <thead>
          <tr>
            <th scope="col">Method</th>
            <th scope="col">Estimand</th>
            <th scope="col">Scale</th>
            <th scope="col">Point</th>
            <th scope="col">Standard error</th>
            <th scope="col">Interval</th>
            <th scope="col">Bootstrap interval</th>
            <th scope="col">p-value</th>
            <th scope="col">Treated / control</th>
            <th scope="col">Validity band</th>
            <th scope="col">Runtime</th>
            <th scope="col">Report</th>
          </tr>
        </thead>
        <tbody>
          {estimates.map((estimate) => {
            const { summary } = estimate;
            if (!summary) {
              return (
                <tr key={estimate.method} data-failed="true">
                  <th scope="row">{estimate.method}</th>
                  <td colSpan={11}>{estimate.error ?? "no estimate was produced"}</td>
                </tr>
              );
            }
            return (
              <tr key={estimate.method}>
                <th scope="row">{estimate.method}</th>
                <td>{summary.estimand}</td>
                <td>{summary.scale}</td>
                <td>{formatNumber(summary.point)}</td>
                <td>{formatNumber(summary.se)}</td>
                <td>
                  {formatInterval(summary.ci_lower, summary.ci_upper)}{" "}
                  <span>
                    {formatLevel(summary.ci_level)} {summary.ci_basis} interval
                  </span>
                </td>
                <td>{formatInterval(summary.bootstrap_ci_lower, summary.bootstrap_ci_upper)}</td>
                <td>{formatP(summary.p_value)}</td>
                <td>
                  {summary.n_treated} / {summary.n_control}
                </td>
                <td data-band={estimate.band ?? ""}>{estimate.band ?? DASH}</td>
                <td>{formatSeconds(summary.runtime_seconds)}</td>
                <td>
                  <Link
                    href={`/analyses/${analysisId}/estimates/${encodeURIComponent(estimate.method)}`}
                  >
                    Evidence report
                  </Link>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
