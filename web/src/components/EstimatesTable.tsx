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
 * The interval is labelled with its own basis and level, and is never given
 * a generic name: for some estimators it is a posterior or a permutation
 * interval.
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
      <h2 id="estimates-heading" className="title-section">
        Estimates
      </h2>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Method</th>
              <th scope="col">Estimand</th>
              <th scope="col">Scale</th>
              <th scope="col" className="num">
                Point
              </th>
              <th scope="col" className="num whitespace-normal">
                Standard error
              </th>
              <th scope="col">Interval</th>
              <th scope="col">Bootstrap interval</th>
              <th scope="col" className="num">
                p-value
              </th>
              <th scope="col" className="num whitespace-normal">
                Treated / control
              </th>
              <th scope="col">Validity band</th>
              <th scope="col" className="num">
                Runtime
              </th>
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
                    <td colSpan={11} className="engine-inline text-alert">
                      {estimate.error ?? "no estimate was produced"}
                    </td>
                  </tr>
                );
              }
              return (
                <tr key={estimate.method}>
                  <th scope="row">{estimate.method}</th>
                  <td>{summary.estimand}</td>
                  <td>{summary.scale}</td>
                  <td className="num font-semibold">{formatNumber(summary.point)}</td>
                  <td className="num">{formatNumber(summary.se)}</td>
                  <td className="whitespace-nowrap">
                    {formatInterval(summary.ci_lower, summary.ci_upper)}{" "}
                    <span className="label block">
                      {formatLevel(summary.ci_level)} {summary.ci_basis} interval
                    </span>
                  </td>
                  <td className="whitespace-nowrap">
                    {formatInterval(summary.bootstrap_ci_lower, summary.bootstrap_ci_upper)}
                  </td>
                  <td className="num">{formatP(summary.p_value)}</td>
                  <td className="num">
                    {summary.n_treated} / {summary.n_control}
                  </td>
                  <td data-band={estimate.band ?? ""} className="band">
                    {estimate.band ?? DASH}
                  </td>
                  <td className="num text-ink-soft">{formatSeconds(summary.runtime_seconds)}</td>
                  <td className="font-sans">
                    <Link
                      href={`/analyses/${analysisId}/estimates/${encodeURIComponent(estimate.method)}`}
                      className="link"
                    >
                      Evidence report
                    </Link>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
