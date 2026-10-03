import type { ComparisonGroup } from "@/lib/api/types";
import { formatNumber } from "@/lib/format";

/** Each comparison group's verdict in the API's own words, or why it was not compared. */
export function ComparisonList({ comparisons }: { comparisons: ComparisonGroup[] }) {
  if (comparisons.length === 0) return null;
  return (
    <section aria-labelledby="comparison-heading">
      <h2 id="comparison-heading" className="title-section">
        Comparison across methods
      </h2>
      {comparisons.map((group) => (
        <article key={`${group.estimand}/${group.scale}`} className="mb-8 last:mb-0">
          <h3 className="title-group">
            {group.estimand}, {group.scale} scale
          </h3>
          <p className="value text-ink-soft">{group.methods.join(", ")}</p>
          {group.verdict ? (
            <>
              <p data-agreement={group.agreement ?? ""} className="value mt-4 mb-2 font-semibold">
                {group.agreement}
              </p>
              <p className="engine">{group.verdict}</p>
              <dl className="mt-5 grid grid-flow-col grid-rows-[auto_auto] justify-start gap-x-8">
                <dt className="label">Point range</dt>
                <dd className="value">
                  {group.point_range
                    ? `${formatNumber(group.point_range[0])} to ${formatNumber(group.point_range[1])}`
                    : formatNumber(null)}
                </dd>
                <dt className="label">Dispersion</dt>
                <dd className="value">{formatNumber(group.dispersion)}</dd>
              </dl>
            </>
          ) : (
            <p className="engine mt-4">{group.not_compared_reason}</p>
          )}
        </article>
      ))}
    </section>
  );
}
