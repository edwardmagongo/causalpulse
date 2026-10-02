import type { ComparisonGroup } from "@/lib/api/types";
import { formatNumber } from "@/lib/format";

/** Each comparison group's verdict in the API's own words, or why it was not compared. */
export function ComparisonList({ comparisons }: { comparisons: ComparisonGroup[] }) {
  if (comparisons.length === 0) return null;
  return (
    <section aria-labelledby="comparison-heading">
      <h2 id="comparison-heading">Comparison across methods</h2>
      {comparisons.map((group) => (
        <article key={`${group.estimand}/${group.scale}`}>
          <h3>
            {group.estimand}, {group.scale} scale
          </h3>
          <p>{group.methods.join(", ")}</p>
          {group.verdict ? (
            <>
              <p data-agreement={group.agreement ?? ""}>{group.agreement}</p>
              <p>{group.verdict}</p>
              <dl>
                <dt>Point range</dt>
                <dd>
                  {group.point_range
                    ? `${formatNumber(group.point_range[0])} to ${formatNumber(group.point_range[1])}`
                    : formatNumber(null)}
                </dd>
                <dt>Dispersion</dt>
                <dd>{formatNumber(group.dispersion)}</dd>
              </dl>
            </>
          ) : (
            <p>{group.not_compared_reason}</p>
          )}
        </article>
      ))}
    </section>
  );
}
