import type { RoutingDecision } from "@/lib/api/types";

/** What the router decided for each estimator, eligible ones first, reasons verbatim. */
export function RoutingTable({ routing }: { routing: RoutingDecision[] }) {
  if (routing.length === 0) return null;
  const ordered = [...routing].sort(
    (a, b) => Number(b.eligible) - Number(a.eligible) || a.estimator.localeCompare(b.estimator),
  );
  return (
    <section aria-labelledby="routing-heading">
      <h2 id="routing-heading" className="title-section">
        Routing
      </h2>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Estimator</th>
              <th scope="col">Eligible</th>
              <th scope="col" className="w-full">
                Reasons
              </th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((decision) => (
              <tr key={decision.estimator}>
                <th scope="row">{decision.estimator}</th>
                <td
                  className={`font-sans whitespace-nowrap ${decision.eligible ? "font-medium" : "text-ink-soft"}`}
                >
                  {decision.eligible ? "eligible" : "not eligible"}
                </td>
                <td className="engine-inline min-w-72">
                  {decision.reasons.length === 0 ? null : (
                    <ul className="flex flex-col gap-1">
                      {decision.reasons.map((reason) => (
                        <li key={reason}>{reason}</li>
                      ))}
                    </ul>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
