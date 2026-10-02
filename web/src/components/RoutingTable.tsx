import type { RoutingDecision } from "@/lib/api/types";

/** What the router decided for each estimator, eligible ones first, reasons verbatim. */
export function RoutingTable({ routing }: { routing: RoutingDecision[] }) {
  if (routing.length === 0) return null;
  const ordered = [...routing].sort(
    (a, b) => Number(b.eligible) - Number(a.eligible) || a.estimator.localeCompare(b.estimator),
  );
  return (
    <section aria-labelledby="routing-heading">
      <h2 id="routing-heading">Routing</h2>
      <table>
        <thead>
          <tr>
            <th scope="col">Estimator</th>
            <th scope="col">Eligible</th>
            <th scope="col">Reasons</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map((decision) => (
            <tr key={decision.estimator}>
              <th scope="row">{decision.estimator}</th>
              <td>{decision.eligible ? "eligible" : "not eligible"}</td>
              <td>
                {decision.reasons.length === 0 ? null : (
                  <ul>
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
    </section>
  );
}
