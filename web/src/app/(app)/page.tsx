import Link from "next/link";

import { Unavailable } from "@/components/Unavailable";
import { listAnalyses } from "@/lib/api/client";
import { load } from "@/lib/api/load";
import { formatTime } from "@/lib/format";

export default async function AnalysesPage({
  searchParams,
}: {
  searchParams: Promise<{ cursor?: string }>;
}) {
  const { cursor } = await searchParams;
  const loaded = await load(() => listAnalyses(cursor));
  if (!loaded.ok) return <Unavailable />;
  const { items, next_cursor: nextCursor } = loaded.data;
  if (items.length === 0 && !cursor) {
    return (
      <section>
        <h1>Analyses</h1>
        <p>No analyses yet.</p>
        <Link href="/analyses/new">Start a new analysis</Link>
      </section>
    );
  }
  return (
    <section>
      <h1>Analyses</h1>
      <table>
        <thead>
          <tr>
            <th scope="col">Dataset</th>
            <th scope="col">Status</th>
            <th scope="col">Estimators</th>
            <th scope="col">Created</th>
            <th scope="col">Finished</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.id}>
              <th scope="row">
                <Link href={`/analyses/${item.id}`}>{item.dataset_filename}</Link>
              </th>
              <td>
                <span data-status={item.status}>{item.status}</span>
                {item.partial ? " (partial)" : null}
              </td>
              <td>{item.requested_estimators?.join(", ") ?? "chosen by the router"}</td>
              <td>{formatTime(item.created_at)}</td>
              <td>{formatTime(item.finished_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {nextCursor ? <Link href={`/?cursor=${encodeURIComponent(nextCursor)}`}>Older analyses</Link> : null}
    </section>
  );
}
