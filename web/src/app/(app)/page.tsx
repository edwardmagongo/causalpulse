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
        <h1 className="title-page">Analyses</h1>
        <p className="mt-6 mb-4 text-ink-soft">No analyses yet.</p>
        <Link href="/analyses/new" className="button button-primary">
          Start a new analysis
        </Link>
      </section>
    );
  }
  return (
    <section>
      <h1 className="title-page mb-6">Analyses</h1>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Dataset</th>
              <th scope="col">Status</th>
              <th scope="col">Estimators</th>
              <th scope="col" className="num">
                Created
              </th>
              <th scope="col" className="num">
                Finished
              </th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.id}>
                <th scope="row">
                  <Link href={`/analyses/${item.id}`} className="link">
                    {item.dataset_filename}
                  </Link>
                </th>
                <td className="whitespace-nowrap">
                  <span data-status={item.status} className="status">
                    {item.status}
                  </span>
                  {item.partial ? " (partial)" : null}
                </td>
                <td className={item.requested_estimators ? "min-w-48" : "font-sans text-ink-soft"}>
                  {item.requested_estimators?.join(", ") ?? "chosen by the router"}
                </td>
                <td className="num text-ink-soft">{formatTime(item.created_at)}</td>
                <td className="num text-ink-soft">{formatTime(item.finished_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {nextCursor ? (
        <Link href={`/?cursor=${encodeURIComponent(nextCursor)}`} className="link mt-6 inline-block text-note">
          Older analyses
        </Link>
      ) : null}
    </section>
  );
}
