import { ComparisonList } from "@/components/ComparisonList";
import { EstimatesTable } from "@/components/EstimatesTable";
import { IntervalPlot } from "@/components/IntervalPlot";
import { Poller } from "@/components/Poller";
import { RoutingTable } from "@/components/RoutingTable";
import { StatusHeader } from "@/components/StatusHeader";
import { Unavailable } from "@/components/Unavailable";
import { getAnalysis, getDataset, getSummary } from "@/lib/api/client";
import { load } from "@/lib/api/load";
import { panels } from "@/lib/plot";
import { inFlight } from "@/lib/polling";

export default async function AnalysisPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const loaded = await load(async () => {
    const analysis = await getAnalysis(id);
    const [dataset, summary] = await Promise.all([
      getDataset(analysis.dataset_id),
      // Only a succeeded analysis has a summary; the API answers 404 otherwise.
      analysis.status === "succeeded" ? getSummary(id) : null,
    ]);
    return { analysis, dataset, comparisons: summary?.comparisons ?? [] };
  });
  if (!loaded.ok) return <Unavailable />;
  const { analysis, dataset, comparisons } = loaded.data;
  return (
    <>
      <StatusHeader analysis={analysis} filename={dataset.filename} />
      {inFlight(analysis.status) ? <Poller analysisId={analysis.id} /> : null}
      <EstimatesTable analysisId={analysis.id} estimates={analysis.estimates} />
      {/* The picture and the verdict on it, side by side where there is room. */}
      <div className="grid grid-cols-[minmax(0,1fr)] gap-x-12 gap-y-section empty:hidden xl:grid-cols-[minmax(0,var(--container-plot))_minmax(0,1fr)]">
        <IntervalPlot panels={panels(analysis.estimates, comparisons)} />
        <ComparisonList comparisons={comparisons} />
      </div>
      <RoutingTable routing={analysis.routing} />
      {analysis.status === "succeeded" ? (
        <p className="text-note">
          <a href={`/api/analyses/${analysis.id}/manifest`} className="link">
            Download the manifest
          </a>
        </p>
      ) : null}
    </>
  );
}
