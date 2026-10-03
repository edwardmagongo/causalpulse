import { NewAnalysisForm } from "@/components/NewAnalysisForm";
import { Unavailable } from "@/components/Unavailable";
import { listDatasets, listEstimators } from "@/lib/api/client";
import { load } from "@/lib/api/load";

export default async function NewAnalysisPage() {
  const loaded = await load(async () => {
    const [datasets, estimators] = await Promise.all([listDatasets(), listEstimators()]);
    return { datasets: datasets.items, estimators };
  });
  if (!loaded.ok) return <Unavailable />;
  return (
    <>
      <h1 className="title-page">New analysis</h1>
      <NewAnalysisForm datasets={loaded.data.datasets} estimators={loaded.data.estimators} />
    </>
  );
}
