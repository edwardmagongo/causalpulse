import { StudyReport, StudyUnavailable } from "@/components/study/StudySections";
import { loadStudy } from "@/lib/study/bundle";
import { parseView } from "@/lib/study/query";

export default async function StudyPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const loaded = loadStudy();
  return (
    <>
      <header>
        <h1 className="title-page">Benchmark study</h1>
      </header>
      {loaded.ok ? (
        <StudyReport study={loaded.study} view={parseView(loaded.study, params)} />
      ) : (
        <StudyUnavailable failure={loaded} />
      )}
    </>
  );
}
