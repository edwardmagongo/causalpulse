import { StudyReport, StudyUnavailable } from "@/components/study/StudySections";
import { loadStudy } from "@/lib/study/bundle";

export default function StudyPage() {
  const loaded = loadStudy();
  return (
    <>
      <header>
        <h1 className="title-page">Benchmark study</h1>
      </header>
      {loaded.ok ? <StudyReport study={loaded.study} /> : <StudyUnavailable failure={loaded} />}
    </>
  );
}
