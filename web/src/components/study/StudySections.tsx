import type {
  Cate,
  Expectations,
  Limitations,
  Overview,
  P7,
  Provenance,
  StudyBundle,
} from "@/lib/study/types";
import type { StudyLoad } from "@/lib/study/bundle";
import { formatNumber } from "@/lib/format";
import { matrixOf, offeredFamilies } from "@/lib/study/matrix";
import { defaultView, viewHref } from "@/lib/study/query";
import type { View } from "@/lib/study/query";
import { scaleFor } from "@/lib/study/scale";

import { MatrixHeatmap } from "./MatrixHeatmap";
import { MatrixTable } from "./MatrixTable";
import { ScaleLegend } from "./ScaleLegend";
import { StudyTable } from "./StudyTable";
import { ViewSwitches } from "./ViewSwitches";

/**
 * The columns of each table, in the report's order. A record's own keys are
 * alphabetical, so the order is declared here; `study.test.tsx` asserts these
 * names, with each metric's `_mcse` and `_n_used`, are exactly the table's keys.
 */
export const CATE_COLUMNS = {
  keys: ["method", "scenario"],
  metrics: ["cate_correlation", "cate_rmse", "cate_calibration_slope"],
} as const;

export const P7_COLUMNS = {
  keys: ["method", "interval"],
  metrics: ["bias", "rmse", "coverage"],
} as const;

const METRIC_LEGEND =
  "Each metric is followed by its Monte Carlo standard error (±). A count of fits appears only where fewer fits than the replications fed a metric.";

/** The whole report, sections in order. Server-compatible: props in, markup out. */
export function StudyReport({ study, view }: { study: StudyBundle; view?: View }) {
  return (
    <>
      <OverviewSection overview={study.overview} provenance={study.provenance} />
      <FindingsSection expectations={study.expectations} />
      <MethodsSection study={study} view={view ?? defaultView(study)} />
      <CateSection cate={study.cate} />
      <P7Section p7={study.p7} />
      <LimitationsSection limitations={study.limitations} />
    </>
  );
}

/** What the page says when the bundle cannot be read: no section, no causal prose. */
export function StudyUnavailable({ failure }: { failure: Exclude<StudyLoad, { ok: true }> }) {
  if (failure.reason === "version") {
    return (
      <div role="status" className="max-w-measure">
        <p>
          This build of the dashboard cannot read the study data: it is schema version{" "}
          <span className="value">{failure.found}</span>, and the versions understood are{" "}
          <span className="value">{failure.known.join(", ")}</span>.
        </p>
      </div>
    );
  }
  return (
    <div role="status" className="max-w-measure">
      <p>This build of the dashboard does not recognise these entries in the study data:</p>
      <ul className="mt-3 flex flex-col gap-1">
        {failure.unknown.map((entry) => (
          <li key={entry} className="value">
            {entry}
          </li>
        ))}
      </ul>
    </div>
  );
}

function Section({ id, title, children }: { id: string; title: string; children?: React.ReactNode }) {
  return (
    <section aria-labelledby={id}>
      <h2 id={id} className="title-section">
        {title}
      </h2>
      {children}
    </section>
  );
}

export function OverviewSection({
  overview,
  provenance,
}: {
  overview: Overview;
  provenance: readonly Provenance[];
}) {
  const facts: [string, string][] = [
    ["Estimators", String(overview.n_estimators)],
    ["Scenarios", String(overview.n_scenarios)],
    ["Cells", String(overview.n_cells)],
    ["Replications", String(overview.replications)],
    ["Coverage Monte Carlo error", formatNumber(overview.coverage_mcse)],
  ];
  return (
    <Section id="overview" title="Overview">
      <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 sm:grid-flow-col sm:grid-cols-none sm:grid-rows-[auto_auto] sm:justify-start sm:gap-x-10 sm:gap-y-0">
        {facts.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="label">{label}</dt>
            <dd className="value">{value}</dd>
          </div>
        ))}
      </dl>
      <ul className="mt-6 flex flex-col gap-1">
        {provenance.map((entry) => (
          <li key={entry.role} className="text-note">
            <span className="label">Run</span> <span className="value font-semibold">{entry.role}</span>
            {" · "}
            <span className="label">tier</span> <span className="value">{entry.tier}</span>
            {" · "}
            <span className="value">{entry.replications}</span> <span className="label">replications</span>
            {" · "}
            <span className="value">
              {entry.cells_done} of {entry.cells_expected}
            </span>{" "}
            <span className="label">cells</span>
            {" · "}
            <span className="label">commit</span>{" "}
            <span className="value" title={entry.git_commit}>
              {entry.git_commit.slice(0, 7)}
            </span>
          </li>
        ))}
      </ul>
    </Section>
  );
}

/** Phase 1's status styling where a class already fits: a failure is `failed`, a met expectation plain ink. */
function verdictStatus(verdict: string): string | undefined {
  if (verdict === "failed") return "failed";
  if (verdict === "met") return "succeeded";
  return undefined;
}

export function FindingsSection({ expectations }: { expectations: Expectations }) {
  const tally: [string, number][] = [
    ["Met", expectations.n_met],
    ["Failed", expectations.n_failed],
    ["Inconclusive", expectations.n_inconclusive],
  ];
  return (
    <Section id="findings" title="Findings">
      <dl aria-label="Tally" className="mb-8 flex gap-x-10">
        {tally.map(([label, count]) => (
          <div key={label}>
            <dt className="label">{label}</dt>
            <dd className="value text-section font-semibold">{count}</dd>
          </div>
        ))}
      </dl>
      <ol className="flex flex-col gap-8">
        {expectations.rows.map((row) => (
          <li key={row.id}>
            <p className="flex flex-wrap items-baseline gap-x-4">
              <span className="value font-semibold">{row.id}</span>
              <span data-status={verdictStatus(row.verdict)} className="status">
                {row.verdict}
              </span>
            </p>
            <p className="mt-1 max-w-measure font-serif text-prose font-semibold">{row.statement}</p>
            <p className="engine mt-3">{row.narrative}</p>
          </li>
        ))}
      </ol>
    </Section>
  );
}

/**
 * The estimator-by-scenario matrix for one view: the switches, the metric's
 * definition and the study's notes (each verbatim), then the matrix as a
 * heatmap or a sortable table. Built here, on the server, from the view the
 * URL named.
 */
export function MethodsSection({ study, view }: { study: StudyBundle; view: View }) {
  const href = (next: View) => viewHref(study, next);
  const matrix = matrixOf(study, view);
  const values = matrix.rows.flatMap((row) => row.cells.flatMap((cell) => (cell.kind === "value" ? [cell.value] : [])));
  const scale = scaleFor(view.metric, values);
  const definition = Object.hasOwn(study.metric_definitions, view.metric)
    ? study.metric_definitions[view.metric]
    : undefined;
  return (
    <Section id="methods" title="Methods">
      <ViewSwitches view={view} offered={offeredFamilies(study)} href={href} />
      {definition ? <p className="engine mt-6">{definition}</p> : null}
      {view.family === "coverage" ? <p className="engine mt-4">{study.report_tables.coverage_note}</p> : null}
      {view.level === "scenario_by_size" ? (
        <p className="engine mt-4">{study.report_tables.exploratory_note}</p>
      ) : null}
      <div className="mt-6">
        {view.display === "heatmap" ? (
          <>
            <ScaleLegend scale={scale} metric={view.metric} />
            <MatrixHeatmap matrix={matrix} scale={scale} level={view.level} />
          </>
        ) : (
          <MatrixTable matrix={matrix} view={view} href={href} />
        )}
      </div>
    </Section>
  );
}

const NO_SECTION = "The report has no such section.";

function ArmSection({
  id,
  title,
  arm,
  columns,
}: {
  id: string;
  title: string;
  arm: Cate | P7 | null;
  columns: { keys: readonly string[]; metrics: readonly string[] };
}) {
  if (arm === null) {
    return (
      <Section id={id} title={title}>
        <p className="max-w-measure text-ink-soft">{NO_SECTION}</p>
      </Section>
    );
  }
  return (
    <Section id={id} title={title}>
      <p className="engine">{arm.note}</p>
      <p className="mt-5 mb-4">
        <span className="value">{arm.replications}</span> <span className="label">replications</span>
        {" · "}
        <span className="value">
          {arm.cells_done} of {arm.cells_expected}
        </span>{" "}
        <span className="label">cells</span>
      </p>
      <StudyTable
        caption={title}
        rows={arm.table}
        keys={columns.keys}
        metrics={columns.metrics}
        replications={arm.replications}
      />
      <p className="label mt-3 max-w-measure">{METRIC_LEGEND}</p>
    </Section>
  );
}

export function CateSection({ cate }: { cate: Cate | null }) {
  return <ArmSection id="cate" title="CATE" arm={cate} columns={CATE_COLUMNS} />;
}

export function P7Section({ p7 }: { p7: P7 | null }) {
  return <ArmSection id="p7" title="Scenario P7" arm={p7} columns={P7_COLUMNS} />;
}

export function LimitationsSection({ limitations }: { limitations: Limitations }) {
  return (
    <Section id="limitations" title="Limitations">
      {limitations.not_covered.length > 0 ? (
        <div className="mb-8">
          <h3 className="title-group mb-3">Not covered</h3>
          <ul className="flex flex-col gap-4">
            {limitations.not_covered.map((entry) => (
              <li key={entry} className="engine">
                {entry}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      <ul className="flex flex-col gap-6">
        {limitations.statements.map((statement) => (
          <li key={statement} className="engine">
            {statement}
          </li>
        ))}
      </ul>
    </Section>
  );
}
