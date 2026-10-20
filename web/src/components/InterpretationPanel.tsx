import type { ReactNode } from "react";

import type { InterpretationDetail, InterpretationOption } from "@/lib/api/types";
import { DASH, formatTime } from "@/lib/format";

import { InterpretButton, optionLabel } from "./InterpretButton";

const VERIFIED_HEADING = "Interpretation (model-written, verified against the facts)";
const REFUSED =
  "The model's narrative could not be verified against this analysis's facts, so it is not shown.";
const OUT_OF_DATE = "Written under an earlier version of the prompt and rules.";

/**
 * The analysis's interpretation, or how to get one.
 *
 * Every sentence about the analysis comes from the API: the statements, the
 * limits and what "verified" means were written by the system and stored by
 * the worker. What this writes is the interface around them. With
 * interpretations off on the server, and none on record, there is no section.
 * Earlier interpretations, newest first, follow under a closed disclosure.
 *
 * `options` is the server's list of models and efforts for this analysis.
 * Only the entries a request would be accepted for are offered, and there is
 * a button when there is one, unless the newest interpretation is still
 * being written.
 */
export function InterpretationPanel({
  analysisId,
  interpretation,
  earlierInterpretations,
  options,
  interpretableReason,
}: {
  analysisId: string;
  interpretation: InterpretationDetail | null;
  earlierInterpretations: InterpretationDetail[];
  options: InterpretationOption[];
  interpretableReason: string | null;
}) {
  const available = options.filter((option) => option.available);
  if (interpretation === null && available.length === 0 && interpretableReason === null) return null;
  const verified = interpretation?.status === "verified";
  return (
    <section aria-labelledby="interpretation-heading" className="flex max-w-measure flex-col gap-4">
      <h2 id="interpretation-heading" className="title-section">
        {verified ? VERIFIED_HEADING : "Interpretation"}
      </h2>
      <Body
        analysisId={analysisId}
        interpretation={interpretation}
        available={available}
        interpretableReason={interpretableReason}
      />
      {earlierInterpretations.length > 0 ? <Earlier interpretations={earlierInterpretations} /> : null}
    </section>
  );
}

function Body({
  analysisId,
  interpretation,
  available,
  interpretableReason,
}: {
  analysisId: string;
  interpretation: InterpretationDetail | null;
  available: InterpretationOption[];
  interpretableReason: string | null;
}) {
  // The button's label follows the newest interpretation.
  const ask = (label: string) =>
    available.length > 0 ? (
      <InterpretButton analysisId={analysisId} label={label} options={available} />
    ) : null;
  if (interpretation === null) {
    if (available.length === 0) return <p className="text-note">{interpretableReason}</p>;
    return ask("Write an interpretation");
  }
  switch (interpretation.status) {
    case "queued":
    case "running":
      return (
        <div className="flex flex-col gap-3">
          <p className="text-note text-ink-soft">Writing the interpretation…</p>
          <MadeBy interpretation={interpretation} />
        </div>
      );
    case "verified":
      return (
        <div className="flex flex-col gap-4">
          <Verified interpretation={interpretation} level="h3" />
          <NewOne interpretation={interpretation}>{ask("Write a new interpretation")}</NewOne>
        </div>
      );
    case "refused":
      return (
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <p>{REFUSED}</p>
            <p className="text-note text-ink-soft">
              Attempts: {interpretation.attempts}.{interpretation.reason ? ` ${interpretation.reason}` : ""}
            </p>
            <MadeBy interpretation={interpretation} />
          </div>
          <NewOne interpretation={interpretation}>{ask("Write a new interpretation")}</NewOne>
        </div>
      );
    case "failed":
      return (
        <div className="flex flex-col gap-3">
          <p role="alert" className="text-note">
            {interpretation.error}
          </p>
          <MadeBy interpretation={interpretation} />
          {ask("Try again")}
        </div>
      );
  }
}

/**
 * Under a finished interpretation, the line saying it was written under an
 * earlier version of the prompt and rules, when it was, then `children`, the
 * button for a new one when there is an entry to ask with. One still being
 * written has no version yet, and a failed one keeps its own retry.
 */
function NewOne({
  interpretation,
  children,
}: {
  interpretation: InterpretationDetail;
  children: ReactNode;
}) {
  if (!interpretation.out_of_date && !children) return null;
  return (
    <div className="flex flex-col gap-3">
      {interpretation.out_of_date ? <p className="text-note text-ink-soft">{OUT_OF_DATE}</p> : null}
      {children}
    </div>
  );
}

/**
 * Which model, at which effort, the interpretation was asked of, said as far
 * as it got. A verified or refused one was written: the model the API
 * answered from when it is on record, the one asked for otherwise, and the
 * prompt when it is on record. One queued or running is still being written;
 * a failed one was only asked of the model.
 */
function MadeBy({ interpretation }: { interpretation: InterpretationDetail }) {
  const { model, effort } = interpretation;
  let line: string;
  switch (interpretation.status) {
    case "verified":
    case "refused":
      line = `Written by ${interpretation.returned_model ?? model}, ${effort} effort${
        interpretation.prompt_version ? `, prompt ${interpretation.prompt_version}` : ""
      }.`;
      break;
    case "queued":
    case "running":
      line = `Being written by ${model}, ${effort} effort.`;
      break;
    case "failed":
      line = `Asked of ${model}, ${effort} effort.`;
      break;
  }
  return <p className="value text-label text-ink-soft">{line}</p>;
}

/**
 * A verified interpretation: each section that has statements, under its
 * heading at `level`, then its limits, the notice on what "verified" means, and
 * the model, effort and prompt that wrote it. Its statements are never shown without
 * the limits and the notice.
 */
function Verified({
  interpretation,
  level,
}: {
  interpretation: InterpretationDetail;
  level: "h3" | "h4";
}) {
  const sections = (interpretation.sections ?? []).filter((s) => s.statements.length > 0);
  const Heading = level;
  return (
    <div className="flex flex-col gap-4">
      {sections.map((section) => (
        <div key={section.name}>
          <Heading className="title-group mb-2">{section.heading}</Heading>
          <ul className="flex list-disc flex-col gap-1 pl-5">
            {section.statements.map((statement, index) => (
              <li key={index}>{statement}</li>
            ))}
          </ul>
        </div>
      ))}
      {interpretation.limits ? <p className="text-note">{interpretation.limits}</p> : null}
      {interpretation.notice ? <p className="text-note">{interpretation.notice}</p> : null}
      <MadeBy interpretation={interpretation} />
    </div>
  );
}

/**
 * The analysis's earlier interpretations, newest first, closed until opened,
 * under their own heading so their sections' headings sit below it. Each shows
 * its status, the model and effort it was asked of (labelled as the choice
 * labels them), when it was asked for and the interpreter version it was
 * written under; opened, what it holds.
 */
function Earlier({ interpretations }: { interpretations: InterpretationDetail[] }) {
  const count = interpretations.length;
  return (
    // The heading stands outside the disclosure. Inside a <summary>, which is
    // exposed as a button, some screen readers do not announce it as a heading.
    <section aria-labelledby="earlier-interpretations-heading" className="flex flex-col gap-2">
      <h3 id="earlier-interpretations-heading" className="title-group">
        Earlier interpretations
      </h3>
      <details className="disclosure">
        <summary>{count === 1 ? "Show the earlier interpretation" : `Show the ${count} earlier interpretations`}</summary>
        <ol className="flex flex-col [&>li+li>.disclosure]:border-t-0">
          {interpretations.map((interpretation) => (
            // Keyed by the interpretation's own id, so an opened entry stays
            // open when a new interpretation pushes the list down.
            <li key={interpretation.id}>
              <details className="disclosure">
                <summary>
                  <span>
                    <span className="status">{interpretation.status}</span>,{" "}
                    <span className="value">{optionLabel(interpretation)}</span>,{" "}
                    <span className="value">{formatTime(interpretation.created_at)}</span>, version{" "}
                    <span className="value">{interpretation.interpreter_version ?? DASH}</span>
                  </span>
                </summary>
                <EarlierBody interpretation={interpretation} />
              </details>
            </li>
          ))}
        </ol>
      </details>
    </section>
  );
}

function EarlierBody({ interpretation }: { interpretation: InterpretationDetail }) {
  switch (interpretation.status) {
    case "verified":
      return <Verified interpretation={interpretation} level="h4" />;
    case "refused":
      return (
        <div className="flex flex-col gap-2">
          <p>{REFUSED}</p>
          <MadeBy interpretation={interpretation} />
        </div>
      );
    case "failed":
      return (
        <div className="flex flex-col gap-2">
          <p className="text-note">{interpretation.error}</p>
          <MadeBy interpretation={interpretation} />
        </div>
      );
    case "queued":
    case "running":
      return <MadeBy interpretation={interpretation} />;
  }
}
