import type { InterpretationDetail } from "@/lib/api/types";

import { InterpretButton } from "./InterpretButton";

const VERIFIED_HEADING = "Interpretation (model-written, verified against the facts)";
const REFUSED =
  "The model's narrative could not be verified against this analysis's facts, so it is not shown.";

/**
 * The analysis's interpretation, or how to get one.
 *
 * Every sentence about the analysis comes from the API: the statements, the
 * limits and what "verified" means were written by the system and stored by
 * the worker. What this writes is the interface around them. With
 * interpretations off on the server, and none on record, there is no section.
 */
export function InterpretationPanel({
  analysisId,
  interpretation,
  interpretable,
  interpretableReason,
}: {
  analysisId: string;
  interpretation: InterpretationDetail | null;
  interpretable: boolean;
  interpretableReason: string | null;
}) {
  if (interpretation === null && !interpretable && interpretableReason === null) return null;
  const verified = interpretation?.status === "verified";
  return (
    <section aria-labelledby="interpretation-heading" className="flex max-w-measure flex-col gap-4">
      <h2 id="interpretation-heading" className="title-section">
        {verified ? VERIFIED_HEADING : "Interpretation"}
      </h2>
      <Body
        analysisId={analysisId}
        interpretation={interpretation}
        interpretable={interpretable}
        interpretableReason={interpretableReason}
      />
    </section>
  );
}

function Body({
  analysisId,
  interpretation,
  interpretable,
  interpretableReason,
}: {
  analysisId: string;
  interpretation: InterpretationDetail | null;
  interpretable: boolean;
  interpretableReason: string | null;
}) {
  if (interpretation === null) {
    if (!interpretable) return <p className="text-note">{interpretableReason}</p>;
    return <InterpretButton analysisId={analysisId} label="Write an interpretation" />;
  }
  switch (interpretation.status) {
    case "queued":
    case "running":
      return <p className="text-note text-ink-soft">Writing the interpretation…</p>;
    case "verified":
      return <Verified interpretation={interpretation} />;
    case "refused":
      return (
        <div className="flex flex-col gap-2">
          <p>{REFUSED}</p>
          <p className="text-note text-ink-soft">
            Attempts: {interpretation.attempts}.{interpretation.reason ? ` ${interpretation.reason}` : ""}
          </p>
        </div>
      );
    case "failed":
      return (
        <div className="flex flex-col gap-3">
          <p role="alert" className="text-note">
            {interpretation.error}
          </p>
          {interpretable ? <InterpretButton analysisId={analysisId} label="Try again" /> : null}
        </div>
      );
  }
}

function Verified({ interpretation }: { interpretation: InterpretationDetail }) {
  const sections = (interpretation.sections ?? []).filter((s) => s.statements.length > 0);
  return (
    <div className="flex flex-col gap-4">
      {sections.map((section) => (
        <div key={section.name}>
          <h3 className="title-group mb-2">{section.heading}</h3>
          <ul className="flex list-disc flex-col gap-1 pl-5">
            {section.statements.map((statement, index) => (
              <li key={index}>{statement}</li>
            ))}
          </ul>
        </div>
      ))}
      {interpretation.limits ? <p className="text-note">{interpretation.limits}</p> : null}
      {interpretation.notice ? <p className="text-note">{interpretation.notice}</p> : null}
      <p className="value text-label text-ink-soft">
        Written by {interpretation.returned_model ?? interpretation.model}
        {interpretation.prompt_version ? `, prompt ${interpretation.prompt_version}` : ""}.
      </p>
    </div>
  );
}
