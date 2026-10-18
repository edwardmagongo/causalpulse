"use client";

import { useRouter } from "next/navigation";
import { type FormEvent, useId, useState, useTransition } from "react";

import { askForInterpretation } from "@/app/(app)/analyses/[id]/actions";
import type { InterpretationOption } from "@/lib/api/types";

/** How an entry of the server's list is named: "model, effort". */
function optionLabel(option: InterpretationOption): string {
  return `${option.model}, ${option.effort}`;
}

// A select's value for an entry; JSON, so no model id can run into its effort.
function keyOf(option: InterpretationOption): string {
  return JSON.stringify([option.model, option.effort]);
}

/**
 * Ask for the analysis's interpretation from one of `options`, the entries a
 * request would be accepted for, of which there is at least one. With more
 * than one, a choice comes before the button, set at first to the default,
 * or to the first entry when the default is not among them; with one, its
 * label stands beside the button and describes it. A refusal shows the API's
 * own sentence until the choice changes; an acceptance refreshes the page,
 * which then polls until the interpretation is written.
 */
export function InterpretButton({
  analysisId,
  label,
  options,
}: {
  analysisId: string;
  label: string;
  options: InterpretationOption[];
}) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);
  const preselected = options.find((option) => option.default) ?? options[0];
  const [picked, setPicked] = useState(keyOf(preselected));
  // A refreshed page can take the picked entry off the list; the preselected one stands in.
  const chosen = options.find((option) => keyOf(option) === picked) ?? preselected;
  const fieldId = useId();
  const entryId = useId();

  function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    startTransition(async () => {
      const result = await askForInterpretation(analysisId, {
        model: chosen.model,
        effort: chosen.effort,
      });
      if ("error" in result) {
        setError(result.error);
        return;
      }
      setError(null);
      router.refresh();
    });
  }

  return (
    <div className="flex flex-col items-start gap-3">
      <form onSubmit={ask} className="flex flex-wrap items-end gap-3">
        {options.length > 1 ? (
          <div className="flex flex-col gap-1.5">
            <label htmlFor={fieldId} className="label">
              Model and effort
            </label>
            <select
              id={fieldId}
              value={keyOf(chosen)}
              onChange={(event) => {
                setPicked(event.target.value);
                // A refusal was for the entry chosen before.
                setError(null);
              }}
              disabled={pending}
              className="field"
            >
              {options.map((option) => (
                <option key={keyOf(option)} value={keyOf(option)}>
                  {optionLabel(option)}
                </option>
              ))}
            </select>
          </div>
        ) : null}
        <button
          type="submit"
          disabled={pending}
          aria-describedby={options.length === 1 ? entryId : undefined}
          className="button"
        >
          {label}
        </button>
        {options.length === 1 ? (
          <span id={entryId} className="value self-center text-ink-soft">
            {optionLabel(options[0])}
          </span>
        ) : null}
      </form>
      {error ? (
        <p role="alert" className="text-note">
          {error}
        </p>
      ) : null}
    </div>
  );
}
