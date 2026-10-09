"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";

import { askForInterpretation } from "@/app/(app)/analyses/[id]/actions";

/**
 * Ask for the analysis's interpretation. A refusal shows the API's own
 * sentence; an acceptance refreshes the page, which then polls until the
 * interpretation is written.
 */
export function InterpretButton({ analysisId, label }: { analysisId: string; label: string }) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);

  function ask() {
    startTransition(async () => {
      const result = await askForInterpretation(analysisId);
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
      <button type="button" onClick={ask} disabled={pending} className="button">
        {label}
      </button>
      {error ? (
        <p role="alert" className="text-note">
          {error}
        </p>
      ) : null}
    </div>
  );
}
