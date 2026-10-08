"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { detailOf } from "@/lib/api/errors";
import type { AnalysisDetail } from "@/lib/api/types";
import { FIRST_DELAY_MS, inFlight, nextDelay } from "@/lib/polling";
import { leaveForSignOut } from "@/lib/sign-out";

/** What the last poll came to: an answer, no service, or the API's sentence for a 429. */
type Trouble = { kind: "none" } | { kind: "unreachable" } | { kind: "limited"; detail: string };

const NONE: Trouble = { kind: "none" };
const UNREACHABLE: Trouble = { kind: "unreachable" };

function statusLine(trouble: Trouble): string {
  if (trouble.kind === "limited") return `${trouble.detail} Still checking.`;
  if (trouble.kind === "unreachable") return "The service is unavailable. Still checking.";
  return "In progress. This page updates when the analysis finishes.";
}

/**
 * Poll an in-flight analysis and refresh the page when it finishes.
 *
 * Rendered only while the status is `queued` or `running`. A failed poll
 * says so and keeps going; a 401 leaves for `/sign-out`. A poll answered 429
 * shows the API's own sentence, which says when to try again, and keeps going
 * at the same lengthening pace as any other.
 */
export function Poller({ analysisId }: { analysisId: string }) {
  const router = useRouter();
  const [trouble, setTrouble] = useState<Trouble>(NONE);

  useEffect(() => {
    let delay = FIRST_DELAY_MS;
    let timer: ReturnType<typeof setTimeout>;
    let stopped = false;

    async function poll() {
      try {
        const response = await fetch(`/api/analyses/${analysisId}`, { cache: "no-store" });
        if (stopped) return;
        if (response.status === 401) {
          leaveForSignOut();
          return;
        }
        if (response.ok) {
          setTrouble(NONE);
          const analysis = (await response.json()) as AnalysisDetail;
          // The page may have been left while the body was arriving.
          if (stopped) return;
          if (!inFlight(analysis.status)) {
            router.refresh();
            return;
          }
        } else if (response.status === 429) {
          const body: unknown = await response.json().catch(() => null);
          if (stopped) return;
          setTrouble({ kind: "limited", detail: detailOf(body, 429) });
        } else {
          setTrouble(UNREACHABLE);
        }
      } catch {
        if (stopped) return;
        setTrouble(UNREACHABLE);
      }
      delay = nextDelay(delay);
      timer = setTimeout(poll, delay);
    }

    timer = setTimeout(poll, delay);
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [analysisId, router]);

  return (
    <p role="status" aria-live="polite" className="-mt-6 text-note text-ink-soft">
      {statusLine(trouble)}
    </p>
  );
}
