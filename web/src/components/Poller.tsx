"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import type { AnalysisDetail } from "@/lib/api/types";
import { FIRST_DELAY_MS, inFlight, nextDelay } from "@/lib/polling";
import { leaveForSignOut } from "@/lib/sign-out";

/**
 * Poll an in-flight analysis and refresh the page when it finishes.
 *
 * Rendered only while the status is `queued` or `running`. A failed poll
 * says so and keeps going; a 401 leaves for `/sign-out`.
 */
export function Poller({ analysisId }: { analysisId: string }) {
  const router = useRouter();
  const [unreachable, setUnreachable] = useState(false);

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
          setUnreachable(false);
          const analysis = (await response.json()) as AnalysisDetail;
          if (!inFlight(analysis.status)) {
            router.refresh();
            return;
          }
        } else {
          setUnreachable(true);
        }
      } catch {
        if (stopped) return;
        setUnreachable(true);
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
    <p role="status" aria-live="polite">
      {unreachable
        ? "The service is unavailable. Still checking."
        : "Running. This page updates when the analysis finishes."}
    </p>
  );
}
