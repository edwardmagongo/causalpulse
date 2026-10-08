"use client";

import { useRouter } from "next/navigation";

import type { NotLoaded } from "@/lib/api/load";

/**
 * Shown in place of data a page could not load. Nothing stale is shown beside it.
 *
 * A read the API refused with 429 shows the API's own sentence; anything else
 * is the service not being reachable.
 */
export function Unavailable({ failure }: { failure: NotLoaded }) {
  const router = useRouter();
  return (
    <div role="status" data-unavailable="true" className="flex max-w-measure flex-col items-start gap-4">
      <p>
        {failure.reason === "limited"
          ? failure.detail
          : "The service is unavailable, so this page has nothing current to show."}
      </p>
      <button type="button" onClick={() => router.refresh()} className="button">
        Try again
      </button>
    </div>
  );
}
