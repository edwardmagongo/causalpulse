"use client";

import { useRouter } from "next/navigation";

/** Shown in place of data the API could not be reached for. Nothing stale is shown beside it. */
export function Unavailable() {
  const router = useRouter();
  return (
    <div role="status" data-unavailable="true" className="flex max-w-measure flex-col items-start gap-4">
      <p>The service is unavailable, so this page has nothing current to show.</p>
      <button type="button" onClick={() => router.refresh()} className="button">
        Try again
      </button>
    </div>
  );
}
