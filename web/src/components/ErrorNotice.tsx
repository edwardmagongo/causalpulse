"use client";

/**
 * Shown in place of a page that threw. The error's own message is not for the
 * reader and is never shown; the digest is, so a report can quote it.
 */
export function ErrorNotice({ digest, retry }: { digest: string | undefined; retry: () => void }) {
  return (
    <div role="alert" className="flex max-w-measure flex-col items-start gap-4">
      <h1 className="title-page">Something went wrong on this page.</h1>
      <p>Nothing was changed by this. You can try again.</p>
      <button type="button" onClick={() => retry()} className="button">
        Try again
      </button>
      {digest ? <p className="value text-label text-ink-soft">Reference: {digest}</p> : null}
    </div>
  );
}
