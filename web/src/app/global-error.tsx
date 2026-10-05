"use client";

import { ErrorNotice } from "@/components/ErrorNotice";

import "./globals.css";

/** Catches a failure of the root layout itself, which it replaces: hence its own document. */
export default function GlobalError({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  return (
    <html lang="en">
      <body>
        <main className="mx-auto flex max-w-page flex-col gap-section px-gutter pt-10 pb-24">
          <ErrorNotice digest={error.digest} retry={retry} />
        </main>
      </body>
    </html>
  );
}
