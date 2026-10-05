"use client";

import { ErrorNotice } from "@/components/ErrorNotice";

/** Catches what a signed-in page throws. The shell's header stays. */
export default function AppError({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  return <ErrorNotice digest={error.digest} retry={retry} />;
}
