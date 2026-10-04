import Link from "next/link";

import { getSession } from "@/lib/session";

import { signOut } from "../sign-in/actions";

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const { name } = await getSession();
  return (
    <>
      <header className="border-b border-rule">
        <div className="mx-auto flex h-header max-w-page items-center justify-between gap-4 px-gutter">
          <nav aria-label="Main" className="brand-mark flex flex-none items-center gap-3 text-note font-medium sm:gap-5">
            <Link href="/" className="hover:text-accent">
              Analyses
            </Link>
            <Link href="/study" aria-label="Benchmark study" className="hover:text-accent">
              {/* At 390 px the full label pushes the bar past one line. */}
              <span className="sm:hidden">Study</span>
              <span className="hidden sm:inline">Benchmark study</span>
            </Link>
            <Link href="/analyses/new" className="button button-primary">
              New analysis
            </Link>
          </nav>
          <form action={signOut} className="flex min-w-0 items-center gap-3">
            <span className="value min-w-0 truncate text-ink-soft">{name}</span>
            <button type="submit" className="link flex-none cursor-pointer text-note">
              Sign out
            </button>
          </form>
        </div>
      </header>
      <main className="mx-auto flex max-w-page flex-col gap-section px-gutter pt-10 pb-24">
        {children}
      </main>
    </>
  );
}
