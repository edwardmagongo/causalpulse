import Link from "next/link";

import { getSession } from "@/lib/session";

import { signOut } from "../sign-in/actions";

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const { name } = await getSession();
  return (
    <>
      <header className="border-b border-rule">
        <div className="mx-auto flex h-header max-w-page items-center justify-between gap-2 px-gutter sm:gap-4">
          <nav aria-label="Main" className="brand-mark flex flex-none items-center gap-2 text-note font-medium sm:gap-5">
            <Link href="/" className="hover:text-accent">
              Analyses
            </Link>
            <Link href="/study" aria-label="Benchmark study" className="hover:text-accent">
              {/* At 390 px the full label pushes the bar past one line. */}
              <span className="sm:hidden">Study</span>
              <span className="hidden sm:inline">Benchmark study</span>
            </Link>
            <Link href="/analyses/new" aria-label="New analysis" className="button button-primary">
              {/*
                Below 24rem (384 px) the full label leaves the bar no slack: it fitted at 320 px
                with Apple's system font by a pixel or two, and with the wider system font of a
                Linux desktop the page scrolled sideways by 2 px.
              */}
              <span className="min-[24rem]:hidden">New</span>
              <span className="hidden min-[24rem]:inline">New analysis</span>
            </Link>
          </nav>
          <form action={signOut} className="flex min-w-0 items-center gap-2 sm:gap-3">
            {/* Below 25.875rem (414 px) the name would get under 44 px, a sliver of two characters, so it goes. */}
            <span className="value min-w-0 truncate text-ink-soft max-[25.875rem]:hidden">{name}</span>
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
