import Link from "next/link";

import { getSession } from "@/lib/session";

import { signOut } from "../sign-in/actions";

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const { name } = await getSession();
  return (
    <>
      <header>
        <nav aria-label="Main">
          <Link href="/">Analyses</Link>
          <Link href="/analyses/new">New analysis</Link>
        </nav>
        <form action={signOut}>
          <span>{name}</span>
          <button type="submit">Sign out</button>
        </form>
      </header>
      <main>{children}</main>
    </>
  );
}
