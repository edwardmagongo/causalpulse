"use client";

import { useActionState } from "react";

import { signIn, type SignInState } from "./actions";

const INITIAL: SignInState = { error: null };

export default function SignInPage() {
  const [state, action, pending] = useActionState(signIn, INITIAL);
  return (
    <main className="mx-auto flex min-h-dvh max-w-form flex-col justify-center px-gutter pt-10 pb-[18vh] md:px-0">
      <h1 className="title-page brand-mark flex items-center gap-3">CausalPulse</h1>
      <form action={action} className="mt-8 flex flex-col">
        <label htmlFor="key" className="label mb-1.5">
          API key
        </label>
        <input id="key" name="key" type="password" autoComplete="off" required className="field" />
        <p className="mt-3 text-note text-ink-soft">
          Keys are issued by <code>cp-api-admin issue-key</code>. The key is kept in a sealed,
          HTTP-only cookie and is never readable by the page.
        </p>
        <button type="submit" disabled={pending} className="button button-primary mt-6">
          Sign in
        </button>
        {state.error ? (
          <p role="alert" className="engine engine-alert mt-6">
            {state.error}
          </p>
        ) : null}
      </form>
    </main>
  );
}
