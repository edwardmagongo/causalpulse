"use client";

import { useActionState } from "react";

import { signIn, type SignInState } from "./actions";

const INITIAL: SignInState = { error: null };

export default function SignInPage() {
  const [state, action, pending] = useActionState(signIn, INITIAL);
  return (
    <main>
      <h1>CausalPulse</h1>
      <form action={action}>
        <label htmlFor="key">API key</label>
        <input id="key" name="key" type="password" autoComplete="off" required />
        <p>
          Keys are issued by <code>cp-api-admin issue-key</code>. The key is kept in a sealed,
          HTTP-only cookie and is never readable by the page.
        </p>
        <button type="submit" disabled={pending}>
          Sign in
        </button>
        {state.error ? <p role="alert">{state.error}</p> : null}
      </form>
    </main>
  );
}
