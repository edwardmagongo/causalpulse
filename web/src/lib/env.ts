/** The two settings the server cannot run without. Read at start and at use. */
export function sessionSecret(): string {
  const secret = process.env.CP_SESSION_SECRET;
  if (!secret || secret.length < 32) {
    throw new Error("CP_SESSION_SECRET must be set to at least 32 characters");
  }
  return secret;
}

export function apiUrl(): string {
  const url = process.env.CP_API_URL;
  if (!url) throw new Error("CP_API_URL must be set to the cp-api base URL");
  return url.replace(/\/$/, "");
}

/**
 * `CP_ENV=production` is set by the production compose file and by nothing
 * else. Everything it switches on is off without it.
 */
export function isProduction(): boolean {
  return process.env.CP_ENV === "production";
}

/**
 * Every session secret that appears in this repository. Anyone can read them,
 * so a production server does not start with one. `startup.test.ts` reads the
 * demo compose file, the Makefile and this directory, and fails if any of them
 * sets a secret that is missing here.
 */
export const DEMO_SESSION_SECRETS: ReadonlySet<string> = new Set([
  "local-demo-only-session-secret-not-for-deployment",
]);
