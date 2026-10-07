import { DEMO_SESSION_SECRETS, apiUrl, isProduction, sessionSecret } from "@/lib/env";

/**
 * Exit unless every required setting is present and, in production, safe: a
 * session secret of the deployment's own, and a cookie sent over HTTPS only.
 * A message names the setting and never its value: it goes to a log.
 *
 * A throw from the instrumentation hook does not stop the server: it stays
 * up and answers every request with a 500 (measured on Next.js 16.3.8). An
 * exit is what a container supervisor can see.
 */
export function requireSettings(): void {
  try {
    const secret = sessionSecret();
    apiUrl();
    if (isProduction()) {
      if (DEMO_SESSION_SECRETS.has(secret)) {
        throw new Error(
          "CP_SESSION_SECRET is a value published as a demo secret. " +
            "CP_ENV=production does not start with it: set the deployment's own.",
        );
      }
      if (process.env.CP_COOKIE_SECURE !== "true") {
        throw new Error(
          'CP_COOKIE_SECURE must be "true" when CP_ENV=production: ' +
            "the session cookie is then sent over HTTPS only.",
        );
      }
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  }
}
