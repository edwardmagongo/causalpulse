import { apiUrl, sessionSecret } from "@/lib/env";

/**
 * Exit unless every required setting is present.
 *
 * A throw from the instrumentation hook does not stop the server: it stays
 * up and answers every request with a 500 (measured on Next.js 16.3.8). An
 * exit is what a container supervisor can see.
 */
export function requireSettings(): void {
  try {
    sessionSecret();
    apiUrl();
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  }
}
