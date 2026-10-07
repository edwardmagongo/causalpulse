import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  // `production.spec.ts` asks for what only the production stack has (a proxy,
  // TLS, rate limits), and is left out unless `make smoke-prod` asks for it.
  testIgnore: process.env.CP_E2E_PRODUCTION ? [] : ["**/production.spec.ts"],
  globalSetup: "./e2e/global-setup.ts",
  // The slowest test ran for 90 s in the composed stack; its own wait for the
  // analysis allows 300 s.
  timeout: 360_000,
  retries: 0,
  // The tests share one stack and one user, and the revocation test must run last.
  workers: 1,
  reporter: "list",
  use: {
    baseURL: process.env.CP_WEB_URL ?? "http://localhost:3000",
    // Unset in the gate, which uses Playwright's own Chromium. Set to "chrome"
    // to run against an installed browser instead.
    channel: process.env.CP_E2E_CHANNEL || undefined,
    // `make smoke-prod` sets this: the production stack's certificate on
    // `localhost` comes from the proxy's own authority, which no browser
    // trusts. Unset, a certificate is checked as usual.
    ignoreHTTPSErrors: process.env.CP_E2E_IGNORE_HTTPS_ERRORS === "true",
    trace: "retain-on-failure",
  },
});
