import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
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
    trace: "retain-on-failure",
  },
});
