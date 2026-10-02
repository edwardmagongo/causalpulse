import { execSync } from "node:child_process";
import { resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

// Set by `make smoke-web`: two keys of one user, and the command that revokes
// a key inside the api container.
const KEY = process.env.CP_E2E_KEY ?? "";
const REVOCABLE_KEY = process.env.CP_E2E_REVOCABLE_KEY ?? "";
const REVOKE = process.env.CP_E2E_REVOKE ?? "";

test.describe.configure({ mode: "serial" });

test.beforeAll(() => {
  for (const [name, value] of Object.entries({ CP_E2E_KEY: KEY, CP_E2E_REVOCABLE_KEY: REVOCABLE_KEY, CP_E2E_REVOKE: REVOKE })) {
    if (!value) throw new Error(`${name} is not set; run this through \`make smoke-web\``);
  }
});

async function signIn(page: Page, key: string) {
  await page.goto("/sign-in");
  await page.getByLabel("API key").fill(key);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("heading", { name: "Analyses" })).toBeVisible();
}

test("a wrong key shows the API's message, and the issued key reaches the list", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/sign-in$/);
  await page.getByLabel("API key").fill("cpk_AAAAAAAAAAAA_wrong");
  await page.getByRole("button", { name: "Sign in" }).click();
  // Next.js adds its own empty role=alert (the route announcer), so filter by text.
  await expect(page.getByRole("alert").filter({ hasText: "missing or invalid API key" })).toBeVisible();

  await signIn(page, KEY);
});

test("a revoked key lands on sign-in at the next navigation", async ({ page }) => {
  await signIn(page, REVOCABLE_KEY);
  const keyId = REVOCABLE_KEY.split("_")[1];
  // From the repository root, where the compose file is.
  execSync(`${REVOKE} ${keyId}`, { stdio: "pipe", cwd: resolve("..") });

  await page.goto("/");
  await expect(page).toHaveURL(/\/sign-in$/);
  // The cookie is gone, not merely refused: a protected page redirects without calling the API.
  expect((await page.context().cookies()).filter((cookie) => cookie.name === "cp_session")).toEqual([]);
});
