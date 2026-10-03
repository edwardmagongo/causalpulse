import { execSync } from "node:child_process";
import { closeSync, mkdtempSync, openSync, rmSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

// Set by `make smoke-web`: two keys of one user, the fixtures' directory, and
// the command that revokes a key inside the api container.
const KEY = process.env.CP_E2E_KEY ?? "";
const REVOCABLE_KEY = process.env.CP_E2E_REVOCABLE_KEY ?? "";
const REVOKE = process.env.CP_E2E_REVOKE ?? "";
const DATA = resolve(process.env.CP_E2E_DATA ?? "../tests/fixtures/real_data");

const MB = 1024 * 1024;
// The API's limit (`cp_api.validation.MAX_UPLOAD_BYTES`).
const LIMIT = 50 * MB;

test.describe.configure({ mode: "serial" });

// The generated CSVs (about 112 MB in all) live in one directory, removed at the end.
let scratch = "";

test.beforeAll(() => {
  for (const [name, value] of Object.entries({ CP_E2E_KEY: KEY, CP_E2E_REVOCABLE_KEY: REVOCABLE_KEY, CP_E2E_REVOKE: REVOKE })) {
    if (!value) throw new Error(`${name} is not set; run this through \`make smoke-web\``);
  }
  scratch = mkdtempSync(join(tmpdir(), "cp-e2e-"));
});

test.afterAll(() => {
  if (scratch) rmSync(scratch, { recursive: true, force: true });
});

async function signIn(page: Page, key: string) {
  await page.goto("/sign-in");
  await page.getByLabel("API key").fill(key);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("heading", { name: "Analyses" })).toBeVisible();
}

/** A CSV of exactly `bytes` bytes: a header, then identical rows, then padding in the last row. */
function csvOf(bytes: number, name: string): string {
  const path = join(scratch, name);
  const header = "unit_id,treat,y\n";
  const row = "1,0,0.5\n";
  const rows = Math.floor((bytes - header.length) / row.length) - 1;
  const tail = bytes - header.length - rows * row.length;
  const file = openSync(path, "w");
  writeSync(file, header);
  const block = row.repeat(4096);
  let written = 0;
  while (written + 4096 <= rows) {
    writeSync(file, block);
    written += 4096;
  }
  writeSync(file, row.repeat(rows - written));
  // The last row is padded with digits to land on the exact size.
  writeSync(file, `1,0,0.${"5".repeat(tail - "1,0,0.\n".length)}\n`);
  closeSync(file);
  return path;
}

async function chooseColumn(page: Page, label: string, column: string) {
  await page.getByLabel(label, { exact: true }).selectOption(column);
}

async function pickEstimators(page: Page, names: string[]) {
  await page.getByRole("radio", { name: "Choose estimators" }).check();
  for (const name of names) {
    await page.getByRole("checkbox", { name, exact: true }).check();
  }
}

async function runToTheEnd(page: Page) {
  await page.getByRole("button", { name: "Run the analysis" }).click();
  await expect(page).toHaveURL(/\/analyses\/[0-9a-f-]{36}$/);
  // LaLonde took 20 s with the worker on the host and 90 s in the composed stack.
  await expect(page.locator("header [data-status]")).toHaveText("succeeded", { timeout: 300_000 });
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

test("LaLonde: upload, schema, two estimators, results and the evidence report", async ({ page }) => {
  await signIn(page, KEY);
  await page.goto("/analyses/new");
  await page.getByLabel("Upload a CSV").setInputFiles(join(DATA, "lalonde.csv"));
  await expect(page.getByText(/lalonde\.csv: \d+ columns/)).toBeVisible();

  await chooseColumn(page, "Unit", "unit_id");
  await chooseColumn(page, "Treatment", "treat");
  await chooseColumn(page, "Outcome", "re78");
  await chooseColumn(page, "Outcome type", "continuous");
  const covariates = page.getByRole("group", { name: "Covariates" });
  for (const column of ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]) {
    await covariates.getByRole("checkbox", { name: column, exact: true }).check();
  }
  // A column holding a single role is not offered as a covariate.
  await expect(covariates.getByRole("checkbox", { name: "re78", exact: true })).toHaveCount(0);
  await pickEstimators(page, ["difference_in_means", "lin_regression"]);
  await runToTheEnd(page);

  const estimates = page.getByRole("region", { name: "Estimates" });
  await expect(estimates.locator("tbody tr")).toHaveCount(2);
  await expect(estimates.locator("td[data-band]")).toHaveText([/^[A-Z]+$/, /^[A-Z]+$/]);
  // One panel: both methods estimate the ATE on the absolute scale.
  await expect(page.getByRole("region", { name: "Points and intervals" }).getByRole("img")).toHaveCount(1);
  await expect(page.getByRole("region", { name: "Comparison across methods" })).toContainText(
    "report intervals that",
  );

  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download the manifest" }).click();
  expect((await download).suggestedFilename()).toBe("manifest.json");

  await estimates.getByRole("row", { name: /lin_regression/ }).getByRole("link", { name: "Evidence report" }).click();
  const frame = page.locator("iframe");
  await expect(frame).toHaveAttribute("sandbox", "");
  await expect(page.frameLocator("iframe").getByRole("heading", { level: 1 })).toHaveText("Causal Pulse Report");
});

test("Proposition 99: panel fields, no covariates, synthetic control", async ({ page }) => {
  await signIn(page, KEY);
  await page.goto("/analyses/new");
  await page.getByLabel("Upload a CSV").setInputFiles(join(DATA, "prop99.csv"));
  await expect(page.getByText(/prop99\.csv: \d+ columns/)).toBeVisible();

  await chooseColumn(page, "Unit", "state");
  await chooseColumn(page, "Treatment", "treatment");
  await chooseColumn(page, "Outcome", "cigsale");
  await chooseColumn(page, "Outcome type", "continuous");
  await expect(page.getByText("None chosen: the analysis will be told there are no covariates")).toBeVisible();
  await page.getByText("Panel", { exact: true }).click();
  await chooseColumn(page, "Time", "year");
  await pickEstimators(page, ["synthetic_control"]);
  await runToTheEnd(page);

  await expect(page.getByRole("region", { name: "Estimates" })).toContainText("permutation interval");
  const routing = page.getByRole("region", { name: "Routing" });
  await expect(routing.getByText("not eligible", { exact: true })).toHaveCount(9);
  await expect(routing).toContainText("requires");
  await expect(page.getByRole("region", { name: "Comparison across methods" })).toContainText(
    "only one method estimated",
  );
});

test("a 12 MB upload arrives whole", async ({ page }) => {
  // Above the 1 MB a server action accepts and the 10 MB the proxy buffers.
  await signIn(page, KEY);
  await page.goto("/analyses/new");
  await page.getByLabel("Upload a CSV").setInputFiles(csvOf(12 * MB, "twelve.csv"));
  await expect(page.getByText("twelve.csv: 3 columns")).toBeVisible({ timeout: 60_000 });
});

test("an upload one byte over the limit shows the API's 413 message", async ({ page }) => {
  await signIn(page, KEY);
  await page.goto("/analyses/new");
  await page.getByLabel("Upload a CSV").setInputFiles(csvOf(LIMIT + 1, "over.csv"));
  await expect(
    page.getByRole("alert").filter({ hasText: `the upload exceeds the ${LIMIT}-byte limit` }),
  ).toBeVisible({ timeout: 60_000 });
});

test("an upload of exactly the limit is accepted", async ({ page }) => {
  // The control for the test above: the same generator, one byte smaller.
  await signIn(page, KEY);
  await page.goto("/analyses/new");
  await page.getByLabel("Upload a CSV").setInputFiles(csvOf(LIMIT, "exact.csv"));
  await expect(page.getByText("exact.csv: 3 columns")).toBeVisible({ timeout: 60_000 });
});

test("an upload posted from another origin, or from a null one, is refused", async ({ page }) => {
  // The signed-in page's request context sends the session cookie, so only the
  // Origin check stands between these posts and the API.
  await signIn(page, KEY);
  const multipart = { file: { name: "a.csv", mimeType: "text/csv", buffer: Buffer.from("a,b\n1,2\n") } };
  for (const origin of ["https://evil.example", "null"]) {
    const response = await page.request.post("/api/datasets", { headers: { origin }, multipart });
    expect(response.status(), `Origin: ${origin}`).toBe(403);
  }
});

test("the header's Sign out button ends the session", async ({ page }) => {
  await signIn(page, KEY);
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page).toHaveURL(/\/sign-in$/);
  expect((await page.context().cookies()).filter((cookie) => cookie.name === "cp_session")).toEqual([]);

  // Nothing is left to carry a visit past the gate.
  await page.goto("/");
  await expect(page).toHaveURL(/\/sign-in$/);
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
