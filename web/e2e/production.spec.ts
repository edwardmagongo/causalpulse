import { execSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { expect, test, type Page, type Request } from "@playwright/test";

/**
 * What only the production stack can show: the proxy's redirect and headers,
 * the `Secure` cookie, each rate limit driven to its refusal, and an upload
 * refused before its body has been read.
 *
 * Run by `make smoke-prod`, after `workflow.spec.ts` and never before it: the
 * last test here leaves this machine's address refused at sign-in for the
 * rest of a quarter of an hour. `playwright.config.ts` leaves this file out
 * unless `CP_E2E_PRODUCTION` is set.
 *
 * The limits are per user, so everything that spends one is done as a second
 * user, `limits`, and `workflow.spec.ts`'s user is only read from. The tests
 * run in the order written and each leaves something the next relies on, but
 * the file is not in serial mode: one failure does not skip the rest, and
 * every test that needs an earlier one's state looks for it and says so.
 *
 * A stack this file has run against has spent its limits. Run it again only
 * against a new one.
 */

// The first user's surviving key: `workflow.spec.ts` revoked the other.
const KEY = process.env.CP_E2E_KEY ?? "";
// Two keys of the second user, and the command that revokes a key.
const LIMITS_KEY = process.env.CP_E2E_LIMITS_KEY ?? "";
const LIMITS_REVOCABLE_KEY = process.env.CP_E2E_LIMITS_REVOCABLE_KEY ?? "";
const REVOKE = process.env.CP_E2E_REVOKE ?? "";
// Where the proxy listens for plain HTTP.
const HTTP = process.env.CP_E2E_HTTP_URL ?? "";
const DATA = resolve(process.env.CP_E2E_DATA ?? "../tests/fixtures/real_data");

// The limits the stack was started with: the gate exports the stack's own variables.
const SIGNIN_FAILURES = Number(process.env.CP_LIMIT_SIGNIN_FAILURES);
const UPLOADS = Number(process.env.CP_LIMIT_UPLOADS_PER_DAY);
const ANALYSES = Number(process.env.CP_LIMIT_ANALYSES_PER_DAY);
const ACTIVE = Number(process.env.CP_LIMIT_ANALYSES_ACTIVE);

// The API's sentences (`cp_api.limits`), each ending in the time left in its window.
const WAIT = String.raw`\d+ (?:hour|minute|second)s?`;
const TOO_MANY_UPLOADS = new RegExp(String.raw`Too many uploads today for this account\. Try again in ${WAIT}\.`);
const TOO_MANY_ANALYSES = new RegExp(String.raw`Too many analyses today for this account\. Try again in ${WAIT}\.`);
const TOO_MANY_SIGNINS = new RegExp(String.raw`Too many failed sign-ins from this address\. Try again in ${WAIT}\.`);
const TOO_MANY_ACTIVE =
  `This account already has ${ACTIVE} ${ACTIVE === 1 ? "analysis" : "analyses"} waiting or running. ` +
  "Try again when one has finished.";

const MB = 1024 * 1024;
const ANALYSIS_URL = /\/analyses\/[0-9a-f-]{36}$/;

// No test here waits for a fit but the last of the analyses', which sets its own.
test.describe.configure({ timeout: 120_000 });

let scratch = "";
// A few rows, and 12 MB: more than the proxy or the dashboard holds in memory,
// so a refusal of it arrives while the browser is still sending.
let small = "";
let twelve = "";

function csvOf(bytes: number, name: string): string {
  const path = join(scratch, name);
  const row = "1,0,0.5\n";
  writeFileSync(path, `unit_id,treat,y\n${row.repeat(Math.ceil(bytes / row.length))}`);
  return path;
}

test.beforeAll(() => {
  const settings = {
    CP_E2E_KEY: KEY,
    CP_E2E_LIMITS_KEY: LIMITS_KEY,
    CP_E2E_LIMITS_REVOCABLE_KEY: LIMITS_REVOCABLE_KEY,
    CP_E2E_REVOKE: REVOKE,
    CP_E2E_HTTP_URL: HTTP,
  };
  for (const [name, value] of Object.entries(settings)) {
    if (!value) throw new Error(`${name} is not set; run this through \`make smoke-prod\``);
  }
  const limits = {
    CP_LIMIT_SIGNIN_FAILURES: SIGNIN_FAILURES,
    CP_LIMIT_UPLOADS_PER_DAY: UPLOADS,
    CP_LIMIT_ANALYSES_PER_DAY: ANALYSES,
    CP_LIMIT_ANALYSES_ACTIVE: ACTIVE,
  };
  for (const [name, value] of Object.entries(limits)) {
    if (!Number.isInteger(value) || value < 1) {
      throw new Error(`${name} is not a limit; run this through \`make smoke-prod\``);
    }
  }
  // The test of the active limit starts that many analyses in one day.
  if (ANALYSES <= ACTIVE) throw new Error("CP_LIMIT_ANALYSES_PER_DAY must be above CP_LIMIT_ANALYSES_ACTIVE");
  scratch = mkdtempSync(join(tmpdir(), "cp-e2e-prod-"));
  small = csvOf(160, "small.csv");
  twelve = csvOf(12 * MB, "twelve.csv");
});

test.afterAll(() => {
  if (scratch) rmSync(scratch, { recursive: true, force: true });
});

async function signIn(page: Page, key: string) {
  await page.goto("/sign-in");
  await page.getByLabel("API key").fill(key);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("heading", { name: "Analyses" })).toBeVisible({ timeout: 15_000 });
}

function isUpload(request: Request): boolean {
  return request.method() === "POST" && new URL(request.url()).pathname === "/api/datasets";
}

/** Choose a file in the upload field and return the status its request is answered with. */
async function upload(page: Page, file: string): Promise<number> {
  const field = page.getByLabel("Upload a CSV");
  // The field is disabled while an earlier upload is in flight.
  await expect(field).toBeEnabled({ timeout: 15_000 });
  const answered = page.waitForResponse((response) => isUpload(response.request()), { timeout: 30_000 });
  await field.setInputFiles(file);
  return (await answered).status();
}

/** Everything the browser is told about the page's uploads from here on, in order. */
function uploadsSeen(page: Page): string[] {
  const seen: string[] = [];
  page.on("response", (response) => {
    if (isUpload(response.request())) seen.push(`answered ${response.status()}`);
  });
  page.on("requestfailed", (request) => {
    if (isUpload(request)) seen.push(`failed (${request.failure()?.errorText ?? "no reason given"})`);
  });
  return seen;
}

/**
 * Run a check on an upload's outcome; if it fails, say what the browser saw
 * and what the page shows. What a browser does with an upload that is refused
 * while it is still being sent was not known when these tests were written.
 */
async function outcome(page: Page, seen: string[], check: () => Promise<void>): Promise<void> {
  try {
    await check();
  } catch (error) {
    const texts = page
      .locator('[role="alert"], [role="status"]')
      .allTextContents()
      .catch(() => ["the page could not be read"]);
    const stuck = new Promise<string[]>((done) => setTimeout(() => done(["the page did not answer"]), 5_000));
    const shown = (await Promise.race([texts, stuck])).filter((text) => text.trim() !== "");
    throw new Error(
      `The upload: ${seen.join(", then ") || "neither answered nor failed"}. ` +
        `The page: ${new URL(page.url()).pathname}, showing ${JSON.stringify(shown)}.\n${String(error)}`,
    );
  }
}

/** The new-analysis form, filled in for LaLonde from the user's earlier upload, ready to run. */
async function lalondeReady(page: Page) {
  await page.goto("/analyses/new");
  const earlier = page.getByLabel("or use an earlier upload");
  await expect(earlier, "the user has no earlier upload; the upload test leaves lalonde.csv").toBeVisible({
    timeout: 15_000,
  });
  const dataset = await earlier.locator("option", { hasText: "lalonde.csv" }).first().getAttribute("value");
  await earlier.selectOption(dataset ?? "");
  await expect(page.getByText(/lalonde\.csv: \d+ columns/)).toBeVisible({ timeout: 15_000 });
  // The analysis `workflow.spec.ts` runs: it took 20 s to 90 s there, long
  // enough to still be in flight when the next is submitted.
  await page.getByLabel("Unit", { exact: true }).selectOption("unit_id");
  await page.getByLabel("Treatment", { exact: true }).selectOption("treat");
  await page.getByLabel("Outcome", { exact: true }).selectOption("re78");
  await page.getByLabel("Outcome type", { exact: true }).selectOption("continuous");
  const covariates = page.getByRole("group", { name: "Covariates" });
  for (const column of ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]) {
    await covariates.getByRole("checkbox", { name: column, exact: true }).check();
  }
  await page.getByRole("radio", { name: "Choose estimators" }).check();
  for (const name of ["difference_in_means", "lin_regression"]) {
    await page.getByRole("checkbox", { name, exact: true }).check();
  }
}

function run(page: Page) {
  return page.getByRole("button", { name: "Run the analysis" }).click();
}

/** Wait until the signed-in user has fewer than `count` analyses queued or running. */
async function activeBelow(page: Page, count: number) {
  await expect(async () => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Analyses" })).toBeVisible();
    const active = page.locator('tbody [data-status="queued"], tbody [data-status="running"]');
    expect(await active.count()).toBeLessThan(count);
  }).toPass({ timeout: 300_000, intervals: [3_000] });
}

function expectTransportHeaders(headers: Record<string, string>, what: string) {
  // Exactly: no `includeSubDomains`, no `preload`, and not sent twice.
  expect(headers["strict-transport-security"], `Strict-Transport-Security on ${what}`).toBe("max-age=31536000");
  expect(headers["x-content-type-options"], `X-Content-Type-Options on ${what}`).toBe("nosniff");
}

// --- the proxy -----------------------------------------------------------------

test("plain HTTP answers with a permanent redirect to the same path over HTTPS", async ({ request }) => {
  const response = await request.get(`${HTTP}/study?level=scenario_by_size`, { maxRedirects: 0 });
  expect([301, 308]).toContain(response.status());
  // Not followed. The gate's HTTPS port is not 443, and the proxy leaves the
  // port out of the address it redirects to, which is right on a server.
  const location = new URL(response.headers()["location"] ?? "");
  expect(location.protocol).toBe("https:");
  expect(location.hostname).toBe(new URL(HTTP).hostname);
  expect(location.pathname).toBe("/study");
  expect(location.search).toBe("?level=scenario_by_size");
  // Nothing of the site is served over HTTP with it.
  expect(await response.text()).not.toContain("<html");
});

test("the sign-in page and the open study carry the transport, sniffing and upgrade headers", async ({ page }) => {
  for (const path of ["/sign-in", "/study"]) {
    const response = await page.goto(path);
    expect(response?.status(), path).toBe(200);
    const headers = response?.headers() ?? {};
    expectTransportHeaders(headers, path);
    expect(headers["content-security-policy"] ?? "", path).toContain("upgrade-insecure-requests");
  }
});

test("the study opens signed out with a Sign in link, and another page still redirects", async ({ page }) => {
  await page.goto("/study");
  await expect(page).toHaveURL(/\/study$/);
  await expect(page.getByRole("heading", { name: "Findings", exact: true })).toBeVisible();
  await expect(page.getByRole("banner").getByRole("link", { name: "Sign in", exact: true })).toBeVisible();

  await page.goto("/");
  await expect(page).toHaveURL(/\/sign-in$/);
});

test("the session cookie is Secure and HttpOnly", async ({ page }) => {
  await signIn(page, KEY);
  const cookies = (await page.context().cookies()).filter((cookie) => cookie.name === "cp_session");
  expect(cookies).toHaveLength(1);
  expect(cookies[0].secure).toBe(true);
  expect(cookies[0].httpOnly).toBe(true);
  expect(cookies[0].sameSite).toBe("Lax");
});

test("a signed-in page and a relayed download carry the transport and sniffing headers", async ({ page }) => {
  await signIn(page, KEY);
  const home = await page.goto("/");
  expectTransportHeaders(home?.headers() ?? {}, "the list of analyses");

  const finished = page.locator("tbody tr").filter({ has: page.locator('[data-status="succeeded"]') });
  await expect(
    finished.first(),
    "the first user has no finished analysis; workflow.spec.ts, which runs before this file, leaves two",
  ).toBeVisible({ timeout: 15_000 });
  const href = await finished.first().getByRole("link").getAttribute("href");
  expect(href).toMatch(ANALYSIS_URL);

  const manifest = await page.request.get(`/api${href}/manifest`);
  expect(manifest.status()).toBe(200);
  expect(manifest.headers()["content-disposition"]).toBe('attachment; filename="manifest.json"');
  expectTransportHeaders(manifest.headers(), "the manifest");
});

// --- the limits, as the second user --------------------------------------------

test("the day's uploads: the allowance is accepted, and the next upload shows the API's sentence", async ({ page }) => {
  await signIn(page, LIMITS_KEY);
  await page.goto("/analyses/new");

  // The first is the dataset the analyses below are run on.
  expect(await upload(page, join(DATA, "lalonde.csv")), "upload 1").toBe(201);
  await expect(page.getByText(/lalonde\.csv: \d+ columns/)).toBeVisible();
  // The same file again is the same dataset, answered 200, and still an upload.
  for (let n = 2; n <= UPLOADS; n += 1) {
    expect([200, 201], `upload ${n} of ${UPLOADS}`).toContain(await upload(page, small));
  }
  await expect(page.getByText("small.csv: 3 columns")).toBeVisible();

  expect(await upload(page, small), `upload ${UPLOADS + 1}`).toBe(429);
  await expect(page.getByRole("alert").filter({ hasText: TOO_MANY_UPLOADS })).toBeVisible();
  // The refused upload is not the chosen dataset.
  await expect(page.getByText("small.csv: 3 columns")).toHaveCount(0);
});

test("a 12 MB upload from a user over the limit is refused with the same sentence", async ({ page }) => {
  await signIn(page, LIMITS_KEY);
  await page.goto("/analyses/new");
  // Over the limit whatever the test above did.
  let refused = false;
  for (let n = 0; n <= UPLOADS && !refused; n += 1) refused = (await upload(page, small)) === 429;
  expect(refused, `whether any of ${UPLOADS + 1} uploads was refused`).toBe(true);

  // A page with no refusal on it yet.
  await page.goto("/analyses/new");
  const seen = uploadsSeen(page);
  await page.getByLabel("Upload a CSV").setInputFiles(twelve);
  await outcome(page, seen, async () => {
    await expect(page.getByRole("alert").filter({ hasText: TOO_MANY_UPLOADS })).toBeVisible({ timeout: 30_000 });
  });
  expect(seen).toEqual(["answered 429"]);
});

test("analyses in flight: one more than the limit shows the API's sentence", async ({ page, context }) => {
  await signIn(page, LIMITS_KEY);
  await activeBelow(page, 1);

  // One form for each analysis the limit allows and one more, all filled in
  // before any is run, so that the runs are a click apart.
  const forms = [page];
  for (let n = 0; n < ACTIVE; n += 1) forms.push(await context.newPage());
  for (const form of forms) await lalondeReady(form);

  for (const form of forms.slice(0, ACTIVE)) {
    await run(form);
    await expect(form).toHaveURL(ANALYSIS_URL, { timeout: 30_000 });
  }
  const last = forms[ACTIVE];
  await run(last);
  await expect(last.getByRole("alert").filter({ hasText: TOO_MANY_ACTIVE })).toBeVisible({ timeout: 30_000 });
  await expect(last).toHaveURL(/\/analyses\/new$/);
});

test("the day's analyses: the allowance is accepted, and the next shows the API's sentence", async ({ page }) => {
  // It waits for the analyses of the test above to finish, one after the other.
  test.setTimeout(900_000);
  await signIn(page, LIMITS_KEY);
  // Every analysis accepted is a row, and a refused one leaves none.
  const accepted = await page.locator("tbody tr").count();
  expect(accepted, "analyses already accepted today").toBeLessThanOrEqual(ANALYSES);

  for (let n = accepted + 1; n <= ANALYSES; n += 1) {
    // Under the other limit first, so that it is not the one that answers.
    await activeBelow(page, ACTIVE);
    await lalondeReady(page);
    await run(page);
    await expect(page, `analysis ${n} of ${ANALYSES}`).toHaveURL(ANALYSIS_URL, { timeout: 30_000 });
  }

  await activeBelow(page, ACTIVE);
  await lalondeReady(page);
  await run(page);
  await expect(page.getByRole("alert").filter({ hasText: TOO_MANY_ANALYSES })).toBeVisible({ timeout: 30_000 });
  await page.goto("/");
  await expect(page.locator("tbody tr")).toHaveCount(ANALYSES);
});

// --- refusals at sign-in: these two stay last, in this order --------------------

test("a 12 MB upload with a revoked key lands on sign-in", async ({ page }) => {
  await signIn(page, LIMITS_REVOCABLE_KEY);
  await page.goto("/analyses/new");
  await expect(page.getByLabel("Upload a CSV")).toBeEnabled({ timeout: 15_000 });
  const keyId = LIMITS_REVOCABLE_KEY.split("_")[1];
  // From the repository root, where the compose file is.
  execSync(`${REVOKE} ${keyId}`, { stdio: "pipe", cwd: resolve("..") });

  // The API answers 401 without reading the body, which counts as one failed
  // sign-in from this address; the user being over the upload limit is not
  // looked at, because the key comes first.
  const seen = uploadsSeen(page);
  await page.getByLabel("Upload a CSV").setInputFiles(twelve);
  await outcome(page, seen, async () => {
    await expect(page).toHaveURL(/\/sign-in$/, { timeout: 30_000 });
  });
  expect(seen).toEqual(["answered 401"]);
  expect((await page.context().cookies()).filter((cookie) => cookie.name === "cp_session")).toEqual([]);
});

test("failed sign-ins: the address is refused at the limit, and then a good key is refused too", async ({ page }) => {
  // The suite has already failed a few sign-ins from this address (the gate's
  // arithmetic is beside the limits in the Makefile), and the window they are
  // counted in may have ended since. Either way the limit is reached within
  // as many failures as it allows, and the attempt after that is refused.
  const invalid = page.getByRole("alert").filter({ hasText: "missing or invalid API key" });
  const refused = page.getByRole("alert").filter({ hasText: TOO_MANY_SIGNINS });
  let attempts = 0;
  for (let locked = false; !locked && attempts <= SIGNIN_FAILURES; attempts += 1) {
    // A new page each time, so that the message read is this attempt's.
    await page.goto("/sign-in");
    await page.getByLabel("API key").fill("cpk_AAAAAAAAAAAA_wrong");
    await page.getByRole("button", { name: "Sign in" }).click();
    await expect(invalid.or(refused)).toBeVisible({ timeout: 15_000 });
    locked = await refused.isVisible();
  }
  await expect(refused, `the sentence after ${attempts} wrong keys`).toBeVisible();

  // Refused before the key is looked at: a key that works is answered the same.
  await page.goto("/sign-in");
  await page.getByLabel("API key").fill(LIMITS_KEY);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(refused).toBeVisible({ timeout: 15_000 });
  await expect(page).toHaveURL(/\/sign-in$/);
});
