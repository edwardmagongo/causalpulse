import { execSync } from "node:child_process";
import { closeSync, mkdtempSync, openSync, readFileSync, rmSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { expect, test, type Page } from "@playwright/test";

// Set by `make smoke-web`: two keys of one user, the fixtures' directory, the
// command that revokes a key inside the api container, and the one that runs
// a Python script, given on standard input, there.
const KEY = process.env.CP_E2E_KEY ?? "";
const REVOCABLE_KEY = process.env.CP_E2E_REVOCABLE_KEY ?? "";
const REVOKE = process.env.CP_E2E_REVOKE ?? "";
const PLANT = process.env.CP_E2E_PLANT ?? "";
const DATA = resolve(process.env.CP_E2E_DATA ?? "../tests/fixtures/real_data");

const MB = 1024 * 1024;
// The API's limit (`cp_api.validation.MAX_UPLOAD_BYTES`).
const LIMIT = 50 * MB;

test.describe.configure({ mode: "serial" });

// The generated CSVs (about 112 MB in all) live in one directory, removed at the end.
let scratch = "";

test.beforeAll(() => {
  for (const [name, value] of Object.entries({
    CP_E2E_KEY: KEY,
    CP_E2E_REVOCABLE_KEY: REVOCABLE_KEY,
    CP_E2E_REVOKE: REVOKE,
    CP_E2E_PLANT: PLANT,
  })) {
    if (!value) throw new Error(`${name} is not set; run this through \`make smoke-web\``);
  }
  scratch = mkdtempSync(join(tmpdir(), "cp-e2e-"));
});

test.afterAll(() => {
  if (scratch) rmSync(scratch, { recursive: true, force: true });
});

// Every page is served under a Content-Security-Policy; a refusal by it is a console error.
let violations: string[] = [];

test.beforeEach(({ page }) => {
  violations = [];
  page.on("console", (message) => {
    if (/content security policy/i.test(message.text())) violations.push(message.text());
  });
});

test.afterEach(() => {
  expect(violations).toEqual([]);
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

/**
 * The cell of the study's matrix for one estimator at one scenario. The row is the one whose
 * row header is the estimator; the column is found from the header row that names the scenarios
 * (the sizes are a second header row at the by-size level, so the set of columns changes with the
 * level). The offset counts every cell of the header row up to the scenario's, and the body rows
 * are laid out the same way. A scenario run at several sizes would own several columns; this
 * returns its first.
 */
async function matrixCell(page: Page, metric: string, estimator: string, scenario: string) {
  const table = page.getByRole("table", { name: `${metric} by estimator and scenario` });
  const offset = await table.evaluate((element, wanted) => {
    const rows = Array.from(element.querySelectorAll("thead tr"));
    const named = rows.find((row) => row.querySelector("th[scope=colgroup]")) ?? rows[rows.length - 1];
    let offset = 0;
    for (const head of Array.from(named.children)) {
      if (head.tagName === "TH" && head.textContent?.trim() === wanted) return offset;
      offset += (head as HTMLTableCellElement).colSpan;
    }
    return -1;
  }, scenario);
  expect(offset, `a column for ${scenario}`).toBeGreaterThan(0);
  const row = table.getByRole("row").filter({ has: page.getByRole("rowheader", { name: estimator, exact: true }) });
  return row.locator("th, td").nth(offset);
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

/** The nonce in the `script-src` of a response's Content-Security-Policy; empty if there is none. */
function scriptNonce(headers: Record<string, string>): string {
  const policy = headers["content-security-policy"] ?? "";
  return /script-src [^;]*'nonce-([^']+)'/.exec(policy)?.[1] ?? "";
}

test("sign-in and a signed-in page each carry the policy, with a nonce of their own", async ({ page }) => {
  const signInPage = await page.goto("/sign-in");
  const first = scriptNonce(signInPage?.headers() ?? {});
  expect(first).not.toBe("");

  await signIn(page, KEY);
  const home = await page.goto("/");
  const second = scriptNonce(home?.headers() ?? {});
  expect(second).not.toBe("");
  expect(second).not.toBe(first);
});

test("a path that is no page redirects signed out, the study alone opens, and signed in it is a 404 under the policy", async ({ page }) => {
  await page.goto("/sign-in/nope");
  await expect(page).toHaveURL(/\/sign-in$/);

  // The benchmark study is the one page a visitor is let into, under the policy like any other.
  const study = await page.goto("/study");
  await expect(page).toHaveURL(/\/study$/);
  expect(study?.status()).toBe(200);
  expect(scriptNonce(study?.headers() ?? {})).not.toBe("");
  await expect(page.getByRole("heading", { name: "Findings", exact: true })).toBeVisible();
  const header = page.getByRole("banner");
  await expect(header.getByRole("link", { name: "Benchmark study", exact: true })).toBeVisible();
  await expect(header.getByRole("link", { name: "Sign in", exact: true })).toHaveAttribute("href", "/sign-in");
  // Counted, not looked for by text: the visitor's header has no button at all, and two links.
  await expect(header.getByRole("button")).toHaveCount(0);
  await expect(header.getByRole("link")).toHaveCount(2);
  await expect(page.getByText("Sign out")).toHaveCount(0);
  await page.waitForLoadState("networkidle");
  // A sub-path of it is gated like any other page.
  await page.goto("/study/nope");
  await expect(page).toHaveURL(/\/sign-in$/);

  await signIn(page, KEY);
  const missing = await page.goto("/no-such-page");
  expect(missing?.status()).toBe(404);
  expect(scriptNonce(missing?.headers() ?? {})).not.toBe("");
  // Let the page's scripts run, so that a refusal by the policy is logged before the test ends.
  await page.waitForLoadState("networkidle");

  const analysis = await page.goto("/analyses/00000000-0000-0000-0000-000000000000");
  expect(analysis?.status()).toBe(404);
  await page.waitForLoadState("networkidle");
});

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

  // The name comes from the analysis itself, not from a second call for the dataset.
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Analysis of lalonde.csv");
  const estimates = page.getByRole("region", { name: "Estimates" });
  await expect(estimates.locator("tbody tr")).toHaveCount(2);
  await expect(estimates).toContainText(/\d+% bootstrap interval/);
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

// The Proposition 99 analysis's id, for the interpretations planted on it below.
let prop99 = "";

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
  // The stack the browser tests run against writes no interpretations, so a
  // finished analysis has no interpretation section at all.
  await expect(page.getByRole("region", { name: /Interpretation/ })).toHaveCount(0);
  prop99 = new URL(page.url()).pathname.split("/").pop() ?? "";
});

test("planted interpretations: a current verified one leads, the newer failed one follows, the rest are earlier", async ({ page }) => {
  // The id goes into a shell command, so it must be exactly an id.
  expect(prop99).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
  // From the repository root, where the compose file is.
  execSync(`${PLANT} ${prop99}`, {
    input: readFileSync(resolve("e2e/plant-interpretations.py")),
    stdio: ["pipe", "pipe", "pipe"],
    cwd: resolve(".."),
  });
  await signIn(page, KEY);

  // Rendered on the server: a helper a server component cannot call fails here, not in vitest.
  const response = await page.goto(`/analyses/${prop99}`);
  expect(response?.status()).toBe(200);
  const section = page.getByRole("region", {
    name: "Interpretation (model-written, verified against the facts)",
  });
  await expect(section.getByText("A planted narrative, written by no model.")).toBeVisible();
  const newer = section.getByRole("region", { name: "Newer interpretation" });
  await expect(newer.getByRole("alert")).toHaveText("planted: the newer interpretation failed");
  await expect(newer.getByText("Asked of claude-sonnet-5-5, low effort.")).toBeVisible();
  // This stack writes no interpretations, so nothing is offered.
  await expect(section.getByRole("button")).toHaveCount(0);

  await section.getByText("Show the earlier interpretation", { exact: true }).click();
  const earlier = section.getByRole("region", { name: "Earlier interpretations" });
  await expect(earlier.getByText(/^failed, claude-opus-5-5, medium, /)).toBeVisible();
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

test("the header fits a phone: one line, no sideways scroll, every control in it, the name only where there is room", async ({ page }) => {
  await signIn(page, KEY);
  const header = page.getByRole("banner");
  const nav = header.getByRole("navigation", { name: "Main" });
  const controls = [
    nav.getByRole("link", { name: "Analyses", exact: true }),
    // Its visible text is shortened on a narrow screen; its accessible name is not.
    nav.getByRole("link", { name: "Benchmark study", exact: true }),
    nav.getByRole("link", { name: "New analysis", exact: true }),
    header.getByRole("button", { name: "Sign out", exact: true }),
  ];
  // The signed-in name: the only span in the sign-out form, hidden by a class below 414 px.
  const name = header.locator("form span");

  // Sign-in leaves the page at the suite's own size, where the bar is one line: its height there is the measure.
  const oneLine = (await header.boundingBox())?.height;
  expect(oneLine, "the header's height at the suite's size").toBeGreaterThan(0);

  for (const width of [320, 360, 390]) {
    await page.setViewportSize({ width, height: 800 });
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Analyses" })).toBeVisible();
    const scroll = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }));
    expect(scroll.scrollWidth, `the document does not scroll sideways at ${width} px`).toBe(scroll.clientWidth);
    // A bar that wrapped onto a second line would be taller, or would spill out of its own box.
    const bar = await header.boundingBox();
    expect(bar, `the header's box at ${width} px`).not.toBeNull();
    expect(bar!.height, `the header's height at ${width} px`).toBe(oneLine);
    expect(bar!.x, `the header's left edge at ${width} px`).toBeGreaterThanOrEqual(0);
    expect(bar!.x + bar!.width, `the header's right edge at ${width} px`).toBeLessThanOrEqual(width);
    for (const control of controls) {
      await expect(control).toBeVisible();
      const box = await control.boundingBox();
      const label = `${await control.textContent()} at ${width} px: ${JSON.stringify(box)} in ${JSON.stringify(bar)}`;
      expect(box, label).not.toBeNull();
      expect(box!.x, label).toBeGreaterThanOrEqual(bar!.x);
      expect(box!.y, label).toBeGreaterThanOrEqual(bar!.y);
      expect(box!.x + box!.width, label).toBeLessThanOrEqual(bar!.x + bar!.width);
      expect(box!.y + box!.height, label).toBeLessThanOrEqual(bar!.y + bar!.height);
    }
    // It is in the page and hidden, not missing: a locator that matched nothing would be hidden too.
    await expect(name).toHaveCount(1);
    await expect(name).toBeHidden();
  }

  // The suite's own size, where there is room for the name.
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto("/");
  await expect(name).toHaveCount(1);
  await expect(name).toBeVisible();
  await expect(name).not.toHaveText("");
});

test("the benchmark study opens from the header, and its view lives in the URL", async ({ page }) => {
  await signIn(page, KEY);
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Benchmark study", exact: true }).click();
  await expect(page).toHaveURL(/\/study$/);
  await expect(page.getByRole("heading", { name: "Findings", exact: true })).toBeVisible();

  const choose = (group: string, name: string) =>
    page.getByRole("group", { name: group, exact: true }).getByRole("link", { name, exact: true });
  await choose("Metric family", "coverage").click();
  await choose("Metric", "coverage_permutation").click();
  const table = page.getByRole("table", { name: "coverage_permutation by estimator and scenario" });
  const colgroups = table.locator('th[scope="colgroup"]');
  // The figure in the published report's coverage table: S2 ran at one size, so it is the same at both levels.
  // The value sits in the cell's first line; the lines under it hold the error and the fits.
  const value = async () =>
    (await matrixCell(page, "coverage_permutation", "synthetic_control", "S2")).locator("span.block:not(.label)");
  await expect(await value()).toHaveText("0.86");
  // At the first level there is no row of scenario headers over a row of sizes.
  await expect(colgroups).toHaveCount(0);

  await choose("Level", "By scenario and size").click();
  await expect(choose("Level", "By scenario and size")).toHaveAttribute("aria-current", "true");
  await expect(page).toHaveURL((url) => url.searchParams.get("level") === "scenario_by_size");
  const query = new URL(page.url()).searchParams;
  expect(query.get("family")).toBe("coverage");
  expect(query.get("metric")).toBe("coverage_permutation");
  expect(query.get("level")).toBe("scenario_by_size");
  await expect(await value()).toHaveText("0.86");
  // The matrix itself changed: S2 heads one group, and the size under it is the one S2 ran at.
  await expect(colgroups.filter({ hasText: /^S2$/ })).toHaveCount(1);
  const column = await (await matrixCell(page, "coverage_permutation", "synthetic_control", "S2")).evaluate(
    (cell) => (cell as HTMLTableCellElement).cellIndex,
  );
  await expect(table.locator("thead tr").nth(1).locator("th").nth(column)).toHaveText("18 \u00d7 24");

  // Every switch is a link, so the browser's back steps to the view before it.
  await page.goBack();
  await expect(choose("Level", "By scenario")).toHaveAttribute("aria-current", "true");
  await expect(colgroups).toHaveCount(0);
  const back = new URL(page.url()).searchParams;
  expect(back.get("family")).toBe("coverage");
  expect(back.get("metric")).toBe("coverage_permutation");
  expect(back.has("level")).toBe(false);
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
