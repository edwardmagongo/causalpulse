// Capture the component tests' fixtures from a running stack.
//
//   CP_API_URL=http://localhost:8000 CP_KEY=cpk_... node scripts/capture-fixtures.mjs
//
// The fixtures are real responses, not hand-written ones: a fixture written to
// agree with a component tests the one case that cannot fail. Run this again
// whenever a response model changes, and commit what it writes.
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const fixtures = resolve(here, "../src/test/fixtures");
const data = resolve(process.env.CP_FIXTURE_DATA ?? resolve(here, "../../tests/fixtures/real_data"));
const api = (process.env.CP_API_URL ?? "").replace(/\/$/, "");
const key = process.env.CP_KEY ?? "";
if (!api || !key) {
  console.error("set CP_API_URL and CP_KEY");
  process.exit(1);
}
const headers = { authorization: `Bearer ${key}` };

async function get(path) {
  const response = await fetch(`${api}${path}`, { headers });
  if (!response.ok) throw new Error(`GET ${path} answered ${response.status}`);
  return response.json();
}

// Upload a file from the data directory, or `content` under `filename` when given.
async function upload(filename, content) {
  const form = new FormData();
  form.append("file", new Blob([content ?? (await readFile(resolve(data, filename)))]), filename);
  const response = await fetch(`${api}/v1/datasets`, { method: "POST", headers, body: form });
  if (!response.ok) throw new Error(`upload of ${filename} answered ${response.status}`);
  return response.json();
}

async function analyse(datasetId, schema, estimators) {
  const response = await fetch(`${api}/v1/analyses`, {
    method: "POST",
    headers: { ...headers, "content-type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, schema, estimators }),
  });
  if (response.status !== 202) throw new Error(`POST /v1/analyses answered ${response.status}`);
  const { id } = await response.json();
  for (;;) {
    const analysis = await get(`/v1/analyses/${id}`);
    if (analysis.status === "succeeded" || analysis.status === "failed") return analysis;
    await new Promise((done) => setTimeout(done, 1000));
  }
}

async function save(name, value) {
  await writeFile(resolve(fixtures, name), `${JSON.stringify(value, null, 2)}\n`);
  console.log(`wrote ${name}`);
}

// The CSV text with one cell replaced: `column` of the first data row becomes
// `value`. The column is found by the header, and the result is compared with
// the original cell by cell: the same shape, and exactly one cell different.
function withCell(text, column, value) {
  if (text.includes('"')) throw new Error("the CSV has quoted fields; splitting on commas would be wrong");
  const cells = (csv) => csv.split("\n").map((line) => line.split(","));
  const before = cells(text);
  const at = before[0].indexOf(column);
  if (at === -1 || before[0].lastIndexOf(column) !== at) {
    throw new Error(`the CSV's header does not name ${column} exactly once`);
  }
  if (before.length < 2 || before[1].length !== before[0].length) {
    throw new Error("the CSV's first data row does not match its header");
  }
  const after = cells(
    before.map((row, index) => (index === 1 ? row.with(at, value) : row).join(",")).join("\n"),
  );
  if (after.length !== before.length || after.some((row, index) => row.length !== before[index].length)) {
    throw new Error("replacing the cell changed the CSV's shape");
  }
  const changed = after.flatMap((row, r) => row.flatMap((cell, c) => (cell === before[r][c] ? [] : [[r, c]])));
  if (changed.length !== 1 || changed[0][0] !== 1 || changed[0][1] !== at || after[1][at] !== value) {
    throw new Error(`expected exactly one changed cell, ${column} of the first data row; got ${changed.length}`);
  }
  return after.map((row) => row.join(",")).join("\n");
}

const LALONDE = {
  unit: "unit_id",
  treatment: "treat",
  outcome: "re78",
  covariates: ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"],
  outcome_type: "continuous",
};
const PROP99 = {
  unit: "state",
  treatment: "treatment",
  outcome: "cigsale",
  covariates: [],
  outcome_type: "continuous",
  time: "year",
};

const lalonde = await upload("lalonde.csv");
const prop99 = await upload("prop99.csv");

const compared = await analyse(lalonde.id, LALONDE, ["difference_in_means", "lin_regression"]);
await save("lalonde.analysis.json", compared);
await save("lalonde.summary.json", await get(`/v1/analyses/${compared.id}/summary`));

const single = await analyse(prop99.id, PROP99, ["synthetic_control"]);
await save("prop99.analysis.json", single);
await save("prop99.summary.json", await get(`/v1/analyses/${single.id}/summary`));

// A panel estimator on a cross-section: the router refuses, and the analysis fails.
await save("refused.analysis.json", await analyse(lalonde.id, LALONDE, ["twfe"]));

// One infinite covariate value: difference in means uses no covariate and
// succeeds, Lin regression fails while fitting, and the analysis is partial.
// The CSV is built in memory from the real file. This is requested before
// analyses.json is saved, on purpose: the list fixture then holds a partial row.
function notPartial(answer) {
  console.error(
    "the engine no longer turns an infinite covariate into a partial analysis " +
      "(wanted succeeded, partial, a summary for difference_in_means and an error for " +
      `lin_regression); the partial fixture needs another source. The answer was: ${answer}`,
  );
  process.exit(1);
}
const infiniteCsv = withCell(await readFile(resolve(data, "lalonde.csv"), "utf8"), "re75", "inf");
// A refusal of the upload or of the request is the same news as a clean answer.
const partial = await upload("lalonde-inf.csv", infiniteCsv)
  .then((infinite) => analyse(infinite.id, LALONDE, ["difference_in_means", "lin_regression"]))
  .catch((error) => notPartial(error.message));
const estimate = (method) => (partial.estimates ?? []).find((row) => row.method === method);
if (
  partial.status !== "succeeded" ||
  partial.partial !== true ||
  !estimate("difference_in_means")?.summary ||
  estimate("difference_in_means").error ||
  typeof estimate("lin_regression")?.error !== "string" ||
  !estimate("lin_regression").error ||
  estimate("lin_regression").summary
) {
  notPartial(
    JSON.stringify({
      status: partial.status,
      partial: partial.partial,
      error: partial.error,
      estimates: (partial.estimates ?? []).map((row) => ({
        method: row.method,
        summary: row.summary ? "present" : row.summary,
        error: row.error,
      })),
    }),
  );
}
await save("partial.analysis.json", partial);
await save("partial.summary.json", await get(`/v1/analyses/${partial.id}/summary`));

await save("estimators.json", await get("/v1/estimators"));
await save("analyses.json", await get("/v1/analyses"));
await save("lalonde.dataset.json", await get(`/v1/datasets/${lalonde.id}`));
