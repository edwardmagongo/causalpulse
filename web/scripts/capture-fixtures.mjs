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

async function upload(filename) {
  const form = new FormData();
  form.append("file", new Blob([await readFile(resolve(data, filename))]), filename);
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

await save("estimators.json", await get("/v1/estimators"));
await save("analyses.json", await get("/v1/analyses"));
await save("lalonde.dataset.json", await get(`/v1/datasets/${lalonde.id}`));
