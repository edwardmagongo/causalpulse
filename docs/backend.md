# Backend

A local demo stack: upload a CSV, declare its schema, and get back estimates, validity bands, and
evidence reports over HTTP. One machine, `docker compose up`; every route except `/v1/healthz` needs a
per-user API key; it runs on one machine, not as a hosted service.

```bash
docker compose up --build
```

This starts six services: `postgres`, `redis`, a one-shot `migrate` that runs the Alembic
migrations and exits, `api`, which waits on `migrate` and serves on `http://localhost:8000`, a
`worker`, which reads jobs off the same Redis queue the API enqueues to, and `web`, the dashboard,
on `http://localhost:3000` (see [Web dashboard](dashboard.md)). Ports (host side) default to `8000` for the
API, `5433` for Postgres, `6380` for Redis, and `3000` for `web`; override with `CP_API_PORT`,
`CP_POSTGRES_PORT`, `CP_REDIS_PORT`, `CP_WEB_PORT` if those clash with something else already
running.

Each analysis has a 30-minute limit; one that runs past it is stopped and marked `failed`, with
an error stating the limit. The limit takes effect between Python-level steps, so a single hung
call into compiled code (XLA, LAPACK) is not interrupted, and the restart policy below does not
help there either, because the worker process is still alive. `api` and `worker` restart unless
stopped. That matters for a worker
that dies mid-job: RQ only marks a dead worker's job failed when a live worker cleans up its
registries, so a killed worker's analysis shows `running` until a worker restarts and the job's
time limit has passed, and only then becomes "worker lost".

## Routes

All routes are under `/v1`. Every route except `GET /healthz` needs an `Authorization: Bearer <key>` header.

| Route | Behaviour |
|---|---|
| `POST /datasets` | Multipart CSV, ≤ 50 MB (413 above). The key is checked before any of the body is read (see "Users and API keys"). Your own copy of the same bytes, uploaded twice, returns the existing row by sha256; another user's identical upload is a separate row. |
| `POST /analyses` | `{dataset_id, schema, estimators?, random_seed}`. Validates the schema via `CausalSchema` and its columns against the CSV header; enqueues; 202 with the id. |
| `GET /me` | `{name}`: whose key this is. |
| `GET /estimators` | The estimator catalogue: each name with its family (`cross_sectional`, `panel`, `donor_pool`). A family is a grouping, not a statement about eligibility. |
| `GET /datasets` | Your datasets, newest first, without their columns. `limit` (1 to 200, default 50) and `cursor`, the previous page's `next_cursor`. |
| `GET /datasets/{id}` | One dataset with its columns. |
| `GET /analyses` | Your analyses, newest first, paged like the datasets. An in-flight analysis whose job is gone is marked `failed` here, as on the single read. |
| `GET /analyses/{id}` | Status, `partial`, error, the dataset's id and `dataset_filename`, routing decisions, and per estimate its method, band, error, and `summary` (point, standard error, intervals, the bootstrap interval's level, p-value, group sizes). It also carries `interpretation`, the newest interpretation (null until one is asked for), and `earlier_interpretations`, the others, newest first (empty when there are none). Each has its own `id`, its status, `model` and `effort` (the entry it was asked for), `interpreter_version` (the prompt and verifier rules it was written under, null when none was recorded), `out_of_date` (true for a verified or refused interpretation when that is not the server's current version, null included; false for a queued, running or failed one) and the fields its status fills. `interpretation_options` lists the entries of the server's `CP_INTERPRETATION_OPTIONS`, in its order, each with `model`, `effort`, `default` (true for the first), `available` (whether a request for that entry would be accepted, the day's limit aside) and `reason` (the 409 sentence such a request would get, null when it is available); it is empty when the server has interpretations off. An entry is available when the analysis succeeded and has a fact table, none of its interpretations is queued or running, and its newest interpretation from that entry is absent, failed, or verified or refused under an earlier version. `interpretable` is true when some entry is available. `interpretable_reason` is null when some entry is available or the server has interpretations off, and otherwise the default entry's `reason`. |
| `POST /analyses/{id}/interpretation` | Asks for a new model-written interpretation of a finished analysis. An optional JSON body, `{"model", "effort"}`, names one entry of the server's `CP_INTERPRETATION_OPTIONS`; with no body, or naming neither, the request is for the first entry, the default. 202 with the interpretation, queued. Every accepted request writes a new row and counts against the daily limit; earlier interpretations are kept. 403 if the server has interpretations off; 422 if the body names the model or the effort without the other, or an entry that is not on the list; 409 if the analysis has not succeeded or has no fact table, if any interpretation of it is already being written, or if its newest interpretation from that model and effort is verified or refused under the current prompt and rules (a failed one, or one written under an earlier version, can be replaced, and another entry can be asked for); 429 over the daily limit. See [Choosing the model and effort](interpretation.md#choosing-the-model-and-effort) and [Interpreting again](interpretation.md#interpreting-again-and-the-earlier-interpretations). |
| `GET /analyses/{id}/estimates/{method}/report.html` | One estimate's evidence report, figures inlined. |
| `GET /analyses/{id}/estimates/{method}/report.md` | The Markdown report. Its relative figure links resolve against the next route. |
| `GET /analyses/{id}/estimates/{method}/{filename}` | One Markdown figure, from `report_assets`. |
| `GET /analyses/{id}/manifest` | The replayable `AnalysisManifest`. |
| `GET /analyses/{id}/summary` | `{"comparisons": ...}`: the comparison groups and their verdicts. |
| `GET /healthz` | Postgres and Redis reachable. |

Migration `0004` gives each interpretation its own id, so an analysis keeps every interpretation it
is given, and adds `interpreter_version`, null on every existing row; the verified and refused ones
therefore count as out of date, and an analysis interpreted before the upgrade can be interpreted
again (a failed one could be asked for again in any case). Its downgrade refuses,
before it changes anything, while any analysis has more than one interpretation: the key it goes
back to holds one row per analysis, so the others would be lost.

## Users and API keys

Every route except `/v1/healthz` needs a key, and every dataset and analysis belongs to the user
whose key created it. Another user's resource answers 404, exactly like one that does not exist.
A request without a valid key answers 401; the one exception is a request whose body is malformed
JSON, which answers 422 even with no key, because FastAPI decodes the JSON before it checks the key.

The upload is the other way round. `POST /v1/datasets` checks the key, and then the day's upload
limit when one is set ([Production stack](production.md)), before it reads any of the body. A missing,
malformed or revoked key is answered 401, and a user over the limit 429, with the body unsent:
`tests/services/test_upload_order.py` sends the headers of a 12 MB upload and none of its body,
over a real socket, and gets the answer. Until this was changed the route read the whole
multipart body, up to 50 MB, before it looked at the key.

The route also counts the request body as it arrives, which it did not do before: a file part of
any length used to be written to a temporary file in full before the 413. The bound is on the
whole body, the 50 MiB limit plus 1 MiB for the form's own framing, 53,477,376 bytes. Past it
nothing more is kept and the answer is 413 `the upload exceeds the 52428800-byte limit`. The rest
of the body is read and thrown away for at most a further 50 MiB, so that a client sending a file
a little too large gets the answer after its request and not in the middle of it; past that the
API answers anyway. Because the bound is on the body and not on the file, two requests are
answered differently than before:

| Request | Before | Now |
|---|---|---|
| A file within the limit beside other form fields, each field within the parser's own 1 MB, the whole body over the bound | 201, the file stored | 413 `the upload exceeds the 52428800-byte limit` |
| A body that is not multipart (URL-encoded), with no `file` field, over the bound | 422, FastAPI's `Field required` for `file` | 413, the same message |

The message names the file's limit although the file itself may be small.
`test_a_body_over_the_bound_is_413_whatever_it_holds` pins both rows. A single field over 1 MB is
still Starlette's own 400, as before.

```bash
docker compose exec api cp-api-admin create-user alice
export CP_KEY=$(docker compose exec -T api cp-api-admin issue-key alice)
```

The key is printed once and only its hash is stored. Revoke it with
`docker compose exec api cp-api-admin revoke-key <key id>`, where the key id is the 12 characters
after `cpk_`. A database from before this change keeps its rows, which now belong to a user
named `legacy`. Issue a key for `legacy` to reach them.

## A walkthrough with the LaLonde fixture

```bash
curl -H "Authorization: Bearer $CP_KEY" -F file=@tests/fixtures/real_data/lalonde.csv http://localhost:8000/v1/datasets
# => {"id": "...", "filename": "lalonde.csv", "sha256": "...", "n_bytes": ..., "columns": [...]}

curl -X POST http://localhost:8000/v1/analyses \
  -H "Authorization: Bearer $CP_KEY" \
  -H 'content-type: application/json' \
  -d '{
        "dataset_id": "<id from above>",
        "schema": {
          "unit": "unit_id",
          "treatment": "treat",
          "outcome": "re78",
          "covariates": ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"],
          "outcome_type": "continuous"
        },
        "estimators": ["difference_in_means", "lin_regression"]
      }'
# => {"id": "<analysis id>", "status": "queued"}

curl -H "Authorization: Bearer $CP_KEY" http://localhost:8000/v1/analyses/<analysis id>
# poll until "status" is "succeeded" or "failed"

curl -H "Authorization: Bearer $CP_KEY" http://localhost:8000/v1/analyses/<analysis id>/estimates/lin_regression/report.html
```

## One report per estimate

Each requested estimator gets its own evidence report and its own validity band, never a report
merged across estimators. Validity is scored per estimate — `score_validity` reads that one
estimate's `assumptions` — and mixing several estimates into a single report would put one
method's band next to another method's assumption table, the exact mismatch `Evidence`'s own
docstring warns against. The cross-method view lives beside the reports instead of inside them:
`GET /analyses/{id}/summary` returns the `compare()` verdicts, and `GET /analyses/{id}` the routing
decisions and every estimate's summary.

## Replay

`GET /analyses/{id}/manifest` returns the analysis's `AnalysisManifest`, which names the uploaded
dataset by filename and content hash rather than embedding it. Replay opens that filename relative
to the current directory, so check an analysis like this:

```bash
mkdir replay-check && cd replay-check
curl -s -H "Authorization: Bearer $CP_KEY" http://localhost:8000/v1/analyses/<analysis id>/manifest > manifest.json
# The filename the manifest records -- its "dataset" -> "path":
python3 -c 'import json; print(json.load(open("manifest.json"))["dataset"]["path"])'
cp /path/to/the/original.csv <that filename>
uv run --project /path/to/causalpulse causalpulse replay manifest.json
```

Read the filename from the manifest rather than assuming it: the same bytes uploaded again under
another name are deduplicated, per user, to that user's first upload, whose filename the manifest keeps. A copy
whose bytes differ at all is refused, since the manifest records their sha256.

`replay` prints each estimator's point estimate to ten significant figures. Compare those with
`estimates[].summary.point` from `GET /analyses/{id}`. The verdict `replay` prints on its own
does not make that comparison: it checks against an `estimates.csv` beside the manifest, which
only `causalpulse analyze` writes and the API does not serve, and without one it compares two
replays of the manifest with each other. The exact, bit-for-bit comparison of every point,
standard error, interval and p-value against the API's stored numbers is what
`tests/services/test_api.py`'s replay-parity test automates -- the check that the API is a thin
layer over the core rather than a second implementation of it.

## Testing the stack itself

```bash
make check-services  # postgres + redis only, project causalpulse-test, ports 5434/6381
make smoke           # api and worker built and run, project causalpulse-smoke, ports 5435/6382/8001
make smoke-web       # whole stack + browser, project causalpulse-web-smoke, 5436/6383/8002/3001
make smoke-prod      # production stack + browser over HTTPS, project causalpulse-prod-smoke, 8091/8444
```

Five separate compose projects exist so none can step on another: the demo stack above
(`causalpulse`, ports 8000/5433/6380/3000), `check-services` (`causalpulse-test`, 5434/6381, no
image build — only `postgres` and `redis`), `smoke` (`causalpulse-smoke`, 5435/6382/8001, `api` and
`worker` with what they depend on), `smoke-web` (`causalpulse-web-smoke`, 5436/6383/8002/3001,
the full stack with the dashboard), and `smoke-prod` (`causalpulse-prod-smoke`, the production
compose file, with only its proxy on the host, at 8091 and 8444; see [Production stack](production.md)). `make smoke` names `api` and `worker`, so it neither builds nor
starts `web`. The services tests read `TEST_DATABASE_URL` and `TEST_REDIS_URL`
exclusively — never the app's own `DATABASE_URL`/`REDIS_URL` — precisely so a developer's local
`DATABASE_URL` export can never point a test run at the demo stack's database by accident.

`make smoke` builds the image, brings the stack up, issues a real key inside the container with
`cp-api-admin`, checks that a request without a key gets 401, uploads the LaLonde fixture over real HTTP,
waits for the analysis to finish, and fetches its report — end to end, through the same routes a
user would call, not through an in-process test client.
