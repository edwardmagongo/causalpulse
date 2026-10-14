# Web dashboard

A browser UI over the backend: sign in with an API key, upload a CSV or pick an earlier upload,
declare its schema, choose estimators or let the router choose, and read the estimates, validity
bands, cross-method comparison and each evidence report. It is a Next.js server in `web/`, and it
is the only thing the browser talks to.

```bash
docker compose up --build
docker compose exec api cp-api-admin create-user alice
docker compose exec -T api cp-api-admin issue-key alice   # shown once; paste it at localhost:3000
```

## What it does with the key

The pasted key is checked with `GET /v1/me` and, if the API accepts it, sealed into an HTTP-only
cookie, `cp_session`, valid for seven days, `SameSite=Lax`, and marked `Secure` only when
`CP_COOKIE_SECURE=true`. The server adds the key to each API call. After sign-in, page JavaScript
never holds it (the sign-in field does, until it is submitted), and the API needs no CORS policy,
because the browser never calls it. The dashboard does not decide whether a key is still good: the
API does, on every call. After a key is revoked, the next page that calls the API gets a 401, clears
the cookie, and lands on sign-in. The server exits at start-up without `CP_SESSION_SECRET` (at least
32 characters) and `CP_API_URL`; the compose file sets both, the secret to a local-only value.
With `CP_ENV=production`, which only the production compose file sets, it also exits on that
local-only secret and unless `CP_COOKIE_SECURE` is `true` ([Production stack](production.md)).

## The session gate and the Content-Security-Policy

`web/src/proxy.ts`, the Next.js middleware, runs on every page request. It looks only for the
session cookie: a request without one is redirected to `/sign-in`. Two paths are let through
without a session, each as that exact path: `/sign-in`, and `/study`, the benchmark explorer,
whose page calls no API and reads no session. A sub-path of either is gated like any other page.
The middleware's matcher leaves
out `/api` and `/sign-out`, each as that path and its sub-paths, and the static files, and nothing
else, so `/sign-in` and `/study` are matched and served under the policy. `/api` has to stay out: when the middleware runs on a request,
Next.js buffers its body and truncates it at 10 MB, which would cut an upload short. The relay
handlers under `/api` answer 401 themselves when there is no session.

The header goes by the session's key. Signed in, it shows "Analyses", "Benchmark study", "New
analysis", the user's name and a "Sign out" button. A visitor, who can only be on `/study`, sees
the "Benchmark study" link and a "Sign in" link to `/sign-in`, and nothing else.

The middleware also sets a Content-Security-Policy on every response it answers, the redirect to
sign-in included, built in `web/src/lib/csp.ts`:

```
default-src 'self'; script-src 'self' 'nonce-<per request>' 'strict-dynamic'; style-src 'self' 'unsafe-inline';
img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self';
form-action 'self'; frame-ancestors 'none'
```

Scripts run only with the request's nonce, which the middleware makes and Next.js puts on its own
scripts. Styles allow inline for two reasons: the study matrix and the interval plot set custom
properties in `style` attributes, and an evidence report is shown in a sandboxed `srcdoc` frame,
which inherits the page's policy and carries its own `<style>` element and `data:` images. The
frame is still granted nothing: no scripts, no forms, no same-origin access. In development
`script-src` also has `'unsafe-eval'`, which React's development build needs. Because the nonce is
per request, no page of the dashboard's is prerendered: the root layout waits for the request, so every page is
rendered per request, sign-in and the 404 page included. The policy is enforced, not report-only,
and the browser suite fails on any violation of it. With `CP_ENV=production` the policy ends with
one more directive, `upgrade-insecure-requests`; without it the policy is the one above, because
the demo stack is served over plain HTTP. `Strict-Transport-Security` is not the dashboard's to
set: the production stack's proxy sets it ([Production stack](production.md)).

## When the API does not answer, and when a page fails

Every call the dashboard's server makes to the API has a deadline, one of three constants in
`web/src/lib/api/client.ts`. A page's own reads and the analysis submission have 30 seconds. A
relayed call (the poll for a running analysis, a manifest or a Markdown report) has 60 seconds.
The upload has 600 seconds,
because its body is streamed from the browser and the deadline covers the time the browser takes
to send the file: 50 MB at 1 Mbit/s is about seven minutes. The deadline keeps running while the
response body arrives. A call that outlives it is treated as a refused connection is: the page
shows its "service is unavailable" state, and a relay handler answers 503 with
`the service is unavailable`. A relayed download that stalls after its headers were sent is cut
off at the deadline.

A read the API refuses with 429 is not the service being down, and the page says what the API
said: the refusal's own sentence, in place of the data, with a "Try again" button. The poll for a
running analysis shows the sentence followed by "Still checking." and keeps polling at its usual
lengthening pace. A 429 whose body holds no sentence is shown as "Too many requests. Try again
later." This matters when rate limits are set: an address over the limit on failed sign-ins is
answered 429 on every route until its window ends.

A signed-in page that throws anything else shows an error page inside the header: "Something went
wrong on this page.", "Nothing was changed by this. You can try again.", and a "Try again" button.
A failure of the root layout itself shows the same words in a document of its own. Neither shows
the error's message or stack; the framework's digest, when there is one, is shown as a reference
that a report can quote.

## What it shows, and what it does not write

The analysis page renders what the API returns as data: status, the routing decisions with their
reasons, each estimate with its interval and validity band, one plot panel per comparison group,
and the comparison verdicts. While an analysis is queued or running the page polls for it. Each
evidence report is the API's own HTML, shown in a frame that is granted nothing, with the Markdown
version offered as a download. Bands, verdicts, refusal reasons and errors are the backend's
strings, shown as they arrive; the dashboard writes no causal prose of its own. An interval is
labelled with its own basis, and methods share a plot axis only inside one of the API's comparison
groups, which are made of estimates with the same estimand and scale. A value the API reports as
`null` is shown as a dash.

The schema form checks that the required fields are filled. Every other rule is the API's, and its
refusal is shown verbatim.

## The benchmark explorer

`/study`, linked from the header as "Benchmark study", is the published benchmark report
(`docs/reports/2026-09-26-benchmark-study/`) made explorable. It is open to visitors: it is the
one page besides sign-in that needs no session. It shows the
report's sections in order (overview, findings, methods, CATE, scenario P7, limitations), and in
Methods an estimator-by-scenario matrix of any headline metric: the six metric families, by
scenario or by scenario and size, filtered by scenario shape and estimator family, as a heatmap or
a sortable table. Each value carries its Monte Carlo standard error and, where fewer fits fed it
than were fitted, the count. A pair with no number says why (not eligible, every fit refused, no
fit feeds the metric, the metric undefined), in a hatch, a word or a dash, and is never coloured
on the metric's scale. The view lives in the URL's query string, and every switch is a link, so a
view can be shared and the back button steps through views.

It reads no API. The data is `web/src/data/benchmark-study.json`, the study bundle, committed, and
imported by one server-only module, so the 1.85 MB file never reaches the browser. The page writes
no claims of its own: narratives, limitations, notes and metric definitions are the bundle's
strings, shown as they are. It refuses a bundle whose `schema_version`, scenario shapes or
estimator families it does not know, with a message that says so.

`bench study-report` writes the bundle in the same run as the report, from the same evidence, and
the bundle records the report's SHA-256. The stores it reads, `benchmarks/runs/study-60`,
`cate-50` and `p7-50`, are git-ignored and exist only on the machine that ran them. From the
repository root, with `RUNS` the directory holding them and `OUT` an existing directory:

```bash
REPORT=docs/reports/2026-09-26-benchmark-study
uv run causalpulse bench study-report "$RUNS/study-60" --out "$OUT" --format markdown \
  --narratives "$REPORT/narratives.json" \
  --cate-run "$RUNS/cate-50" --p7-run "$RUNS/p7-50" \
  --limitation "<the report's first Limitations bullet, verbatim>" \
  --limitation "<the report's second Limitations bullet, verbatim>" \
  --bundle web/src/data/benchmark-study.json
```

The two `--limitation` texts are the published report's own two Limitations bullets, without the
leading `- `; read them from `$REPORT/benchmark-report.md` instead of retyping them. Point `OUT` at
an empty scratch directory to check that the current code reproduces the committed report byte for
byte, or at `$REPORT` to replace it. Report and bundle must be regenerated together:
`tests/report/test_bundle_committed.py` fails when one is regenerated without the other. The
bundle is written only by this command and is never edited by hand. A bundle whose shape the page
does not expect fails `make check-web`, which type-checks the import. The Python test
`tests/report/test_bundle_committed.py` also ties the bundle's copies of the code's own constants
(the metric definitions, each family's metrics, the levels and each estimator's family) to the code.
Neither check can show that the rest of the bundle is what the current code would write; only
running the command shows that.

## Its API types cannot drift

`web/openapi.json` is the API's OpenAPI document, committed. `make openapi` rewrites it. A Python
test fails when a route or response model changes and the file does not, and `make check-web`
fails when the file changes and the generated TypeScript types do not.

## Testing it

```bash
make check-web   # lint, generated types current, type-check, unit tests; no services
make smoke-web   # whole stack + browser; project causalpulse-web-smoke, ports 5436/6383/8002/3001
```

`make check-web` and `make smoke-web` both run `pnpm` on the host, so both need Node.js 24 and
`pnpm`. `make smoke-web` builds both images, issues two keys of one user inside the container,
installs Playwright's Chromium, and runs thirteen browser tests: sign-in and a signed-in page each
served under the policy, with different nonces; a path that is no page, which redirects to sign-in
without a session, while `/study` alone opens for a visitor, under the policy, with a "Sign in"
link and no "Sign out", and a sub-path of it redirects, and which is a 404 under the policy with a
session; a wrong key and then the issued one;
LaLonde from upload through two estimators to an opened evidence report; Proposition 99 with
`synthetic_control` through to the end of its analysis; a 12 MB upload; an upload one byte over the
API's 50 MB limit and one of exactly that size; an upload posted from another origin, which is
refused with a 403; the header at 320, 360 and 390 px wide, on one line with every control inside
it and no sideways scroll; the benchmark study opened from the header, with a figure from the
published report found at its estimator and scenario at both levels; the header's "Sign out"
button; and a revoked key. After every test the suite asserts that the browser's console logged no
violation of the policy. It tears the stack down afterwards.

`make smoke-prod` runs the same thirteen tests through the production stack's proxy over HTTPS,
and then eleven more in `web/e2e/production.spec.ts` that only that stack can pass ("Production
stack" below). `make smoke-web` leaves that file out.
