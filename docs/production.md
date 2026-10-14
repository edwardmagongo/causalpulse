# Production stack

`docker-compose.prod.yml` is a second compose file, beside the demo one, for putting the stack on
one small server on the open internet. It is built and it is tested on a developer's machine
(`make smoke-prod`, below). The configuration for a server, a deploy pipeline, backups and a
runbook now exists; see [Deployment](deployment.md), which also says whether the server has been created.

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

The demo file (`docker-compose.yml`) is unchanged, and so is every gate that uses it. The
production file runs the same services from the same two images and differs in this:

- **One more service, `proxy`, and it is the only one with a port on the host.** The demo file
  publishes the API, Postgres, Redis and the dashboard; here each is reached over the stack's own
  network and from nowhere else.
- **No secret has a default.** Without `POSTGRES_PASSWORD`, `CP_SESSION_SECRET` or
  `CP_SITE_ADDRESS`, compose stops and names the one that is missing.
- **`CP_ENV=production`** is set on the API, the worker, the migration and the dashboard, here and
  nowhere else. Everything it switches on is off without it.
- **The session cookie is `Secure`** (`CP_COOKIE_SECURE=true`).
- **The five rate limits are set**, each to a default.
- **Every long-running service** restarts unless stopped and has a health check; the dashboard
  waits for the API to be healthy and the proxy for the dashboard. Each service's log is kept to
  three files of 10 MB.
- **The worker has 2 GB**, memory and swap together. A fit that grows past it is killed and the
  worker restarted, and the analysis it held is then treated as any analysis whose worker died
  ([Backend](backend.md)). The
  figure is twice the largest peak measured for one analysis (1,013 MB, `synthetic_control` with
  `bsts` on Proposition 99).
- **Postgres's data and the proxy's certificates are on named volumes.** Redis has none.

## Settings

They come from the environment, which on a server is a file named `.env` beside the compose file.
`.env.example` lists every name with a comment and no value, and says how to generate the two
secrets; `.env` itself is ignored by git.

| Name | What it is |
|---|---|
| `CP_SITE_ADDRESS` | The site's address, as Caddy reads it. A public name gets a public certificate; `localhost` gets one from Caddy's own local authority. Required. |
| `POSTGRES_PASSWORD` | The database's password, placed in `DATABASE_URL` as it is, so letters and digits only. Required. |
| `CP_SESSION_SECRET` | What the dashboard seals its session cookie with, at least 32 characters. Required. |
| `CP_LIMIT_SIGNIN_FAILURES`, `CP_LIMIT_UPLOADS_PER_DAY`, `CP_LIMIT_ANALYSES_PER_DAY`, `CP_LIMIT_ANALYSES_ACTIVE`, `CP_LIMIT_INTERPRETATIONS_PER_DAY` | The rate limits, below. Optional. `CP_LIMIT_INTERPRETATIONS_PER_DAY` is per user and defaults to 5 in the production file. |
| `CP_INTERPRETATION` | `on`, exactly, turns model-written interpretations on; anything else, or nothing, leaves them off. The API and the worker both read it; a worker with it off fails an interpretation it is handed without calling the model. The production compose file feeds both from this one setting. Optional. |
| `ANTHROPIC_API_KEY` | The model's key. Read by the worker only, never by the API. Required when `CP_INTERPRETATION=on`. |
| `CP_HTTP_PORT`, `CP_HTTPS_PORT` | The host ports the proxy listens on. Empty on a server (80 and 443); `make smoke-prod` sets them. |

`DATABASE_URL` and `REDIS_URL` are assembled in the compose file. Redis has no password: it is on
the stack's network only.

## What the services refuse to start with

With `CP_ENV=production`:

- The API, the worker, the migration and `cp-api-admin` exit before they contact anything if
  `DATABASE_URL` is unset or cannot be read as a URL, has no password, has a password this
  repository publishes (the demo stack's, or an empty one), or carries a `password` query
  parameter, whatever that holds: the driver would use it in preference to the one before the
  `@`.
- The dashboard exits if `CP_SESSION_SECRET` is the demo stack's, or if `CP_COOKIE_SECURE` is not
  `true`.

Each prints one line that names the setting and never its value, and exits non-zero. Compose then
shows the service restarting, and nothing is served with it. The lists of published values sit
beside the code that reads each setting, and two tests (`tests/api/test_settings.py`,
`web/src/lib/startup.test.ts`) read the compose files at the root and the Makefile, and the second
`web/` too, and fail if a password or session secret written there is missing from its list.
Without `CP_ENV=production` none of this runs.

One refusal applies in every environment: the worker exits at start-up if `CP_INTERPRETATION=on` and `ANTHROPIC_API_KEY` is not set.

The API also stops at start-up, in any environment, on a rate limit whose setting is not a whole
number greater than zero.

## Rate limits

Counters in Redis, in fixed windows, kept by the API.

| Setting | In the production file | Counts | Per |
|---|---|---|---|
| `CP_LIMIT_SIGNIN_FAILURES` | 10 in 15 minutes | requests answered 401 | client address |
| `CP_LIMIT_UPLOADS_PER_DAY` | 20 | uploads received | user |
| `CP_LIMIT_ANALYSES_PER_DAY` | 30 | analyses accepted | user |
| `CP_LIMIT_ANALYSES_ACTIVE` | 3 | the user's analyses queued or running | user |
| `CP_LIMIT_INTERPRETATIONS_PER_DAY` | 5 | interpretations accepted | user |

- **Uploads are counted when received**, after the key is checked and before any of the body is
  read, and the count is kept whatever becomes of the upload: one refused as too large or as not
  a CSV, and one that repeats an earlier file, each used one. What the allowance bounds is the
  reading, and only a count taken first can refuse without it.
- **Analyses are counted when accepted.** A submission answered 404 or 422, or refused by the
  active limit, is not one of the day's.
- **Interpretations are counted when accepted**, after every refusal check: a request answered 403, 404 or 409 is not one of the day's. A request that is accepted counts whatever becomes of the interpretation, including one that later fails.
- **A day is 24 hours from the first request counted**, not a calendar day, and the 15 minutes
  likewise. A refused request does not push the window's end back.
- **Failed sign-ins** are every 401 on a `/v1` route, the upload route included, counted against
  the client's address. An address that has reached the limit is answered 429 on every `/v1`
  route but the health check, before its key is looked at, for the rest of its 15 minutes: a good
  key from that address is refused too, and a request that succeeds earlier does not reset the
  count. The count is read on the way in and added to after the answer, so requests made at the
  same moment can all be answered before any is counted. That is accepted: a key's secret is 256
  bits.
- **The active limit is a count of rows** read when an analysis is submitted, not a counter, so
  it cannot drift. Before it refuses, the API checks whether any of the user's queued or running
  analyses has lost its job, marks those failed and counts again; it does not judge one
  submitted in the last 30 seconds.
- **A counted submission that then fails keeps its place in the count.** If the insert or the
  enqueue fails after the analysis was counted, the day's count is not given back.
- **Per user, not per key.** Two keys of one user share one allowance.
- **If Redis cannot be reached, or does not answer within half a second, every limit allows the
  request**, and the API logs that once.

Each limit is off unless its setting is present: the API treats an absent or empty setting as no
limit, so the demo stack and the other gates have none. The production file sets all four, and
there an empty value takes the default, so a name left blank in `.env` does not switch a limit
off. To change one, set its number in `.env` and recreate the API
(`docker compose -f docker-compose.prod.yml up -d api`).

A refused request is answered 429 with `{"detail": "<sentence>"}` and a `Retry-After` header in
seconds: the time left in the window, or 60 for the active limit. The sentences:

```
Too many failed sign-ins from this address. Try again in 12 minutes.
Too many uploads today for this account. Try again in 3 hours.
Too many analyses today for this account. Try again in 3 hours.
This account already has 3 analyses waiting or running. Try again when one has finished.
```

The wait is in its largest whole unit, rounded up. The dashboard shows the sentence wherever the
refusal lands (sign-in, the upload, the submission, a page's read, the poll); its upload relay
passes the status and the body on and not the `Retry-After` header.

## The proxy

Stock Caddy (`caddy:2`, configured by `deploy/Caddyfile`). It obtains and renews the site's
certificate by itself, redirects HTTP to HTTPS, and passes everything else to the dashboard. It
adds two headers to every response, relayed downloads included:
`Strict-Transport-Security: max-age=31536000`, without `includeSubDomains` and without `preload`,
and `X-Content-Type-Options: nosniff`. It serves HTTP/1.1 and HTTP/2 and not HTTP/3, has its admin
endpoint off, and writes no access log. Caddy's error log does record the client's address for a request
that fails with a 5xx; it is kept within the container log's size limit.

- **Published on IPv4 only** (`0.0.0.0:80` and `0.0.0.0:443`). On a Linux server Docker's
  userland proxy hands an IPv6 connection to the container from the bridge gateway's IPv4
  address, so every IPv6 visitor would reach Caddy as one client, and the limit on failed
  sign-ins would lock them all out together. **The site's name must therefore have an A record
  and no AAAA record.**
- **It runs as root inside its container**, because the stock image does: it binds 80 and 443 and
  owns its certificate volume. Every capability is dropped but `NET_BIND_SERVICE`, and it cannot
  gain a privilege through a setuid binary. One consequence: root without the capability to read
  past a file's permissions reads `deploy/Caddyfile`, which is mounted from the checkout, only if
  the file is readable by everyone. **On the server `deploy/Caddyfile` must be world-readable**
  (mode 644); checked out under a restrictive umask, Caddy could not read its configuration. A
  deploy does not check the file out: it copies it out of the app image ([Deployment](deployment.md)), and
  `make smoke-deploy` runs the proxy on a file taken out that way. The other long-running processes
  of the two application images are not root: the Python image runs as `app` (uid 10001) and the
  dashboard's as `node`.

## How the client's address reaches the API

Failed sign-ins are counted by address, and the API never sees the browser's connection. Caddy
sets `X-Forwarded-For` to the peer it accepted the connection from. The dashboard takes the last
entry of that header and sends it to the API as `X-CP-Client-Address` on every call it makes,
after removing any header of that name a caller supplied. The API believes the header only with
`CP_ENV=production`, where nothing but the dashboard can reach it, and only if it parses as an
address; otherwise, and in every other environment, it uses the connection's own peer.

This assumes exactly one proxy in front of the dashboard. A second one (a CDN, a load balancer)
would make the last entry that proxy's address: every user would then be one address, and ten
failed sign-ins by anyone would refuse everyone.

## Upload size

Two bounds, and a user meets the first. The API refuses a file over 50 MiB, and a request body
over 50 MiB plus 1 MiB, with a 413 and its own sentence
(`the upload exceeds the 52428800-byte limit`), which the dashboard shows. The proxy refuses a
body over 64 MB (64,000,000 bytes, above the API's 53,477,376). That is the backstop for a body
that never ends, which the API alone would go on receiving for as long as the client kept the
connection open. Caddy's 413 has no body, so if it is ever the one that answers, the dashboard
shows "the API answered 413".

## What `down -v` and a restart cost

- `docker compose -f docker-compose.prod.yml down` stops the stack and keeps its volumes.
  **`down -v` removes them, which destroys the database and the certificates.** The compose
  file makes no backup. On the server of [Deployment](deployment.md), the disk that holds the volumes is
  snapshotted each night.
- **Redis has no volume, on purpose.** What it holds is lost when its container is restarted: the
  queue and the limits' counters. An analysis that was queued is then marked failed, with the
  worker-lost error, the next time it is read, and has to be submitted again; and every limit
  starts counting again from nothing.

## `/study` is public

The benchmark explorer opens without a session, in this stack and in the demo one. It calls no
API. Every other page still redirects a visitor to sign-in.

## Testing it

```bash
make smoke-prod   # production stack + browser over HTTPS; project causalpulse-prod-smoke, ports 8091/8444
```

It needs Docker, `openssl`, `curl`, Node.js 24 and `pnpm`. It generates both secrets in the
recipe's shell and writes neither to a file. It brings the stack up three times, each torn down
with `down -v` whatever happened; a step that fails stops the gate there:

1. **The whole stack**, with `CP_SITE_ADDRESS=localhost:8444` and the limits set low. Then:
   - `deploy/only-the-proxy-publishes.sh` reads the containers as Docker created them and fails
     unless `proxy` is the only service with a port on the host and none is on the host's
     network;
   - the thirteen tests of `workflow.spec.ts` run through the proxy over HTTPS, with Node
     trusting the root certificate copied out of the proxy and the browser told to accept it;
   - the eleven tests of `production.spec.ts` run, whether or not the first run passed: the
     redirect from HTTP, to the same path and query; the two headers and
     `upgrade-insecure-requests` on sign-in and on `/study`, and the two headers on a signed-in
     page and a relayed download; `/study` open to a visitor while another page redirects; the
     cookie `Secure` and `HttpOnly`; the three limits kept per user each driven to its refusal as
     a second user, with the API's sentence on the page; a 12 MB upload from a user over the limit, shown
     that sentence; a 12 MB upload with a revoked key, which lands on sign-in; and failed
     sign-ins until the address is refused, after which a good key is refused too.
2. **The dashboard alone, with the demo stack's session secret.** Its log must say so, Docker
   must have restarted it, and the proxy, started in front of it, must answer with a 5xx.
3. **The API alone, with the demo stack's database password.** Its log must say so and Docker
   must have restarted it.

The two browser runs write their traces to separate directories under `web/test-results/`. A
stack that `production.spec.ts` has run against has spent its limits, so the file cannot be run
twice against one stack. In CI the gate runs on pushes to `main` only ("Continuous integration"
above).

## What is not there yet

- A server that exists, and with it a domain and a real certificate. The configuration for the
  server, the deploy pipeline, the backups and the image registry is written, with a runbook,
  and is not live ([Deployment](deployment.md)).
- IPv6.
- Monitoring beyond the health checks.

The compose file by itself still builds its two images where it runs. On the server an override
(`deploy/compose.server.yml`) names images pulled from a registry in their place.
