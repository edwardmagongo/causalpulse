# Continuous integration

GitHub Actions runs the same `make` targets a developer runs. A workflow sets a job up and calls
one target; nothing a gate does is written a second time in YAML, so a gate means the same thing
on a laptop and on a runner. One thing is not of that shape: the Deploy workflow is not a gate
at all ([Deployment](deployment.md)).

| When | Job | Runs |
|---|---|---|
| Every pull request, and every push to `main` | Static | `make lint types imports` |
| | Services | `make check-services` |
| | Web | `make check-web` |
| | Browser | `make smoke-web` |
| | Stack | `make smoke` |
| Every push to `main`, and no pull request | Production | `make smoke-prod` |
| A pull request that changes what it tests, and every push to `main` | Fast suite | `make test-fast SHARD=I/4 WORKERS="$(nproc)"`, four shards in parallel |
| A pull request or a push to `main` that changes what it checks | Infrastructure | `make check-infra` |
| Monthly (the 1st, 03:00 UTC), and by hand | Slow tier | `make test-slow SHARD=I/8 SLOW_WORKERS="$(nproc)"`, eight shards in parallel |
| By hand only | Live model call | `make check-llm` |
| By hand only, from `main` | Deploy | no `make` target: it puts one commit of `main` on the server ([Deployment](deployment.md)) |

The repository is private, so its Actions minutes are counted against a monthly allowance, and
the shape above is what fits it. The five quick jobs from Static to Stack, which are all a pull
request runs unless it touches what the fast suite or the infrastructure gate reads, cost about
ten billed minutes together (each job is rounded up to a whole minute).
The Infrastructure gate has a workflow of its own (`infra.yml`), because a workflow can be
limited to the paths a change touches and a job cannot. It runs, on a pull request and on a
push to `main` alike, only when a changed file is under `infra/`, `deploy/` or
`.github/workflows/`, or is one of the two test files `tests/test_deploy_*.py`, the `Makefile`,
the `Dockerfile`, `.dockerignore` or `docker-compose.prod.yml`; a change to the Python or the
dashboard alone does not start it and is billed nothing for it. It installs one release of
`shellcheck` and one of `actionlint`, each checked against its SHA-256, so that what the gate
objects to does not change with the runner's image. `make check-infra` itself ends by running the
two test files, which run the deploy script and the server's deploy command against stand-ins;
the fast suite holds them too, but its pull-request paths do not include `deploy/` or `infra/`.
What the workflow costs has not been measured on a runner: it has not run there yet.
A push to `main` also runs Production, which builds the images the Browser job has already
built and runs that job's browser tests a second time before its own, so it is kept off pull
requests; what it costs has not been measured on a runner. A pull request that changes the
production stack is checked with `make smoke-prod` on a developer's machine before it is merged,
by convention, and by that job once it is.
The fast suite cost about 45 minutes as one job on the two-CPU runners a private repository is
given. It now runs as four shards at once, split by test file; measured locally, the largest shard
holds about 19 of the suite's 63 minutes of summed test time, and the shards' time on a runner has
not been measured yet. Each shard repeats the checkout and `uv sync`, so the four cost a little
more runner time than the one job did. On a pull request it runs only when the packages, the
tests, the dependency pins or a committed file the tests check or read has changed: the OpenAPI
document, the study bundle, the published reports, the import contracts, the `Dockerfile`, the
compose files at the root and the Makefile
(`.github/workflows/fast.yml` lists the paths); a pull request that touches only the dashboard or
the documentation does not pay for it. Every push to `main` runs it regardless. The 45 minutes
were measured with four test workers on two CPUs, in one job. The workflows now ask for one worker
for each CPU the runner reports (`nproc`), for each fast and slow shard. Locally the defaults are unchanged: `auto` for the fast suite and four
for the slow tier.

GitHub does not enforce branch protection on a private repository under the free plan. Changes
to `main` go by pull request, merged when its checks are green, by convention and not by a rule
the platform applies: a direct push is accepted. A branch has no run until it has a pull request.

The slow tier is the tests marked `slow` that need no services and no model API. It is split by
test file (`pytest --shard I/N`, a hook in `tests/conftest.py`), each shard keeps its JUnit report
as an artifact, and a scheduled run in which any shard does not succeed opens an issue naming it.
Start it by hand with `gh workflow run slow.yml`. On four-core runners one run took about two
hours and about 170 runner-minutes, with four workers a shard (it has not been run on the smaller
runners of a private repository, or with a worker for each CPU, where it will differ), set by one file, `tests/causal/test_hierarchical_did_recovery.py`; the other seven
shards finish in under 20 minutes. GitHub stops scheduled workflows in a repository with no
activity for 60 days.

The live model call is billed, so it never runs on a schedule or for a pull request. It needs the
repository secret `ANTHROPIC_API_KEY` and is started with `gh workflow run llm-live.yml`.

The interpreter is the one `.python-version` names, 3.13, which is what the suite's thresholds
were measured on. The Docker image is built from `python:3.13-slim`, so the containers the Stack,
Browser and Production jobs test run 3.13 too, and `tests/test_image.py` fails if the image and
`.python-version` come to name different versions. Its processes run as a user that is not root
(`app`, uid 10001), which can read `/app` and cannot write to it; what they write while running
goes under `/tmp`.

Each `pytest-xdist` worker is given one numerical thread (`ONE_THREAD` in the Makefile). Without
it, four workers on a four-core Linux runner each open a full set of BLAS and OpenMP threads: the
fast suite reached 22% in an hour there, and with the cap it finished in about half an hour. macOS
does not show the problem; the likely reason, not tested, is that numpy uses Accelerate there.
That was four workers because the workflow then asked for four; it now asks for as many as the
runner has CPUs, and the cap stays.
