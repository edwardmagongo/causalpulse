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
| Every pull request, and every push to `main` | Fast suite | `make test-fast SHARD=I/4 WORKERS="$(nproc)"`, four shards in parallel |
| Every pull request, and every push to `main` | Infrastructure | `make check-infra` |
| Monthly (the 1st, 03:00 UTC), and by hand | Slow tier | `make test-slow SHARD=I/3 SLOW_WORKERS="$(nproc)"`, three shards in parallel |
| By hand only | Live model call | `make check-llm` |
| By hand only, from `main` | Deploy | no `make` target: it puts one commit of `main` on the server ([Deployment](deployment.md)) |

The repository has been public since 2026-11-01, so its Actions minutes are counted against no
allowance and its runners report four CPUs rather than two. The shape above is no longer a budget:
every gate except Production, the slow tier and the two started by hand runs on every pull
request. The five quick jobs from Static to Stack cost about ten runner-minutes together.
The Infrastructure gate has a workflow of its own (`infra.yml`). Until 2026-11-17 it ran, on a
pull request and on a push to `main` alike, only when a changed file was under `infra/`,
`deploy/` or `.github/workflows/`, or was one of a few others the gate reads, to spend fewer of
the minutes a private repository was billed for. It now runs on every pull request and every push
to `main`: a required check a path filter skips never reports, so the pull request waiting for it
waits for ever, and the filter was what kept this gate from being required. It installs one release of
`shellcheck` and one of `actionlint`, each checked against its SHA-256, so that what the gate
objects to does not change with the runner's image. `make check-infra` itself ends by running the
two test files, which run the deploy script and the server's deploy command against stand-ins;
the fast suite holds them too. Measured on its last two runs before the filter went, the whole
workflow takes about 70 seconds.
A push to `main` also runs Production, which builds the images the Browser job has already
built and runs that job's browser tests a second time before its own, so it is kept off pull
requests; what it costs has not been measured on a runner. A pull request that changes the
production stack is checked with `make smoke-prod` on a developer's machine before it is merged,
by convention, and by that job once it is.
The fast suite cost about 45 minutes as one job on the two-CPU runners a private repository was
given. It now runs as four shards at once, split by test file; measured locally, the largest shard
holds about 19 of the suite's 63 minutes of summed test time. On a hosted runner the four shards
of pull request #8 took 3m0s, 7m43s, 8m56s and 15m26s on 2026-11-17, the first time they were
measured there. Each shard repeats the checkout and `uv sync`, so the four cost a little
more runner time than the one job did. Every pull request runs all four shards, and so does every
push to `main`; until 2026-11-17 a path filter kept the suite off a pull request that touched
nothing it reads, which also kept the shards from being required checks. The 45 minutes
were measured with four test workers on two CPUs, in one job, before the runners reported four. The workflows now ask for one worker
for each CPU the runner reports (`nproc`), for each fast and slow shard. Locally the defaults are unchanged: `auto` for the fast suite and four
for the slow tier.

`main` has been protected since 2026-11-17, which the free plan allows on a public repository.
Ten checks are required -- Static, Services, Web, Browser, Stack, the four fast-suite shards and
Infrastructure -- so GitHub itself refuses to merge a pull request until each has reported
success. Production is not among them, because it runs on no pull request, and nor is any check a
path filter could skip. The protection does not apply to administrators, so a direct push by the
owner is still accepted, and no review is required: a sole maintainer cannot approve their own
pull request, and requiring an approval would leave every pull request unmergeable.
A branch has no run until it has a pull request.

The slow tier is the tests marked `slow` that need no services and no model API. It is split by
test file (`pytest --shard I/N`, a hook in `tests/conftest.py`), each shard keeps its JUnit report
as an artifact, and a scheduled run in which any shard does not succeed opens an issue naming it.
Start it by hand with `gh workflow run slow.yml`. Measured on 2026-10-29, as eight shards with
four workers each (`nproc` reported four): 1h58m of wall clock and 215 runner-minutes, all of the
wall clock set by one file, `tests/causal/test_hierarchical_did_recovery.py`, and within it by one
test, which asks for five module-scoped MCMC fixtures and spent 7,035s building them. Five of the
other seven shards did under seven minutes of work each, so the tier now runs as three shards:
the same wall clock, since no split can finish before that one file, at an expected 180
runner-minutes rather than 215. Cutting the two hours means changing what that file fits -- fewer
seeds, shared fits, or parallelism inside them -- not how the files are dealt into shards. GitHub stops scheduled workflows in a repository with no
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
