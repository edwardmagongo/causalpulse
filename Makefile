.PHONY: install fmt lint types imports test test-fast test-slow cov check check-services check-llm check-web smoke smoke-web smoke-prod smoke-deploy check-infra openapi

install:
	uv sync

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

types:
	uv run mypy packages

imports:
	uv run lint-imports

# One numerical thread per pytest-xdist worker. Without this each worker opens
# a machine's worth of BLAS, OpenMP and XLA threads, and N workers on N cores
# contend: on a 4-core Linux runner the fast suite reached 22% in an hour and
# stopped moving, with one test taking 31 minutes. macOS hides it (numpy uses
# Accelerate there). The names are the ones `cp_evaluation.execute` caps for
# the benchmark's own workers, for the same reason.
ONE_THREAD = OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

test:
	$(ONE_THREAD) uv run pytest -v -n auto

# WORKERS is the number of pytest-xdist workers. `auto` locally; CI passes
# the number of CPUs its runner has (`nproc` there), which is two on the
# runners a private repository is given.
WORKERS ?= auto

# With SHARD=I/N it runs only shard I of N, split by test file as for
# `test-slow`: CI runs the fast suite as four shards at once. SHARD is shared
# with `test-slow` and read from the environment, so leave it unexported.
test-fast:
	$(ONE_THREAD) uv run pytest -v -m "not slow" -n $(WORKERS) --durations=25 $(if $(SHARD),--shard $(SHARD))

# The slow tier that needs no services and no model API, as CI runs it: one
# shard of the slow test files per job (`make test-slow SHARD=3/8`), each
# writing its own JUnit report, `slow-3-of-8.xml`. With SHARD unset it runs
# every slow test and writes `slow-all.xml`.
#
# SLOW_WORKERS is the number of pytest-xdist workers: four locally, the
# project's cap; CI passes the number of CPUs its runner has, as for WORKERS.
SHARD ?=
SLOW_WORKERS ?= 4

test-slow:
	$(ONE_THREAD) uv run pytest -v -m "slow and not services and not live" -n $(SLOW_WORKERS) --durations=25 $(if $(SHARD),--shard $(SHARD) --junitxml=slow-$(subst /,-of-,$(SHARD)).xml,--junitxml=slow-all.xml)

cov:
	uv run pytest --cov=packages --cov-report=term-missing

check: lint types imports test

SERVICES_ENV = TEST_DATABASE_URL=postgresql+psycopg://causalpulse:causalpulse@localhost:5434/causalpulse TEST_REDIS_URL=redis://localhost:6381/0
TEST_STACK = CP_POSTGRES_PORT=5434 CP_REDIS_PORT=6381 docker compose -p causalpulse-test

# Serial, not -n auto: the tests share one database. Runs against its own
# compose project on 5434/6381, never the demo stack's 5433/6380, so it can
# never wipe a running demo.
check-services:
	$(TEST_STACK) up -d --wait postgres redis || { $(TEST_STACK) down -v; exit 1; }
	$(SERVICES_ENV) uv run pytest -v -m "services and not slow"; status=$$?; $(TEST_STACK) down -v; exit $$status

# One real interpretation (one or two model calls) and one judge call. Needs ANTHROPIC_API_KEY (or an `ant auth login` profile)
# and is billed. -s shows the token usage the test prints.
check-llm:
	CP_LLM_LIVE=1 uv run pytest -v -s -m live

# End-to-end: builds the image, brings up the whole stack (postgres, redis,
# migrate, api, worker), and uploads the LaLonde fixture over real HTTP. Runs
# against its own compose project on 5435/6382/8001, never the demo stack's
# 5433/6380/8000 or check-services' 5434/6381, so it can never wipe either.
# It names `api` and `worker` (their dependencies follow), so it never builds or
# starts the dashboard; `smoke-web` does that.
SMOKE_STACK = CP_POSTGRES_PORT=5435 CP_REDIS_PORT=6382 CP_API_PORT=8001 docker compose -p causalpulse-smoke

smoke:
	$(SMOKE_STACK) up -d --build api worker || { $(SMOKE_STACK) down -v; exit 1; }
	$(SMOKE_STACK) exec -T api cp-api-admin create-user smoke && KEY=$$($(SMOKE_STACK) exec -T api cp-api-admin issue-key smoke) && CP_STACK_KEY=$$KEY CP_STACK_URL=http://localhost:8001 uv run pytest -v tests/services/test_stack.py -m "services and slow"; status=$$?; $(SMOKE_STACK) down -v; exit $$status

# The API's OpenAPI document, which the dashboard's types are generated from.
# `tests/api/test_openapi.py` fails when this file is out of date.
openapi:
	uv run cp-api-openapi > web/openapi.json

# The dashboard's service-free gate: lint, generated types current, types, unit tests.
check-web:
	cd web && pnpm install --frozen-lockfile && pnpm run check

# The dashboard end to end, in a browser, against the whole stack. Its own
# compose project on 5436/6383/8002/3001, so it can never touch the demo,
# check-services or smoke stacks. Two keys of one user: the second is revoked
# by the last test.
#
# PLAYWRIGHT_INSTALL is passed to `playwright install`. Empty locally, where
# the browser's system libraries are already present; CI passes `--with-deps`
# to install them on a bare runner.
PLAYWRIGHT_INSTALL ?=
WEB_SMOKE_STACK = CP_POSTGRES_PORT=5436 CP_REDIS_PORT=6383 CP_API_PORT=8002 CP_WEB_PORT=3001 docker compose -p causalpulse-web-smoke

smoke-web:
	$(WEB_SMOKE_STACK) up -d --build || { $(WEB_SMOKE_STACK) down -v; exit 1; }
	$(WEB_SMOKE_STACK) exec -T api cp-api-admin create-user smoke && KEY=$$($(WEB_SMOKE_STACK) exec -T api cp-api-admin issue-key smoke) && REVOCABLE=$$($(WEB_SMOKE_STACK) exec -T api cp-api-admin issue-key smoke) && (cd web && pnpm install --frozen-lockfile && pnpm exec playwright install $(PLAYWRIGHT_INSTALL) chromium && CP_WEB_URL=http://localhost:3001 CP_E2E_API_HEALTH=http://localhost:8002/v1/healthz CP_E2E_KEY=$$KEY CP_E2E_REVOCABLE_KEY=$$REVOCABLE CP_E2E_REVOKE='$(WEB_SMOKE_STACK) exec -T api cp-api-admin revoke-key' CP_E2E_PLANT='$(WEB_SMOKE_STACK) exec -T api python -' pnpm run e2e); status=$$?; $(WEB_SMOKE_STACK) down -v; exit $$status

# The production stack (`docker-compose.prod.yml`) end to end, in a browser,
# over HTTPS through its proxy. Its own compose project, and 8091 and 8444 on
# the host for the proxy's HTTP and HTTPS: no other port is published, and
# those two are none of the demo stack's or the other gates' (5433 to 5436,
# 6380 to 6383, 8000 to 8002, 3000 and 3001).
#
# It is a recipe here and not a script under `deploy/`, because the tests that
# look for a published secret in what configures a stack read this file
# (`tests/api/test_settings.py`, `web/src/lib/startup.test.ts`) and read no
# script. Both secrets are generated in the recipe's shell and exported from
# it; nothing is written to a file but the proxy's root certificate, which is
# public, to a temporary directory outside the repository.
#
# The limits are set low enough to reach in a few requests, and high enough
# for `workflow.spec.ts`, which runs first as the user `smoke`:
#   CP_LIMIT_UPLOADS_PER_DAY=6    it sends five uploads (LaLonde, Proposition
#                                 99, 12 MB, one byte over 50 MB, exactly 50
#                                 MB; the two cross-origin posts are refused
#                                 by the dashboard and reach no API).
#   CP_LIMIT_ANALYSES_PER_DAY=3   it starts two analyses,
#   CP_LIMIT_ANALYSES_ACTIVE=2    and waits for each before the next.
#   CP_LIMIT_SIGNIN_FAILURES=8    counted by address, and every test comes
#                                 from one: two 401s in `workflow.spec.ts` (a
#                                 wrong key, a revoked one) and one in
#                                 `production.spec.ts` (an upload with a
#                                 revoked key) before its last test, which
#                                 fails sign-ins until the address is refused.
# `production.spec.ts` drives each to its refusal as a second user, `limits`,
# and reads the four numbers from the same variables the stack does.
PROD_SMOKE_PROJECT = causalpulse-prod-smoke
PROD_HTTP_PORT = 8091
PROD_HTTPS_PORT = 8444
PROD_SMOKE_STACK = docker compose -p $(PROD_SMOKE_PROJECT) -f docker-compose.prod.yml
PROD_SMOKE_SETTINGS = CP_SITE_ADDRESS=localhost:$(PROD_HTTPS_PORT) CP_HTTP_PORT=$(PROD_HTTP_PORT) CP_HTTPS_PORT=$(PROD_HTTPS_PORT) \
	CP_LIMIT_SIGNIN_FAILURES=8 CP_LIMIT_UPLOADS_PER_DAY=6 CP_LIMIT_ANALYSES_PER_DAY=3 CP_LIMIT_ANALYSES_ACTIVE=2

# Shell functions each line of the recipe starts with (a line is one shell).
#   retry COMMAND...     run it every two seconds until it succeeds; give up after a minute
#   quietly COMMAND...   run it without what it writes to standard error
#   logged SERVICE TEXT  the service's log holds the text
#   restarted SERVICE    Docker has restarted the service's container at least once
#   unserved URL         the URL is answered with a 5xx: the proxy is up and has nothing behind it
PROD_SMOKE_SH = retry() { n=0; until "$$@"; do n=$$((n + 1)); [ $$n -lt 30 ] || { echo "smoke-prod: gave up waiting for: $$*" >&2; return 1; }; sleep 2; done; }; \
	quietly() { "$$@" 2>/dev/null; }; \
	logged() { $(PROD_SMOKE_STACK) logs "$$1" 2>&1 | grep -q "$$2"; }; \
	restarted() { [ "$$(docker inspect -f '{{.RestartCount}}' "$$($(PROD_SMOKE_STACK) ps -a -q "$$1")" 2>/dev/null)" -gt 0 ] 2>/dev/null; }; \
	unserved() { case "$$(curl -sk -o /dev/null -w '%{http_code}' --max-time 5 "$$1")" in 5??) ;; *) return 1 ;; esac; };

# Three lines, each its own stack, each torn down whatever happened.
#
# 1. The whole stack, after removing whatever an interrupted run left: a
#    database volume made with an earlier run's password would refuse this
#    one's. `up -d` returns once the proxy has started, which waits for the
#    dashboard to be healthy, which waits for the API. Then: no
#    service but the proxy has a port on the host; the proxy's root
#    certificate, for Node (the browser is told to accept it), copied
#    `quietly`: it is not there until Caddy has made it; if `retry` gives up,
#    the copy is run once more without `quietly`, so the failure is shown; two users and
#    two keys each, in shell variables; `workflow.spec.ts`, and then
#    `production.spec.ts` whether or not the first passed, so one run reports
#    both, each with its own directory under `web/test-results/`, because
#    a run empties the directory it writes its traces to. In that order: the
#    second ends with its address locked out of sign-in. The services' last
#    log lines are printed if anything failed.
# 2. The dashboard alone, with the demo stack's session secret: it must say so
#    and be restarted by Docker, and the proxy, started in front of it, must
#    have nothing to serve.
# 3. The API alone, with the demo stack's database password: the same.
smoke-prod:
	$(PROD_SMOKE_SH) export $(PROD_SMOKE_SETTINGS) POSTGRES_PASSWORD=$$(openssl rand -hex 24) CP_SESSION_SECRET=$$(openssl rand -hex 32); \
	ca=$$(mktemp -d); \
	$(PROD_SMOKE_STACK) down -v >/dev/null 2>&1; \
	( $(PROD_SMOKE_STACK) up -d --build \
	&& sh deploy/only-the-proxy-publishes.sh $(PROD_SMOKE_PROJECT) \
	&& { retry quietly $(PROD_SMOKE_STACK) cp proxy:/data/caddy/pki/authorities/local/root.crt "$$ca/root.crt" \
	    || $(PROD_SMOKE_STACK) cp proxy:/data/caddy/pki/authorities/local/root.crt "$$ca/root.crt"; } \
	&& $(PROD_SMOKE_STACK) exec -T api cp-api-admin create-user smoke \
	&& $(PROD_SMOKE_STACK) exec -T api cp-api-admin create-user limits \
	&& KEY=$$($(PROD_SMOKE_STACK) exec -T api cp-api-admin issue-key smoke) \
	&& REVOCABLE=$$($(PROD_SMOKE_STACK) exec -T api cp-api-admin issue-key smoke) \
	&& LIMITS_KEY=$$($(PROD_SMOKE_STACK) exec -T api cp-api-admin issue-key limits) \
	&& LIMITS_REVOCABLE=$$($(PROD_SMOKE_STACK) exec -T api cp-api-admin issue-key limits) \
	&& cd web && pnpm install --frozen-lockfile && pnpm exec playwright install $(PLAYWRIGHT_INSTALL) chromium \
	&& export CP_WEB_URL=https://localhost:$(PROD_HTTPS_PORT) CP_E2E_HTTP_URL=http://localhost:$(PROD_HTTP_PORT) CP_E2E_IGNORE_HTTPS_ERRORS=true NODE_EXTRA_CA_CERTS="$$ca/root.crt" \
		CP_E2E_KEY=$$KEY CP_E2E_REVOCABLE_KEY=$$REVOCABLE CP_E2E_LIMITS_KEY=$$LIMITS_KEY CP_E2E_LIMITS_REVOCABLE_KEY=$$LIMITS_REVOCABLE \
		CP_E2E_REVOKE='$(PROD_SMOKE_STACK) exec -T api cp-api-admin revoke-key' CP_E2E_PLANT='$(PROD_SMOKE_STACK) exec -T api python -' \
	&& { pnpm exec playwright test --output test-results/workflow; workflow=$$?; CP_E2E_PRODUCTION=true pnpm exec playwright test --output test-results/production production.spec.ts; production=$$?; [ $$workflow -eq 0 ] && [ $$production -eq 0 ]; } ); status=$$?; \
	[ $$status -eq 0 ] || $(PROD_SMOKE_STACK) logs --no-color --tail 60; \
	$(PROD_SMOKE_STACK) down -v; rm -rf "$$ca"; exit $$status
	$(PROD_SMOKE_SH) export $(PROD_SMOKE_SETTINGS) POSTGRES_PASSWORD=$$(openssl rand -hex 24) CP_SESSION_SECRET=local-demo-only-session-secret-not-for-deployment; \
	( $(PROD_SMOKE_STACK) up -d --no-deps web \
	&& retry logged web 'published as a demo secret' && retry restarted web \
	&& $(PROD_SMOKE_STACK) up -d --no-deps proxy \
	&& retry unserved https://localhost:$(PROD_HTTPS_PORT)/sign-in ); status=$$?; \
	[ $$status -eq 0 ] || $(PROD_SMOKE_STACK) logs --no-color --tail 60; \
	$(PROD_SMOKE_STACK) down -v; exit $$status
	$(PROD_SMOKE_SH) export $(PROD_SMOKE_SETTINGS) POSTGRES_PASSWORD=causalpulse CP_SESSION_SECRET=$$(openssl rand -hex 32); \
	( $(PROD_SMOKE_STACK) up -d --no-deps api \
	&& retry logged api 'published as a demo value' && retry restarted api ); status=$$?; \
	[ $$status -eq 0 ] || $(PROD_SMOKE_STACK) logs --no-color --tail 60; \
	$(PROD_SMOKE_STACK) down -v; exit $$status

# The deploy, as a server does it: the script of the deploy command in
# `infra/deploy-command.tf`, which takes `deploy/deploy.sh` out of the app
# image and runs it, against a throwaway registry on 127.0.0.1:5005 and its
# own compose project. It proves that a first deploy from nothing comes up
# healthy, that a second tag replaces the first, that the first tag can be
# deployed again without an image being created, that a broken image fails
# and leaves the recorded SHA alone, that a failed migration leaves the
# running containers running, and that it leaves the release directory
# holding the running commit's files. The script says how, what it does not
# prove, and removes what it made.
#
# The proxy is on 8091 and 8444, as in `smoke-prod`: not at the same time.
# CI does not run this: it creates both images, and the minutes are counted.
smoke-deploy:
	sh deploy/smoke-deploy.sh

# The infrastructure, checked without an account: Terraform's formatting, its
# validation, and its tests against a mocked provider (`infra/tests/`), then
# the first-boot script as Terraform renders it, the deploy document's
# script, the deploy scripts and the workflows, and last the two test files
# that run the deploy script and the deploy document's script against
# stand-ins for Docker. Nothing is planned against AWS and nothing is
# created. It needs `terraform`, `shellcheck`, `actionlint` and `uv`.
#
# A test reads the resources it names and no other. `infra/tests/inventory.sh`
# is what fails when a resource is added that `infra/tests/inventory.txt`
# does not list, or when one of the texts it refuses appears (a key pair,
# port 22, IPv6, a policy the tests do not read). It comes after the
# formatting check because it reads the files as that leaves them.
#
# Terraform is run with every source of AWS credentials taken away, so a
# developer who is signed in reaches nothing through this target. `init`
# downloads the provider, which needs the network and no account, and refuses
# to change the lock file.
TERRAFORM_NO_AWS = env -u AWS_PROFILE -u AWS_DEFAULT_PROFILE -u AWS_ACCESS_KEY_ID \
	-u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN -u AWS_ROLE_ARN \
	-u AWS_WEB_IDENTITY_TOKEN_FILE -u AWS_CONTAINER_CREDENTIALS_RELATIVE_URI \
	-u AWS_CONTAINER_CREDENTIALS_FULL_URI AWS_EC2_METADATA_DISABLED=true \
	AWS_SHARED_CREDENTIALS_FILE=/dev/null AWS_CONFIG_FILE=/dev/null terraform

check-infra:
	$(TERRAFORM_NO_AWS) -chdir=infra fmt -check -recursive
	sh infra/tests/inventory.sh
	$(TERRAFORM_NO_AWS) -chdir=infra init -backend=false -input=false -lockfile=readonly
	$(TERRAFORM_NO_AWS) -chdir=infra validate
	$(TERRAFORM_NO_AWS) -chdir=infra test
	$(TERRAFORM_NO_AWS) -chdir=infra/bootstrap init -backend=false -input=false -lockfile=readonly
	$(TERRAFORM_NO_AWS) -chdir=infra/bootstrap validate
	rendered=$$(mktemp -d) && trap 'rm -rf "$$rendered"' EXIT && \
	sh infra/tests/render-first-boot.sh >"$$rendered/first-boot.sh" && \
	bash -n "$$rendered/first-boot.sh" && shellcheck "$$rendered/first-boot.sh" && \
	sh infra/tests/render-deploy-command.sh >"$$rendered/deploy-command.sh" && \
	sh -n "$$rendered/deploy-command.sh" && shellcheck -s sh "$$rendered/deploy-command.sh"
	shellcheck deploy/*.sh infra/tests/*.sh
	actionlint
	env -u DATABASE_URL -u REDIS_URL uv run pytest tests/test_deploy_files.py tests/test_deploy_command.py -q -p no:cacheprovider
