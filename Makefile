.PHONY: install fmt lint types imports test test-fast test-slow cov check check-services check-llm check-web smoke smoke-web openapi

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

test:
	uv run pytest -v -n auto

test-fast:
	uv run pytest -v -m "not slow" -n auto

# The slow tier that needs no services and no model API, as CI runs it: one
# shard of the slow test files per job (`make test-slow SHARD=3/8`), each
# writing its own JUnit report, `slow-3-of-8.xml`. With SHARD unset it runs
# every slow test and writes `slow-all.xml`.
SHARD ?=

test-slow:
	uv run pytest -v -m "slow and not services and not live" -n 4 $(if $(SHARD),--shard $(SHARD) --junitxml=slow-$(subst /,-of-,$(SHARD)).xml,--junitxml=slow-all.xml)

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

# One real interpretation: one or two model calls. Needs ANTHROPIC_API_KEY (or an `ant auth login` profile)
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
	$(WEB_SMOKE_STACK) exec -T api cp-api-admin create-user smoke && KEY=$$($(WEB_SMOKE_STACK) exec -T api cp-api-admin issue-key smoke) && REVOCABLE=$$($(WEB_SMOKE_STACK) exec -T api cp-api-admin issue-key smoke) && (cd web && pnpm install --frozen-lockfile && pnpm exec playwright install $(PLAYWRIGHT_INSTALL) chromium && CP_WEB_URL=http://localhost:3001 CP_E2E_API_HEALTH=http://localhost:8002/v1/healthz CP_E2E_KEY=$$KEY CP_E2E_REVOCABLE_KEY=$$REVOCABLE CP_E2E_REVOKE='$(WEB_SMOKE_STACK) exec -T api cp-api-admin revoke-key' pnpm run e2e); status=$$?; $(WEB_SMOKE_STACK) down -v; exit $$status
