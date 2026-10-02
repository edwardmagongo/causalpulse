.PHONY: install fmt lint types imports test test-fast cov check check-services check-llm smoke

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
SMOKE_STACK = CP_POSTGRES_PORT=5435 CP_REDIS_PORT=6382 CP_API_PORT=8001 docker compose -p causalpulse-smoke

smoke:
	$(SMOKE_STACK) up -d --build || { $(SMOKE_STACK) down -v; exit 1; }
	$(SMOKE_STACK) exec -T api cp-api-admin create-user smoke && KEY=$$($(SMOKE_STACK) exec -T api cp-api-admin issue-key smoke) && CP_STACK_KEY=$$KEY CP_STACK_URL=http://localhost:8001 uv run pytest -v tests/services/test_stack.py -m "services and slow"; status=$$?; $(SMOKE_STACK) down -v; exit $$status
