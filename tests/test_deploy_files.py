"""What a server deploys from: the files the app image carries, the override
that names the registry's images, and the deploy script. Nothing here needs
Docker: the script is run against a stand-in `docker` that records how it was
called and fails where a test tells it to.

`make smoke-deploy` runs the same script against a real Docker.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
OVERRIDE = ROOT / "deploy" / "compose.server.yml"
SCRIPT = ROOT / "deploy" / "deploy.sh"

# Where each deploy file is in the image, and where it comes from. The layout
# is the repository's, because `docker-compose.prod.yml` mounts
# `./deploy/Caddyfile` relative to itself.
DEPLOY_FILES = {
    "/deploy/docker-compose.prod.yml": "docker-compose.prod.yml",
    "/deploy/deploy/compose.server.yml": "deploy/compose.server.yml",
    "/deploy/deploy/Caddyfile": "deploy/Caddyfile",
    "/deploy/deploy/deploy.sh": "deploy/deploy.sh",
}

REGISTRY = "registry.test:5005/team"
SHA = "a" * 40
PREVIOUS = "b" * 40
OLDER = "c" * 40
MARKER = "not-to-be-printed-0123456789"


# ---- The image ----


def _dockerfile_instructions() -> list[tuple[str, str]]:
    """Each instruction as (keyword, arguments), with continued lines joined."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    joined = re.sub(r"\\\n", " ", text)
    instructions = []
    for line in joined.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        keyword, _, arguments = stripped.partition(" ")
        instructions.append((keyword.upper(), arguments.strip()))
    return instructions


def _copied() -> dict[str, tuple[int, str]]:
    """Destination path -> (the instruction's position, source), for every
    file a `COPY` from the build context names one by one."""
    copied = {}
    for position, (keyword, arguments) in enumerate(_dockerfile_instructions()):
        if keyword != "COPY":
            continue
        words = arguments.split()
        if any(word.startswith("--from") for word in words):
            continue
        *sources, destination = [word for word in words if not word.startswith("--")]
        for source in sources:
            target = destination + Path(source).name if destination.endswith("/") else destination
            copied[target] = (position, source)
    return copied


def test_the_image_carries_exactly_the_four_deploy_files_in_the_repository_layout():
    under_deploy = {
        target: source for target, (_, source) in _copied().items() if target.startswith("/deploy")
    }

    assert under_deploy == DEPLOY_FILES


def test_every_deploy_file_exists_and_reaches_the_build_context():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()

    for source in DEPLOY_FILES.values():
        assert (ROOT / source).is_file(), source
        # The ignore file names directories and dotfiles by their first
        # component; a pattern naming one of these would drop it silently.
        assert source not in ignored and source.split("/")[0] not in ignored, source


def test_the_deploy_files_are_copied_as_root_after_the_install_and_before_the_user():
    instructions = _dockerfile_instructions()
    positions = [position for position, _ in _copied().values()]
    deploy_positions = [
        position for target, (position, _) in _copied().items() if target.startswith("/deploy")
    ]
    last_sync = max(
        index
        for index, (keyword, arguments) in enumerate(instructions)
        if keyword == "RUN" and "uv sync" in arguments
    )
    first_user = min(index for index, (keyword, _) in enumerate(instructions) if keyword == "USER")

    assert positions
    # After the install, so a change to a deploy file rebuilds neither layer
    # of dependencies; before `USER`, so the files are root's.
    assert all(last_sync < position < first_user for position in deploy_positions)
    for position in deploy_positions:
        assert "--chown" not in instructions[position][1]


# ---- The override ----


def _override() -> dict[str, object]:
    loaded = yaml.safe_load(OVERRIDE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_the_override_sets_the_image_and_nothing_else_for_the_four_built_services():
    services = _override()["services"]

    assert isinstance(services, dict)
    assert sorted(services) == ["api", "migrate", "web", "worker"]
    for name, service in services.items():
        assert list(service) == ["image"], name


def test_the_override_names_the_app_image_three_times_and_the_web_image_once():
    services = _override()["services"]
    assert isinstance(services, dict)

    def image(name: str) -> str:
        return re.sub(r"\$\{(\w+):\?[^}]*\}", r"<\1>", services[name]["image"])

    app = "<CP_IMAGE_REGISTRY>/causalpulse-app:<CP_IMAGE_TAG>"
    assert [image(name) for name in ("migrate", "api", "worker")] == [app] * 3
    assert image("web") == "<CP_IMAGE_REGISTRY>/causalpulse-web:<CP_IMAGE_TAG>"


def test_the_override_reads_two_variables_and_requires_both():
    text = "\n".join(
        line
        for line in OVERRIDE.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    interpolations = re.findall(r"\$\{[^}]*\}|\$\w+", text)

    assert interpolations
    for interpolation in interpolations:
        # `${NAME:?message}`: compose stops, with the message, when it is
        # unset or empty.
        assert re.fullmatch(r"\$\{(CP_IMAGE_REGISTRY|CP_IMAGE_TAG):\?[^}]+\}", interpolation), (
            interpolation
        )
    assert text.count("$") == len(interpolations)
    assert {"CP_IMAGE_REGISTRY", "CP_IMAGE_TAG"} == set(re.findall(r"\$\{(\w+)", text))


def test_the_override_declares_nothing_a_server_must_not_change():
    def keys(node: object) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {key for value in node.values() for key in keys(value)}
        if isinstance(node, list):
            return {key for value in node for key in keys(value)}
        return set()

    assert not keys(_override()) & {"build", "ports", "environment", "volumes", "name"}


# ---- The script, read ----


def _script_commands() -> list[str]:
    """The script's lines that are not comments, with continued lines joined."""
    joined = re.sub(r"\\\n", " ", SCRIPT.read_text(encoding="utf-8"))
    return [
        line.strip()
        for line in joined.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_the_script_is_posix_sh_that_parses():
    assert SCRIPT.read_text(encoding="utf-8").startswith("#!/bin/sh\n")
    subprocess.run(["sh", "-n", str(SCRIPT)], check=True)


def test_the_script_stops_at_the_first_failure_and_at_an_unset_variable():
    assert _script_commands()[0] == "set -eu"


def test_the_script_knows_nothing_about_aws_or_a_registry_login():
    text = "\n".join(_script_commands())

    assert not re.search(r"\baws\b", text, re.IGNORECASE)
    assert not re.search(r"docker\s+login", text)
    assert "ecr" not in text.lower()


def test_the_script_never_builds():
    commands = _script_commands()
    ups = [line for line in commands if re.search(r"\bcompose up\b", line)]
    runs = [line for line in commands if re.search(r"\bcompose run\b", line)]

    assert ups and runs
    for line in ups:
        assert "--no-build" in line, line
    for line in runs:
        # `compose run` has no `--no-build`. It builds only an image that is
        # missing, and the script has checked it is there; `--pull never`
        # keeps it from fetching another in its place.
        assert "--pull never" in line and "--build" not in line, line
    # The word appears in no other way: not as a subcommand, not as `--build`.
    for line in commands:
        assert "build" not in line.replace("--no-build", ""), line


def test_the_script_never_shows_its_commands_or_the_environment_file():
    text = "\n".join(_script_commands())

    assert not re.search(r"set -\w*x", text)
    assert not re.search(r"\bcompose config\b", text)
    assert not re.search(r"\b(cat|head|tail|sed|awk|grep)\b[^\n]*\.env", text)


# ---- The script, run against a stand-in for Docker ----

# Records its arguments, fails when they contain `STUB_FAIL`, and answers the
# few questions the script asks of Docker.
STUB = r"""#!/bin/sh
all="$*"
last=${all##* }
echo "$all" >> "$STUB_LOG"
if [ -n "${STUB_FAIL:-}" ]; then
    case "$all" in *"$STUB_FAIL"*) exit 1 ;; esac
fi
case "$all" in
    *" ps -q --status running postgres") [ -z "${STUB_POSTGRES_RUNNING:-}" ] || echo id-postgres ;;
    *" ps -a -q "*) echo "id-$last" ;;
    "inspect "*)
        if [ "id-${STUB_UNHEALTHY:-none}" = "$last" ]; then echo unhealthy; else echo healthy; fi ;;
    "image ls "*) printf '%s\n' $STUB_TAGS ;;
esac
exit 0
"""


class Deploy:
    """A release directory and a server root under `tmp_path`, and the script
    run from them."""

    def __init__(self, tmp_path: Path) -> None:
        self.release = tmp_path / "release"
        self.root = tmp_path / "root"
        self.log = tmp_path / "docker.log"
        bin_directory = tmp_path / "bin"
        for source in DEPLOY_FILES.values():
            target = self.release / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / source, target)
        self.root.mkdir()
        (self.root / ".env").write_text(f"POSTGRES_PASSWORD={MARKER}\n", encoding="utf-8")
        bin_directory.mkdir()
        stub = bin_directory / "docker"
        stub.write_text(STUB, encoding="utf-8")
        stub.chmod(0o755)
        self.search = f"{bin_directory}{os.pathsep}{os.environ['PATH']}"

    def run(
        self, *arguments: str, cwd: Path | None = None, **settings: str
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            "PATH": self.search,
            "STUB_LOG": str(self.log),
            "STUB_TAGS": f"{SHA} {PREVIOUS} {OLDER} <none>",
            "CP_DEPLOY_ROOT": str(self.root),
            "CP_DEPLOY_HEALTH_DEADLINE": "0",
            **settings,
        }
        return subprocess.run(
            ["sh", "deploy/deploy.sh", *arguments],
            cwd=cwd or self.release,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def calls(self) -> list[str]:
        """What Docker was asked, without compose's leading options."""
        if not self.log.exists():
            return []
        lines = self.log.read_text(encoding="utf-8").splitlines()
        options = r"^compose .*? -f deploy/compose\.server\.yml "
        return [re.sub(options, "compose ", line) for line in lines]

    def recorded(self) -> str | None:
        current = self.root / "current-sha"
        return current.read_text(encoding="utf-8") if current.exists() else None


@pytest.fixture
def deploy(tmp_path: Path) -> Deploy:
    return Deploy(tmp_path)


def _index(calls: list[str], prefix: str) -> int:
    matches = [index for index, call in enumerate(calls) if call.startswith(prefix)]
    assert len(matches) == 1, (prefix, calls)
    return matches[0]


UP = "compose up -d --no-build --remove-orphans"
MIGRATE = "compose run "
START_POSTGRES = "compose up -d --no-build postgres"


def test_a_first_deploy_starts_postgres_migrates_and_only_then_brings_the_stack_up(deploy):
    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 0, result.stderr
    calls = deploy.calls()
    order = [
        _index(calls, "compose pull "),
        _index(calls, START_POSTGRES),
        _index(calls, MIGRATE),
        _index(calls, UP),
    ]
    assert order == sorted(order), calls
    assert deploy.recorded() == SHA + "\n"


def test_the_migration_is_one_container_that_starts_no_other_service(deploy):
    deploy.run(REGISTRY, SHA)

    migrate = deploy.calls()[_index(deploy.calls(), MIGRATE)]
    assert migrate.split()[-1] == "migrate"
    assert {"--rm", "--no-deps", "-T"} <= set(migrate.split())


def test_compose_is_told_the_project_the_environment_file_and_both_files(deploy):
    deploy.run(REGISTRY, SHA, CP_DEPLOY_PROJECT="some-project")

    lines = deploy.log.read_text(encoding="utf-8").splitlines()
    prefix = (
        f"compose -p some-project --env-file {deploy.root}/.env"
        " -f docker-compose.prod.yml -f deploy/compose.server.yml "
    )
    composed = [line for line in lines if line.startswith("compose ")]
    assert composed and all(line.startswith(prefix) for line in composed), composed


def test_the_project_defaults_to_the_production_one(deploy):
    deploy.run(REGISTRY, SHA)

    assert deploy.log.read_text(encoding="utf-8").startswith("compose -p causalpulse-prod ")


def test_the_script_works_from_another_directory(deploy, tmp_path):
    result = subprocess.run(
        ["sh", str(deploy.release / "deploy" / "deploy.sh"), REGISTRY, SHA],
        cwd=tmp_path,
        env={
            "PATH": deploy.search,
            "STUB_LOG": str(deploy.log),
            "STUB_TAGS": "",
            "CP_DEPLOY_ROOT": str(deploy.root),
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert deploy.recorded() == SHA + "\n"


def test_a_running_postgres_is_left_as_it_is(deploy):
    result = deploy.run(REGISTRY, SHA, STUB_POSTGRES_RUNNING="yes")

    assert result.returncode == 0, result.stderr
    assert not [call for call in deploy.calls() if call.startswith(START_POSTGRES)]
    # And nothing is brought up before the migration.
    calls = deploy.calls()
    assert not [call for call in calls[: _index(calls, MIGRATE)] if " up " in f" {call} "]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        [REGISTRY],
        [REGISTRY, SHA, "extra"],
        [REGISTRY, "a" * 39],
        [REGISTRY, "a" * 41],
        [REGISTRY, "A" * 40],
        [REGISTRY, "g" * 40],
        [REGISTRY, "a" * 39 + " "],
        [REGISTRY, "a" * 40 + "\n"],
        [REGISTRY, "latest"],
        [REGISTRY, ""],
        ["", SHA],
        ["registry with space", SHA],
        ["registry;id", SHA],
        ["registry$(id)", SHA],
        ["registry\n", SHA],
    ],
)
def test_anything_but_a_registry_and_a_lowercase_sha_is_refused_before_docker(deploy, arguments):
    result = deploy.run(*arguments)

    assert result.returncode == 2
    assert result.stderr.startswith("deploy: ")
    assert deploy.calls() == []
    assert deploy.recorded() is None


@pytest.mark.parametrize("deadline", ["", "soon", "-1", "1.5"])
def test_a_deadline_that_is_not_a_number_of_seconds_is_refused_before_docker(deploy, deadline):
    result = deploy.run(REGISTRY, SHA, CP_DEPLOY_HEALTH_DEADLINE=deadline)

    # An empty one takes the default, as an unset one does.
    assert result.returncode == (0 if deadline == "" else 2), result.stderr
    assert deadline == "" or deploy.calls() == []


@pytest.mark.parametrize(
    "missing", ["docker-compose.prod.yml", "deploy/compose.server.yml", "deploy/Caddyfile"]
)
def test_a_release_without_one_of_its_files_is_exit_10_before_docker(deploy, missing):
    (deploy.release / missing).unlink()

    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 10
    assert missing in result.stderr
    assert deploy.calls() == []


def test_a_server_without_its_environment_file_is_exit_10_before_docker(deploy):
    (deploy.root / ".env").unlink()

    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 10
    assert deploy.calls() == []


@pytest.mark.parametrize(
    "failing",
    [
        " pull ",
        f"image inspect {REGISTRY}/causalpulse-app",
        f"image inspect {REGISTRY}/causalpulse-web",
    ],
)
def test_an_image_that_cannot_be_had_is_exit_10_and_nothing_is_started(deploy, failing):
    (deploy.root / "current-sha").write_text(PREVIOUS + "\n", encoding="utf-8")

    result = deploy.run(REGISTRY, SHA, STUB_FAIL=failing)

    assert result.returncode == 10
    calls = deploy.calls()
    assert not [call for call in calls if call.startswith(("compose up", "compose run"))], calls
    assert deploy.recorded() == PREVIOUS + "\n"


def test_a_failed_migration_is_exit_20_and_no_service_is_replaced(deploy):
    (deploy.root / "current-sha").write_text(PREVIOUS + "\n", encoding="utf-8")

    result = deploy.run(REGISTRY, SHA, STUB_FAIL=" run ", STUB_POSTGRES_RUNNING="yes")

    assert result.returncode == 20
    assert "migration" in result.stderr
    assert not [call for call in deploy.calls() if call.startswith("compose up")], deploy.calls()
    assert not [call for call in deploy.calls() if call.startswith("image rm")]
    assert deploy.recorded() == PREVIOUS + "\n"


def test_a_postgres_that_does_not_start_is_exit_20_before_the_migration(deploy):
    result = deploy.run(REGISTRY, SHA, STUB_FAIL=START_POSTGRES.removeprefix("compose "))

    assert result.returncode == 20
    assert not [call for call in deploy.calls() if call.startswith((MIGRATE, UP))]
    assert deploy.recorded() is None


def test_a_postgres_that_is_not_healthy_is_exit_20_before_the_migration(deploy):
    result = deploy.run(REGISTRY, SHA, STUB_UNHEALTHY="postgres")

    assert result.returncode == 20
    assert "postgres=unhealthy" in result.stderr
    assert not [call for call in deploy.calls() if call.startswith((MIGRATE, UP))]


def test_a_stack_that_does_not_come_up_is_exit_30_with_its_state_and_logs(deploy):
    (deploy.root / "current-sha").write_text(PREVIOUS + "\n", encoding="utf-8")

    result = deploy.run(REGISTRY, SHA, STUB_FAIL="--remove-orphans")

    assert result.returncode == 30
    calls = deploy.calls()
    assert "compose ps -a" in calls
    assert "compose logs --no-color --tail 40" in calls
    assert deploy.recorded() == PREVIOUS + "\n"
    assert not [call for call in calls if call.startswith("image rm")]


@pytest.mark.parametrize("service", ["redis", "api", "worker", "web", "proxy"])
def test_a_service_that_is_not_healthy_by_the_deadline_is_exit_30(deploy, service):
    (deploy.root / "current-sha").write_text(PREVIOUS + "\n", encoding="utf-8")

    result = deploy.run(REGISTRY, SHA, STUB_UNHEALTHY=service)

    assert result.returncode == 30
    assert f"{service}=unhealthy" in result.stderr
    calls = deploy.calls()
    assert "compose logs --no-color --tail 40" in calls
    # The SHA is recorded only for a stack that is healthy.
    assert deploy.recorded() == PREVIOUS + "\n"
    assert not [call for call in calls if call.startswith("image rm")]


def test_every_long_running_service_is_asked_for_its_health(deploy):
    deploy.run(REGISTRY, SHA)

    asked = [call.split()[-1] for call in deploy.calls() if call.startswith("inspect ")]
    assert {"id-postgres", "id-redis", "id-api", "id-worker", "id-web", "id-proxy"} == set(asked)


def test_tags_other_than_this_sha_and_the_one_before_it_are_removed(deploy):
    (deploy.root / "current-sha").write_text(PREVIOUS + "\n", encoding="utf-8")

    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 0, result.stderr
    calls = deploy.calls()
    assert [call for call in calls if call.startswith("image rm")] == [
        f"image rm {REGISTRY}/causalpulse-app:{OLDER}",
        f"image rm {REGISTRY}/causalpulse-web:{OLDER}",
    ]
    assert calls[-1] == "image prune -f"
    # After the SHA is recorded, which nothing in the clean-up can undo.
    assert deploy.recorded() == SHA + "\n"


def test_with_no_recorded_sha_only_this_one_is_kept(deploy):
    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 0, result.stderr
    removed = [call.rsplit(":", 1)[1] for call in deploy.calls() if call.startswith("image rm")]
    assert removed == [PREVIOUS, OLDER, PREVIOUS, OLDER]


def test_a_recorded_sha_that_is_not_one_is_ignored_and_not_printed(deploy):
    (deploy.root / "current-sha").write_text(f"{MARKER}\n", encoding="utf-8")

    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 0, result.stderr
    assert MARKER not in result.stderr
    assert deploy.recorded() == SHA + "\n"


@pytest.mark.parametrize("failing", ["image ls", "image rm", "image prune"])
def test_a_clean_up_that_fails_is_reported_and_does_not_fail_the_deploy(deploy, failing):
    result = deploy.run(REGISTRY, SHA, STUB_FAIL=failing)

    assert result.returncode == 0
    assert "could not" in result.stderr
    assert deploy.recorded() == SHA + "\n"


@pytest.mark.parametrize("settings", [{}, {"STUB_FAIL": " run "}, {"STUB_UNHEALTHY": "worker"}])
def test_every_message_is_on_stderr_and_begins_with_deploy(deploy, settings):
    result = deploy.run(REGISTRY, SHA, **settings)

    assert result.stdout == ""
    lines = result.stderr.splitlines()
    assert lines and all(line.startswith("deploy: ") for line in lines), lines
    assert MARKER not in result.stderr


def test_no_temporary_file_is_left_beside_the_recorded_sha(deploy):
    deploy.run(REGISTRY, SHA)

    assert sorted(path.name for path in deploy.root.iterdir()) == [".env", "current-sha"]
