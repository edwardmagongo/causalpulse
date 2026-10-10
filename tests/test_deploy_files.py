"""What a server deploys from: the files the app image carries, the override
that names the registry's images, and the deploy script. Nothing here needs
Docker: the script is run against a stand-in `docker` that records how it was
called and fails where a test tells it to.

`make smoke-deploy` runs the same script against a real Docker.
"""

import hashlib
import json
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
# The script's `settle`: the seconds between the two looks at the stack.
SETTLE = 25
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


def test_every_deploy_file_exists_and_is_not_named_by_the_ignore_file():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()

    for source in DEPLOY_FILES.values():
        assert (ROOT / source).is_file(), source
        # Only a line that is the file's own path or its first component is
        # seen here. The ignore file's patterns are not interpreted: a line
        # such as `deploy/*.sh` or `*.yml` would drop a file from the build
        # context and pass this test. `make smoke-deploy` is what would fail.
        assert source not in ignored and source.split("/")[0] not in ignored, source


def test_the_deploy_files_are_copied_after_the_install_and_given_to_no_other_owner():
    instructions = _dockerfile_instructions()
    deploy_positions = [
        position for target, (position, _) in _copied().items() if target.startswith("/deploy")
    ]
    last_sync = max(
        index
        for index, (keyword, arguments) in enumerate(instructions)
        if keyword == "RUN" and "uv sync" in arguments
    )

    assert deploy_positions
    # After the install, so a change to a deploy file rebuilds neither layer
    # of dependencies.
    assert all(last_sync < position for position in deploy_positions)
    # `COPY` gives a file to root unless it is told otherwise, wherever the
    # instruction stands, and the services' user must not be able to write
    # these.
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
    assert sorted(services) == ["api", "migrate", "proxy", "web", "worker"]
    for name in ("api", "migrate", "web", "worker"):
        assert list(services[name]) == ["image"], name


def test_the_override_gives_the_proxy_one_label_holding_the_caddyfiles_digest_and_no_more():
    services = _override()["services"]
    assert isinstance(services, dict)
    proxy = services["proxy"]

    # The label is what makes Compose replace the proxy when the Caddyfile's
    # content changes: without it a deploy leaves the old configuration in
    # use. Nothing else of the proxy's is the server's to change.
    assert list(proxy) == ["labels"]
    assert isinstance(proxy["labels"], dict)
    assert list(proxy["labels"]) == ["causalpulse.caddyfile-sha256"]
    value = proxy["labels"]["causalpulse.caddyfile-sha256"]
    assert re.fullmatch(r"\$\{CP_CADDYFILE_DIGEST:\?[^}]+\}", value), value


def test_the_override_names_the_app_image_three_times_and_the_web_image_once():
    services = _override()["services"]
    assert isinstance(services, dict)

    def image(name: str) -> str:
        return re.sub(r"\$\{(\w+):\?[^}]*\}", r"<\1>", services[name]["image"])

    app = "<CP_IMAGE_REGISTRY>/causalpulse-app:<CP_IMAGE_TAG>"
    assert [image(name) for name in ("migrate", "api", "worker")] == [app] * 3
    assert image("web") == "<CP_IMAGE_REGISTRY>/causalpulse-web:<CP_IMAGE_TAG>"


def test_the_override_reads_three_variables_and_requires_each():
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
        assert re.fullmatch(
            r"\$\{(CP_IMAGE_REGISTRY|CP_IMAGE_TAG|CP_CADDYFILE_DIGEST):\?[^}]+\}", interpolation
        ), interpolation
    assert text.count("$") == len(interpolations)
    assert {"CP_IMAGE_REGISTRY", "CP_IMAGE_TAG", "CP_CADDYFILE_DIGEST"} == set(
        re.findall(r"\$\{(\w+)", text)
    )
    # The digest is read once, by the proxy's label, and by no image name.
    assert text.count("CP_CADDYFILE_DIGEST") == 1


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


def test_the_script_removes_no_image_it_does_not_name():
    # `docker image prune` removes every untagged image on the machine,
    # whoever made it; on a developer's machine that is not the deploy's to
    # do.
    assert "prune" not in "\n".join(_script_commands())


def test_the_script_never_shows_its_commands_or_the_environment_file():
    text = "\n".join(_script_commands())

    assert not re.search(r"set -\w*x", text)
    assert not re.search(r"\bcompose config\b", text)
    assert not re.search(r"\b(cat|head|tail|sed|awk|grep)\b[^\n]*\.env", text)


# ---- The script, run against a stand-in for Docker ----

# Records its arguments, fails when they contain `STUB_FAIL`, and answers the
# few questions the script asks of Docker. For a service named NAME:
#   STUB_REPORT_NAME  what `inspect` says of its container, as
#                     `<state> <health> <restarts>`; several, separated by
#                     `;`, are given one per ask, and the last from then on
#   STUB_IDS_NAME     the same, for the first part of its container's id
#   STUB_UNHEALTHY    NAME: short for a report of `running unhealthy 0`
#   STUB_ABSENT       NAME: it has no container
# Each call made through compose also records the Caddyfile's digest as the
# script exported it.
STUB = r"""#!/bin/sh
all="$*"
last=${all##* }
echo "$all" >> "$STUB_LOG"
case "$all" in
    "compose "*) echo "${CP_CADDYFILE_DIGEST-unset}" >> "$STUB_LOG.digests" ;;
esac
if [ -n "${STUB_FAIL:-}" ]; then
    case "$all" in *"$STUB_FAIL"*) exit 1 ;; esac
fi

# nth LIST FILE: the item of LIST for this ask, FILE counting the asks.
nth() {
    asked=$(cat "$2" 2>/dev/null || echo 0)
    asked=$((asked + 1))
    echo "$asked" > "$2"
    count=0
    item=
    IFS=';'
    for item in $1; do
        count=$((count + 1))
        [ "$count" -lt "$asked" ] || break
    done
    unset IFS
    echo "$item"
}

case "$all" in
    *" ps -q --status running postgres") [ -z "${STUB_POSTGRES_RUNNING:-}" ] || echo id-postgres ;;
    *" ps -a -q "*)
        if [ "${STUB_ABSENT:-}" != "$last" ]; then
            eval "ids=\${STUB_IDS_$last:-id}"
            echo "$(nth "$ids" "$STUB_LOG.ids.$last")-$last"
        fi ;;
    "inspect "*)
        name=${last##*-}
        eval "reports=\${STUB_REPORT_$name:-running healthy 0}"
        [ "${STUB_UNHEALTHY:-}" != "$name" ] || reports="running unhealthy 0"
        echo "$(nth "$reports" "$STUB_LOG.reports.$name")/$last" ;;
    "image ls "*) printf '%s\n' $STUB_TAGS ;;
esac
exit 0
"""

# The script's waits, recorded and not waited for.
SLEEP_STUB = """#!/bin/sh
echo "sleep $*" >> "$STUB_LOG"
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
        for name, text in (("docker", STUB), ("sleep", SLEEP_STUB)):
            stub = bin_directory / name
            stub.write_text(text, encoding="utf-8")
            stub.chmod(0o755)
        self.bin_directory = bin_directory
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

    def looks(self) -> list[str]:
        """The script's looks at containers and its waits, in order: the
        service each `inspect` was for, and each `sleep`."""
        return [
            call if call.startswith("sleep ") else call.split()[-1].removeprefix("id-")
            for call in self.calls()
            if call.startswith(("inspect ", "sleep "))
        ]

    def digests(self) -> set[str]:
        """The Caddyfile's digest as each compose call was given it."""
        return set(Path(f"{self.log}.digests").read_text(encoding="utf-8").splitlines())

    def recorded(self, name: str = "current-sha") -> str | None:
        record = self.root / name
        return record.read_text(encoding="utf-8") if record.is_file() else None


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
        # Docker would read the first two as options and the third as a path.
        ["-f", SHA],
        ["--help", SHA],
        ["/registry", SHA],
    ],
)
def test_anything_but_a_registry_and_a_lowercase_sha_is_refused_before_docker(deploy, arguments):
    result = deploy.run(*arguments)

    assert result.returncode == 2
    assert result.stderr.startswith("deploy: ")
    assert deploy.calls() == []
    assert deploy.recorded() is None


# The last four are numbers the shell's arithmetic would stop the script on,
# part of the way through a deploy: a leading zero is read as octal, and
# `08` is not octal; the long ones do not fit.
@pytest.mark.parametrize("deadline", ["soon", "-1", "1.5", "08", "00", "0300", "1234567", "9" * 23])
def test_a_deadline_that_is_not_a_number_of_seconds_is_refused_before_docker(deploy, deadline):
    result = deploy.run(REGISTRY, SHA, CP_DEPLOY_HEALTH_DEADLINE=deadline)

    assert result.returncode == 2, result.stderr
    assert result.stderr.startswith("deploy: ")
    assert deploy.calls() == []


# An empty one takes the default, as an unset one does.
@pytest.mark.parametrize("deadline", ["", "0", "7", "300", "999999"])
def test_a_deadline_that_is_a_number_of_seconds_is_taken(deploy, deadline):
    result = deploy.run(REGISTRY, SHA, CP_DEPLOY_HEALTH_DEADLINE=deadline)

    assert result.returncode == 0, result.stderr


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


@pytest.mark.parametrize("name", ["current-sha", "previous-sha", "deploy.env"])
def test_a_record_that_is_there_and_is_not_a_file_is_exit_10_before_docker(deploy, name):
    (deploy.root / name).mkdir()

    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 10
    assert name in result.stderr
    assert deploy.calls() == []
    assert list((deploy.root / name).iterdir()) == []


# ---- The Caddyfile's digest ----


def _digest(deploy: Deploy) -> str:
    return hashlib.sha256((deploy.release / "deploy" / "Caddyfile").read_bytes()).hexdigest()


@pytest.mark.parametrize("added", ["", "# one more line\n"])
def test_every_compose_call_is_given_the_sha256_of_the_releases_caddyfile(deploy, added):
    unchanged = _digest(deploy)
    with (deploy.release / "deploy" / "Caddyfile").open("a", encoding="utf-8") as caddyfile:
        caddyfile.write(added)

    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 0, result.stderr
    # Every call, `up` among them: the override requires the name, and the
    # proxy's label is what it fills.
    assert deploy.digests() == {_digest(deploy)}
    assert (_digest(deploy) == unchanged) == (added == "")


def _only_these_tools(deploy: Deploy, *digesters: str) -> str:
    """A search path holding the stand-ins, what the script needs of the
    system, and of the two digest commands only those named."""
    tools = deploy.bin_directory.parent / "tools"
    tools.mkdir()
    for name in ("sh", "dirname", "cat", "date", "head", "mv", "rm", *digesters):
        found = shutil.which(name)
        if found is None:
            pytest.skip(f"this machine has no {name}")
        (tools / name).symlink_to(found)
    return f"{deploy.bin_directory}{os.pathsep}{tools}"


@pytest.mark.parametrize("digester", ["sha256sum", "shasum"])
def test_either_digest_command_alone_gives_the_same_digest(deploy, digester):
    result = deploy.run(REGISTRY, SHA, PATH=_only_these_tools(deploy, digester))

    assert result.returncode == 0, result.stderr
    assert deploy.digests() == {_digest(deploy)}


def test_with_neither_digest_command_it_is_exit_10_before_docker(deploy):
    result = deploy.run(REGISTRY, SHA, PATH=_only_these_tools(deploy))

    assert result.returncode == 10
    lines = result.stderr.splitlines()
    assert lines and all(line.startswith("deploy: ") for line in lines), lines
    assert "Caddyfile" in result.stderr
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
    # What a person needs after it: the containers' state and their logs.
    assert "compose ps -a" in deploy.calls()
    assert "compose logs --no-color --tail 40" in deploy.calls()
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


# ---- The wait for health ----

SERVICES = ["postgres", "redis", "api", "worker", "web", "proxy"]
SETTLING = f"sleep {SETTLE}"


def test_the_stack_is_looked_at_twice_with_the_settling_time_between(deploy):
    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 0, result.stderr
    # One look at Postgres before the migration, with no wait after it; then
    # every long-running service, the settling time, and every one again.
    assert deploy.looks() == ["postgres", *SERVICES, SETTLING, *SERVICES]
    calls = deploy.calls()
    assert not [call for call in calls[: _index(calls, UP)] if call.startswith("sleep")]


def test_the_settling_time_covers_the_start_period_of_the_health_checks():
    # A container first reads healthy `interval` after it starts, so the
    # second look proves a life of `interval` + `settle`: the half minute the
    # compose file allows a service to start in.
    compose = yaml.safe_load((ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8"))
    check = compose["x-check"]
    assert (check["interval"], check["start_period"]) == ("5s", "30s")
    assert f"settle={SETTLE}" in _script_commands()
    assert 5 + SETTLE >= 30


def test_docker_is_asked_for_the_state_the_health_the_restarts_and_the_id(deploy):
    deploy.run(REGISTRY, SHA)

    asked = {call.rsplit(" ", 1)[0] for call in deploy.calls() if call.startswith("inspect ")}
    # The stand-in does not interpret this, so it is held to the letter: a
    # container with no health check must not read as healthy.
    assert asked == {
        "inspect -f {{.State.Status}}"
        " {{if .State.Health}}{{.State.Health.Status}}{{else}}unchecked{{end}}"
        " {{.RestartCount}}/{{.Id}}"
    }


def test_a_service_restarted_between_the_two_looks_is_not_well(deploy):
    (deploy.root / "current-sha").write_text(PREVIOUS + "\n", encoding="utf-8")

    # Healthy at both looks, as a worker is that dies a few seconds after
    # each start: only the count of restarts tells.
    result = deploy.run(
        REGISTRY, SHA, STUB_REPORT_worker="running healthy 0;running healthy 1"
    )

    assert result.returncode == 30
    assert "worker=restarted-or-replaced" in result.stderr
    assert deploy.looks() == ["postgres", *SERVICES, SETTLING, *SERVICES]
    assert deploy.recorded() == PREVIOUS + "\n"
    assert not [call for call in deploy.calls() if call.startswith("image rm")]


def test_a_service_replaced_between_the_two_looks_is_not_well(deploy):
    result = deploy.run(REGISTRY, SHA, STUB_IDS_proxy="id;other")

    assert result.returncode == 30
    assert "proxy=restarted-or-replaced" in result.stderr
    assert deploy.recorded() is None


def test_a_restart_before_the_deadline_starts_the_settling_time_again(deploy):
    result = deploy.run(
        REGISTRY,
        SHA,
        STUB_REPORT_worker="running healthy 0;running healthy 1",
        CP_DEPLOY_HEALTH_DEADLINE="600",
    )

    # The third look finds what the second did, and that pair is the one
    # that counts.
    assert result.returncode == 0, result.stderr
    assert deploy.looks() == ["postgres", *SERVICES, SETTLING, *SERVICES, SETTLING, *SERVICES]
    assert deploy.recorded() == SHA + "\n"


@pytest.mark.parametrize("state", ["restarting", "exited", "paused", "created"])
def test_a_container_that_is_not_running_is_not_well_whatever_its_health(deploy, state):
    result = deploy.run(REGISTRY, SHA, STUB_REPORT_worker=f"{state} healthy 0")

    assert result.returncode == 30
    assert f"worker={state}" in result.stderr
    assert deploy.recorded() is None


def test_a_container_that_is_not_running_at_the_second_look_is_not_well(deploy):
    result = deploy.run(
        REGISTRY, SHA, STUB_REPORT_worker="running healthy 0;restarting healthy 0"
    )

    assert result.returncode == 30
    assert "worker=restarting" in result.stderr
    assert deploy.recorded() is None


def test_a_service_that_is_healthy_a_few_looks_on_is_waited_for(deploy):
    result = deploy.run(
        REGISTRY,
        SHA,
        STUB_REPORT_worker="running starting 0;running starting 0;running healthy 0",
        CP_DEPLOY_HEALTH_DEADLINE="600",
    )

    assert result.returncode == 0, result.stderr
    assert deploy.looks() == [
        "postgres",
        *SERVICES,
        "sleep 2",
        *SERVICES,
        "sleep 2",
        *SERVICES,
        SETTLING,
        *SERVICES,
    ]
    assert deploy.recorded() == SHA + "\n"


def test_a_container_with_no_health_check_is_not_well(deploy):
    result = deploy.run(REGISTRY, SHA, STUB_REPORT_worker="running unchecked 0")

    assert result.returncode == 30
    assert "worker=unchecked" in result.stderr
    assert deploy.recorded() is None


def test_a_service_with_no_container_is_not_well(deploy):
    result = deploy.run(REGISTRY, SHA, STUB_ABSENT="worker")

    assert result.returncode == 30
    assert "worker=absent" in result.stderr
    assert deploy.recorded() is None


def test_a_container_docker_does_not_answer_for_is_not_well(deploy):
    result = deploy.run(REGISTRY, SHA, STUB_FAIL="id-worker")

    assert result.returncode == 30
    assert "worker=unknown" in result.stderr
    assert deploy.recorded() is None


# ---- The records and the clean-up ----


def test_a_new_sha_is_recorded_and_the_one_before_it_becomes_the_previous(deploy):
    (deploy.root / "current-sha").write_text(PREVIOUS + "\n", encoding="utf-8")
    (deploy.root / "previous-sha").write_text(OLDER + "\n", encoding="utf-8")

    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 0, result.stderr
    assert deploy.recorded() == SHA + "\n"
    assert deploy.recorded("previous-sha") == PREVIOUS + "\n"


def test_a_deploy_of_the_sha_already_recorded_keeps_the_previous_one_and_its_images(deploy):
    (deploy.root / "current-sha").write_text(SHA + "\n", encoding="utf-8")
    (deploy.root / "previous-sha").write_text(PREVIOUS + "\n", encoding="utf-8")

    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 0, result.stderr
    assert deploy.recorded() == SHA + "\n"
    assert deploy.recorded("previous-sha") == PREVIOUS + "\n"
    assert [call for call in deploy.calls() if call.startswith("image rm")] == [
        f"image rm {REGISTRY}/causalpulse-app:{OLDER}",
        f"image rm {REGISTRY}/causalpulse-web:{OLDER}",
    ]


def test_a_deploy_of_the_sha_already_recorded_with_no_previous_one_keeps_only_its_own(deploy):
    (deploy.root / "current-sha").write_text(SHA + "\n", encoding="utf-8")

    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 0, result.stderr
    removed = [call.rsplit(":", 1)[1] for call in deploy.calls() if call.startswith("image rm")]
    assert removed == [PREVIOUS, OLDER, PREVIOUS, OLDER]
    assert deploy.recorded("previous-sha") is None


def test_a_previous_sha_that_is_not_one_is_ignored_and_not_printed(deploy):
    (deploy.root / "current-sha").write_text(SHA + "\n", encoding="utf-8")
    (deploy.root / "previous-sha").write_text(f"{MARKER}\n", encoding="utf-8")

    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 0, result.stderr
    assert MARKER not in result.stderr
    assert "previous-sha does not hold a SHA" in result.stderr


@pytest.mark.parametrize("name", ["current-sha", "previous-sha", "deploy.env"])
def test_a_record_that_cannot_be_written_is_exit_40_and_no_image_is_removed(deploy, name):
    (deploy.root / "current-sha").write_text(PREVIOUS + "\n", encoding="utf-8")
    # A directory where the script writes the new record first: the write
    # fails whoever runs it, root included.
    (deploy.root / f"{name}.new").mkdir()

    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 40
    lines = result.stderr.splitlines()
    assert lines and all(line.startswith("deploy: ") for line in lines), lines
    assert f"{name} could not be written" in lines[-1]
    # The stack was well: the failure is the record's alone.
    assert _index(deploy.calls(), UP)
    assert deploy.recorded() == PREVIOUS + "\n"
    assert not [call for call in deploy.calls() if call.startswith("image rm")]


def test_tags_other_than_this_sha_and_the_one_before_it_are_removed(deploy):
    (deploy.root / "current-sha").write_text(PREVIOUS + "\n", encoding="utf-8")

    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 0, result.stderr
    calls = deploy.calls()
    assert [call for call in calls if call.startswith("image rm")] == [
        f"image rm {REGISTRY}/causalpulse-app:{OLDER}",
        f"image rm {REGISTRY}/causalpulse-web:{OLDER}",
    ]
    # Nothing else is removed: no image the script does not name.
    assert not [call for call in calls if "prune" in call or call.startswith("rmi")]
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


@pytest.mark.parametrize("failing", ["image ls", "image rm"])
def test_a_clean_up_that_fails_is_reported_and_does_not_fail_the_deploy(deploy, failing):
    result = deploy.run(REGISTRY, SHA, STUB_FAIL=failing)

    assert result.returncode == 0
    assert "could not" in result.stderr
    assert deploy.recorded() == SHA + "\n"


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"STUB_FAIL": " run "},
        {"STUB_UNHEALTHY": "worker"},
        {"STUB_REPORT_worker": "running healthy 0;running healthy 1"},
        {"STUB_FAIL": "image rm"},
    ],
)
def test_every_message_is_on_stderr_and_begins_with_deploy(deploy, settings):
    result = deploy.run(REGISTRY, SHA, **settings)

    assert result.stdout == ""
    lines = result.stderr.splitlines()
    assert lines and all(line.startswith("deploy: ") for line in lines), lines
    assert MARKER not in result.stderr


def test_no_temporary_file_is_left_beside_the_records(deploy):
    deploy.run(REGISTRY, SHA)

    assert sorted(path.name for path in deploy.root.iterdir()) == [
        ".env",
        "current-sha",
        "deploy.env",
    ]

    deploy.run(REGISTRY, PREVIOUS)

    assert sorted(path.name for path in deploy.root.iterdir()) == [
        ".env",
        "current-sha",
        "deploy.env",
        "previous-sha",
    ]


# ---- The settings of the commit that is running, for a person ----


def _deploy_env(deploy: Deploy) -> tuple[list[str], dict[str, str]]:
    """`deploy.env` as (its comment lines, its settings in order)."""
    lines = (deploy.root / "deploy.env").read_text(encoding="utf-8").splitlines()
    comments = [line for line in lines if line.startswith("#")]
    settings = dict(line.split("=", 1) for line in lines if not line.startswith("#"))
    assert len(comments) + len(settings) == len(lines)
    return comments, settings


def test_a_deploy_that_ends_well_writes_the_three_settings_compose_was_given(deploy):
    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 0, result.stderr
    comments, settings = _deploy_env(deploy)
    # Exactly the three the override reads, as every compose call had them.
    assert settings == {
        "CP_IMAGE_REGISTRY": REGISTRY,
        "CP_IMAGE_TAG": SHA,
        "CP_CADDYFILE_DIGEST": _digest(deploy),
    }
    assert deploy.digests() == {settings["CP_CADDYFILE_DIGEST"]}
    assert "This file holds no secret." in " ".join(comments)
    # And it holds none: nothing of the environment file is in it.
    assert MARKER not in (deploy.root / "deploy.env").read_text(encoding="utf-8")


def test_a_deploy_of_the_sha_already_recorded_writes_the_settings_again(deploy):
    (deploy.root / "current-sha").write_text(SHA + "\n", encoding="utf-8")
    (deploy.root / "deploy.env").write_text("CP_IMAGE_TAG=stale\n", encoding="utf-8")

    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 0, result.stderr
    assert _deploy_env(deploy)[1]["CP_IMAGE_TAG"] == SHA


@pytest.mark.parametrize(
    "settings",
    [
        {"STUB_FAIL": " pull "},
        {"STUB_FAIL": " run "},
        {"STUB_FAIL": "--remove-orphans"},
        {"STUB_UNHEALTHY": "api"},
    ],
)
def test_a_deploy_that_does_not_end_well_leaves_the_settings_of_the_one_before(deploy, settings):
    assert deploy.run(REGISTRY, PREVIOUS).returncode == 0
    before = (deploy.root / "deploy.env").read_bytes()

    result = deploy.run(REGISTRY, SHA, **settings)

    assert result.returncode in (10, 20, 30)
    assert (deploy.root / "deploy.env").read_bytes() == before
    assert _deploy_env(deploy)[1]["CP_IMAGE_TAG"] == PREVIOUS


def test_the_settings_are_written_before_the_sha_is_recorded(deploy):
    # `current-sha` is written last, so a machine whose record names a commit
    # has that commit's settings beside it.
    (deploy.root / "current-sha").write_text(PREVIOUS + "\n", encoding="utf-8")
    (deploy.root / "current-sha.new").mkdir()

    result = deploy.run(REGISTRY, SHA)

    assert result.returncode == 40
    assert deploy.recorded() == PREVIOUS + "\n"
    assert _deploy_env(deploy)[1]["CP_IMAGE_TAG"] == SHA


def test_the_script_writes_the_settings_file_and_never_reads_it():
    naming = [line for line in _script_commands() if "deploy.env" in line]

    # The look that refuses a directory of that name, the write, and the
    # sentence for a write that failed.
    assert len(naming) == 3
    assert naming[0] == "for name in current-sha previous-sha deploy.env; do"
    assert naming[1].startswith('if ! record deploy.env "')
    assert naming[2].startswith("say ")


# ---- Which workflow may reach AWS ----

WORKFLOWS = ROOT / ".github" / "workflows"


def _workflow(name: str) -> dict[object, object]:
    loaded = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), name
    return loaded


def _triggers(workflow: dict[object, object]) -> object:
    # YAML 1.1 reads the key `on` as the boolean true.
    return workflow[True] if True in workflow else workflow["on"]


def _jobs(workflow: dict[object, object]) -> dict[str, dict[str, object]]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict) and jobs
    assert all(isinstance(job, dict) for job in jobs.values())
    return jobs


def test_only_the_deploy_workflow_asks_for_an_identity_token():
    # The pipeline's role is given to any job on `main` that presents GitHub's
    # identity token (`infra/pipeline.tf`), and a job gets the token only
    # through the `id-token` permission. The role is root on the server, so a
    # second workflow that has the permission is a second way to it.
    #
    # Each file is parsed, and not searched for the word: `permissions:
    # write-all` gives every permission, this one among them, without naming
    # it. So a `permissions` here is a mapping wherever it stands, and the
    # workflow has one of its own: without it the jobs get the repository's
    # default, which is a setting this test cannot read.
    files = sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])
    asking = []
    for file in files:
        workflow = _workflow(file.name)
        assert "permissions" in workflow, f"{file.name} has no permissions of its own"
        places = {"the workflow": workflow["permissions"]}
        for name, job in _jobs(workflow).items():
            if "permissions" in job:
                places[f"the job {name}"] = job["permissions"]
        for place, permissions in places.items():
            assert isinstance(permissions, dict), (
                f"{file.name}: the permissions of {place} are {permissions!r}, not a list of names"
            )
            if "id-token" in permissions:
                asking.append((file.name, place))

    assert len(files) > 1
    assert asking == [("deploy.yml", "the workflow")]


def test_the_deploy_workflow_is_started_by_hand_and_in_no_other_way():
    loaded = _workflow("deploy.yml")
    triggers = _triggers(loaded)

    assert isinstance(triggers, dict)
    assert list(triggers) == ["workflow_dispatch"]
    assert loaded["permissions"] == {"contents": "read", "actions": "read", "id-token": "write"}


# ---- The deploy workflow's own promises ----


def _deploy_job() -> dict[str, object]:
    return _jobs(_workflow("deploy.yml"))["deploy"]


def _deploy_step(key: str, value: str) -> dict[str, object]:
    """The one step of the deploy job whose `key` is, or begins with, `value`."""
    steps = _deploy_job()["steps"]
    assert isinstance(steps, list)
    found = [step for step in steps if str(step.get(key, "")).split("@")[0] == value]
    assert len(found) == 1, value
    return found[0]


def _number(pattern: str, file: Path) -> int:
    """The one number `pattern` finds in the file."""
    found = re.findall(pattern, file.read_text(encoding="utf-8"), re.MULTILINE)
    assert len(found) == 1, (pattern, file.name)
    return int(found[0])


def test_the_deploy_job_ends_before_its_credentials_do():
    # The role is taken before the images are built and is still needed for
    # the last look at the command. A job that could outlive the session
    # would have that look refused, and say nothing of a deploy that ran.
    job = _deploy_job()
    credentials = _deploy_step("uses", "aws-actions/configure-aws-credentials")["with"]
    assert isinstance(credentials, dict)
    asked = credentials["role-duration-seconds"]
    allowed = _number(r"^\s*max_session_duration\s*=\s*(\d+)$", ROOT / "infra" / "pipeline.tf")

    assert isinstance(job["timeout-minutes"], int)
    assert isinstance(asked, int)
    assert job["timeout-minutes"] * 60 < asked <= allowed


def test_that_the_command_was_sent_is_written_down_before_its_result_is_waited_for():
    # A run stopped while it waits must leave "sent" for the summary, so the
    # output is written by the line after the one that sends, in the step the
    # summary reads.
    step = _deploy_step("name", "Deploy on the server")
    script = step["run"]
    assert isinstance(script, str)
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    sending = [index for index, line in enumerate(lines) if "aws ssm send-command" in line]
    assert len(sending) == 1
    end = sending[0]
    while lines[end].endswith("\\"):
        end += 1
    waiting = min(index for index, line in enumerate(lines) if "get-command-invocation" in line)

    assert lines[end + 1] == 'echo "status=Sent; its result was not read" >>"$GITHUB_OUTPUT"'
    assert end + 1 < waiting
    assert step["id"] == "deploy"
    summary = _deploy_step("name", "Summarise")
    read = summary["env"]
    assert isinstance(read, dict)
    assert summary["if"] == "always()"
    assert read["DEPLOY"] == "${{ steps.deploy.outputs.status }}"


def test_the_server_is_given_time_to_finish_and_the_workflow_waits_longer_than_that():
    # The document's limit is the only one on the pulls and the migration.
    # It has to cover the deploy script's two waits for health, each of which
    # may last the deadline, the settling time after the second, and an
    # allowance for the pulls and the migration, taken here as ten minutes.
    # And the workflow must wait longer than the server may take, delivery
    # included, or it reports a deploy as still running that then ends.
    allowance = 600
    limit = _number(r'^\s*timeoutSeconds\s*=\s*"(\d+)"$', ROOT / "infra" / "deploy-command.tf")
    deadline = _number(r"^deadline=\$\{CP_DEPLOY_HEALTH_DEADLINE:-(\d+)\}$", SCRIPT)
    settle = _number(r"^settle=(\d+)$", SCRIPT)
    minutes = _deploy_job()["timeout-minutes"]
    waits = _deploy_step("name", "Deploy on the server")["env"]
    assert isinstance(minutes, int)
    assert isinstance(waits, dict)

    assert settle == SETTLE
    assert limit >= 2 * deadline + settle + allowance
    assert waits["DELIVERY_SECONDS"] + limit <= waits["WAIT_SECONDS"]
    assert waits["WAIT_SECONDS"] < minutes * 60


# ---- The gates a commit must have passed ----

# What `gh run list` is here. It answers with the file a test wrote for the
# workflow asked about, whatever else it was asked: GitHub does the choosing
# by event, branch and commit on its side, and the step checks what came back
# again, which is what the tests of a run for another branch or event try.
GH = r"""#!/bin/sh
echo "gh $*" >>"$STUB_GH/calls"
workflow=
while [ "$#" -gt 0 ]; do
    if [ "$1" = --workflow ]; then workflow=$2; fi
    shift
done
if [ -e "$STUB_GH/$workflow.fails" ]; then
    echo "gh: could not reach GitHub, as the test asked" >&2
    exit 1
fi
if [ -e "$STUB_GH/$workflow" ]; then cat "$STUB_GH/$workflow"; else echo "[]"; fi
"""

GATE_NAMES = {"ci.yml": "CI", "fast.yml": "Fast suite", "infra.yml": "Infrastructure"}


def _run_of(file: str, conclusion: str | None = "success", **changed: str) -> dict[str, object]:
    """One run as `gh run list --json` gives it. A run that has not ended has
    the conclusion "" there."""
    run: dict[str, object] = {
        "workflowName": GATE_NAMES[file],
        "event": "push",
        "headBranch": "main",
        "headSha": SHA,
        "status": "completed",
        "conclusion": conclusion,
    }
    run.update(changed)
    return run


PASSED = {"ci.yml": [_run_of("ci.yml")]}

# What GitHub answers for each workflow, and whether the step goes on.
GATES = {
    "CI passed and the fast suite has no run": (PASSED, True),
    "CI and the fast suite passed": (PASSED | {"fast.yml": [_run_of("fast.yml")]}, True),
    "all three passed": (
        PASSED | {"fast.yml": [_run_of("fast.yml")], "infra.yml": [_run_of("infra.yml")]},
        True,
    ),
    "a second run of the fast suite passed": (
        PASSED | {"fast.yml": [_run_of("fast.yml", "failure"), _run_of("fast.yml")]},
        True,
    ),
    "CI has no run": ({}, False),
    "CI failed": ({"ci.yml": [_run_of("ci.yml", "failure")]}, False),
    "CI is in progress": ({"ci.yml": [_run_of("ci.yml", "", status="in_progress")]}, False),
    "CI has no conclusion": ({"ci.yml": [_run_of("ci.yml", None, status="in_progress")]}, False),
    "CI passed for another branch": ({"ci.yml": [_run_of("ci.yml", headBranch="other")]}, False),
    "CI passed for a pull request": ({"ci.yml": [_run_of("ci.yml", event="pull_request")]}, False),
    "CI passed for another commit": ({"ci.yml": [_run_of("ci.yml", headSha=PREVIOUS)]}, False),
    "another workflow passed under CI's file": (
        {"ci.yml": [_run_of("ci.yml", workflowName="Fast suite")]},
        False,
    ),
    "GitHub cannot be asked about CI": ({"ci.yml": None}, False),
    "GitHub answers something that is not JSON": ({"ci.yml": "<html>"}, False),
    "the fast suite failed": (PASSED | {"fast.yml": [_run_of("fast.yml", "failure")]}, False),
    "the fast suite is in progress": (
        PASSED | {"fast.yml": [_run_of("fast.yml", "", status="in_progress")]},
        False,
    ),
    "the fast suite is queued": (
        PASSED | {"fast.yml": [_run_of("fast.yml", "", status="queued")]},
        False,
    ),
    "the fast suite was cancelled": (
        PASSED | {"fast.yml": [_run_of("fast.yml", "cancelled")]},
        False,
    ),
    "the fast suite was skipped": (PASSED | {"fast.yml": [_run_of("fast.yml", "skipped")]}, False),
    "GitHub cannot be asked about the fast suite": (PASSED | {"fast.yml": None}, False),
    "the infrastructure gate failed": (
        PASSED | {"infra.yml": [_run_of("infra.yml", "failure")]},
        False,
    ),
    "GitHub cannot be asked about the infrastructure gate": (PASSED | {"infra.yml": None}, False),
}


def test_the_gate_step_cannot_be_passed_over_and_comes_before_the_credentials():
    # The cases below run the step's script, not its YAML, so a step that
    # carried `continue-on-error` or an `if` would refuse a commit and the job
    # would go on to ask for the role regardless.
    steps = _deploy_job()["steps"]
    assert isinstance(steps, list)
    gate = _deploy_step("name", "Check the gates passed for the commit")
    credentials = _deploy_step("uses", "aws-actions/configure-aws-credentials")

    assert "continue-on-error" not in gate
    assert "if" not in gate
    assert steps.index(gate) < steps.index(credentials)


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("jq") is None, reason="needs bash and jq"
)
@pytest.mark.parametrize("case", GATES)
def test_a_commit_is_deployed_only_when_the_gates_that_ran_for_it_passed(tmp_path, case):
    # The step's own script, taken out of the workflow and run as the runner
    # runs it (`bash -e`), with a stand-in `gh` first on the search path.
    answers, goes_on = GATES[case]
    step = _deploy_step("name", "Check the gates passed for the commit")
    script = tmp_path / "gates.sh"
    script.write_text(str(step["run"]), encoding="utf-8")
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    gh = bin_directory / "gh"
    gh.write_text(GH, encoding="utf-8")
    gh.chmod(0o755)
    stub = tmp_path / "github"
    stub.mkdir()
    for file, answer in answers.items():
        if answer is None:
            (stub / f"{file}.fails").touch()
        elif isinstance(answer, str):
            (stub / file).write_text(answer, encoding="utf-8")
        else:
            (stub / file).write_text(json.dumps(answer), encoding="utf-8")

    result = subprocess.run(
        ["bash", "-e", str(script)],
        env={
            "PATH": f"{bin_directory}{os.pathsep}{os.environ['PATH']}",
            "GH_TOKEN": "not-a-token",
            "REPOSITORY": "owner/name",
            "SHA": SHA,
            "STUB_GH": str(stub),
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    calls = (stub / "calls").read_text(encoding="utf-8").splitlines()
    # Each workflow is asked for by its file, for a push to `main` with this
    # commit at its tip.
    assert all(f"--event push --branch main --commit {SHA} " in call for call in calls)
    if goes_on:
        assert result.returncode == 0, result.stdout + result.stderr
        assert [call.split("--workflow ")[1].split()[0] for call in calls] == list(GATE_NAMES)
        assert "CI succeeded for this commit on main." in result.stdout
        if "fast.yml" not in answers:
            assert "::warning::The fast suite has no run for this commit" in result.stdout
    else:
        assert result.returncode != 0, result.stdout + result.stderr
        # An answer that cannot be read is refused with `jq`'s own words and
        # no sentence of the step's.
        if not any(isinstance(answer, str) for answer in answers.values()):
            assert "::error::" in result.stdout + result.stderr
        # Nothing after the gate that refused is asked about.
        asked = [call.split("--workflow ")[1].split()[0] for call in calls]
        assert asked == list(GATE_NAMES)[: len(asked)]
