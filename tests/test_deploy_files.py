"""What a server deploys from: the files the app image carries, the override
that names the registry's images, and the deploy script. Nothing here needs
Docker: the script is run against a stand-in `docker` that records how it was
called and fails where a test tells it to.

`make smoke-deploy` runs the same script against a real Docker.
"""

import hashlib
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


@pytest.mark.parametrize("name", ["current-sha", "previous-sha"])
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


@pytest.mark.parametrize("name", ["current-sha", "previous-sha"])
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

    assert sorted(path.name for path in deploy.root.iterdir()) == [".env", "current-sha"]

    deploy.run(REGISTRY, PREVIOUS)

    assert sorted(path.name for path in deploy.root.iterdir()) == [
        ".env",
        "current-sha",
        "previous-sha",
    ]
