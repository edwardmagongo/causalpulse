"""The deploy command and the first boot, run and not only read.

The deploy document's script (`infra/deploy-command.tf`) is rendered by
`infra/tests/render-deploy-command.sh`, which needs no Terraform, with a
temporary directory in place of `/srv/causalpulse`, and run under `sh` against
stand-ins for `docker`, `aws` and `flock` that come first on the search path.
The stand-in for `docker` hands out a release whose `deploy/deploy.sh` is a
stand-in too: it says what a test tells it to say and ends with the status a
test gives it. So what is tested here is the document's own part: the marker,
the lock, the release directories, the search for the environment file's
values, the logs and the status. `tests/test_deploy_files.py` tests the real
deploy script, and `make smoke-deploy` runs the two together.

Of the first-boot script (`deploy/first-boot.sh.tftpl`) one function is run,
taken out of the template as text, and the branch that formats the data volume
is held as text: nothing here can run that.

Nothing here needs Docker, Terraform or an account.
"""

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RENDER = ROOT / "infra" / "tests" / "render-deploy-command.sh"
FIRST_BOOT = ROOT / "deploy" / "first-boot.sh.tftpl"

SHA = "a" * 40
REGISTRY = "registry.test:5005"
OLD = "the Caddyfile of the commit that is running\n"
NEW = f"the Caddyfile of {REGISTRY}/causalpulse-app:{SHA}\n"

# The shells the script is run under. On the server and on a Ubuntu runner
# `sh` is dash; on a Mac it is bash, and dash is tried as well when it is
# there and is another program.
SHELLS = list(
    {
        os.path.realpath(found): shell
        for shell in ("dash", "sh")
        if (found := shutil.which(shell)) is not None
    }.values()
)

# Records how it was called and fails, saying so on standard error as Docker
# does, when the call's first word is the one in STUB_FAIL. `cp` hands out the
# release: a Caddyfile that names the image the carrier was created from, and
# the stand-in for the deploy script.
DOCKER = r"""#!/bin/sh
echo "docker $*" >>"$STUB_LOG"
if [ "${STUB_FAIL:-}" = "$1" ]; then
    echo "docker: $1 failed, as the test asked" >&2
    exit 1
fi
case $1 in
    login) cat >/dev/null ;;
    create)
        echo "$2" >"$STUB_LOG.image"
        echo carrier
        ;;
    cp)
        mkdir -p "$3/deploy"
        echo "the Caddyfile of $(cat "$STUB_LOG.image")" >"$3/deploy/Caddyfile"
        cp "$STUB_DEPLOY_SH" "$3/deploy/deploy.sh"
        ;;
esac
exit 0
"""

AWS = """#!/bin/sh
echo "aws $*" >>"$STUB_LOG"
echo a-registry-password
"""

# `flock -n 9`: the lock is free unless a test says it is held.
FLOCK = """#!/bin/sh
echo "flock $*" >>"$STUB_LOG"
exit "${STUB_LOCK_HELD:-0}"
"""

# `mv` is the machine's own, and does one thing more when a test asks: just
# after `release` has been renamed to `release.new`, which only the two
# put-backs do, it makes the path again as empty directories. That is what
# Docker does when the proxy starts while its Caddyfile's path is missing.
MV = """#!/bin/sh
/bin/mv "$@" || exit
case "${STUB_PROXY_STARTS:-}:$1:$2" in
    yes:*/release:*/release.new) mkdir -p "$1/deploy/Caddyfile" ;;
esac
exit 0
"""

# What the release's `deploy/deploy.sh` is here. Like the real one it says
# everything on standard error.
DEPLOY_SH = """#!/bin/sh
echo "deploy.sh $* in $(pwd)" >>"$STUB_LOG"
if [ -n "${STUB_SAYS:-}" ]; then
    cat "$STUB_SAYS" >&2
fi
echo "deploy: the stand-in ends with status ${STUB_STATUS:-0}" >&2
exit "${STUB_STATUS:-0}"
"""


class Server:
    """A temporary root, the marker, the stand-ins, and the rendered script."""

    def __init__(self, tmp_path: Path, shell: str) -> None:
        self.shell = shell
        self.root = tmp_path / "srv"
        self.marker = tmp_path / "first-boot-done"
        self.log = tmp_path / "calls.log"
        self.says = tmp_path / "says"
        self.script = tmp_path / "deploy-command.sh"
        bin_directory = tmp_path / "bin"
        bin_directory.mkdir()
        self.root.mkdir()
        self.marker.write_text("2026-01-01T00:00:00Z\n", encoding="utf-8")
        (self.root / ".env").write_text("POSTGRES_PASSWORD=0123456789abcdef\n", encoding="utf-8")
        deploy_sh = tmp_path / "deploy.sh"
        deploy_sh.write_text(DEPLOY_SH, encoding="utf-8")
        self.deploy_sh = deploy_sh
        for name, text in (("docker", DOCKER), ("aws", AWS), ("flock", FLOCK), ("mv", MV)):
            stub = bin_directory / name
            stub.write_text(text, encoding="utf-8")
            stub.chmod(0o755)
        rendered = subprocess.run(
            [
                "sh",
                str(RENDER),
                f"root={self.root}",
                f"marker={self.marker}",
                f"path={bin_directory}{os.pathsep}{os.environ['PATH']}",
                f"registry={REGISTRY}",
                f"sha={SHA}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.script.write_text(rendered.stdout, encoding="utf-8")

    def release(self, name: str = "release", caddyfile: str = OLD) -> Path:
        """A release directory under the root, as an earlier deploy left it."""
        directory = self.root / name / "deploy"
        directory.mkdir(parents=True)
        (directory / "Caddyfile").write_text(caddyfile, encoding="utf-8")
        return directory.parent

    def run(self, status: int = 0, **settings: str) -> subprocess.CompletedProcess[str]:
        environment = {
            # The script sets its own search path; this one is only for the
            # shell that starts it.
            "PATH": os.environ["PATH"],
            "HOME": str(self.root.parent),
            "STUB_LOG": str(self.log),
            "STUB_DEPLOY_SH": str(self.deploy_sh),
            "STUB_STATUS": str(status),
            **settings,
        }
        if self.says.exists():
            environment["STUB_SAYS"] = str(self.says)
        return subprocess.run(
            [self.shell, str(self.script)],
            cwd=self.root.parent,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def calls(self) -> list[str]:
        if not self.log.exists():
            return []
        return self.log.read_text(encoding="utf-8").splitlines()

    def caddyfile(self) -> str | None:
        """The Caddyfile the release directory holds, or nothing."""
        held = self.root / "release" / "deploy" / "Caddyfile"
        return held.read_text(encoding="utf-8") if held.is_file() else None

    def releases(self) -> list[str]:
        """The release directories that are there, by name."""
        return sorted(path.name for path in self.root.iterdir() if path.name.startswith("release"))

    def names(self) -> list[str]:
        return sorted(path.name for path in self.root.iterdir())


@pytest.fixture(params=SHELLS)
def server(tmp_path: Path, request: pytest.FixtureRequest) -> Server:
    return Server(tmp_path, request.param)


# ---- The rendered script ----


def test_the_rendered_script_differs_from_the_documents_in_five_lines_only(server):
    plain = subprocess.run(
        ["sh", str(RENDER)], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    rendered = server.script.read_text(encoding="utf-8").splitlines()

    assert len(plain) == len(rendered)
    changed = [theirs for ours, theirs in zip(rendered, plain, strict=True) if ours != theirs]
    assert [theirs.split("=", 1)[0] for theirs in changed] == [
        "PATH",
        "registry",
        "sha",
        "root",
        "marker",
    ]
    assert f"root='{server.root}'" in rendered


def test_the_script_reads_no_setting_of_its_own_from_the_environment(server):
    # Whatever a caller exports, the root, the marker, the registry and the
    # search path are the ones written into the script.
    server.release()

    result = server.run(
        root="/nonexistent/root",
        marker="/nonexistent/marker",
        registry="elsewhere.test",
        sha="b" * 40,
        CP_DEPLOY_COMMAND_ROOT="/nonexistent/root",
    )

    assert result.returncode == 0, result.stdout
    assert server.caddyfile() == NEW
    assert f"docker pull --quiet {REGISTRY}/causalpulse-app:{SHA}" in server.calls()


def test_the_renderer_refuses_a_value_it_could_not_quote(tmp_path):
    result = subprocess.run(
        ["sh", str(RENDER), "root=/tmp/it's"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 2
    assert result.stdout == ""


# ---- The marker and the lock ----


def test_a_machine_without_the_first_boots_marker_is_exit_76_and_nothing_is_touched(server):
    server.marker.unlink()
    server.release()
    before = server.names()

    result = server.run()

    assert result.returncode == 76
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    assert "first boot of this machine did not finish" in result.stdout
    assert "'The first boot did not finish' in docs/runbook.md" in result.stdout
    assert server.names() == before
    assert server.calls() == []


def test_without_the_marker_it_is_exit_76_even_when_the_root_is_not_there(server):
    server.marker.unlink()
    shutil.rmtree(server.root)

    result = server.run()

    assert result.returncode == 76
    assert not server.root.exists()


def test_a_held_lock_is_exit_75_and_nothing_but_the_lock_file_is_touched(server):
    server.release()

    result = server.run(STUB_LOCK_HELD="1")

    assert result.returncode == 75
    assert result.stdout == "deploy-command: another deploy is running; this one did nothing\n"
    assert server.names() == [".env", "deploy.lock", "release"]
    assert server.calls() == ["flock -n 9"]
    assert server.caddyfile() == OLD


# ---- The status and the release directory ----


@pytest.mark.parametrize("status", [0, 1, 2, 10, 20, 30, 40])
def test_the_commands_status_is_the_deploy_scripts(server, status):
    server.release()

    result = server.run(status)

    assert result.returncode == status
    assert result.stderr == ""
    assert result.stdout.splitlines()[-1].startswith(f"deploy-command: exit status {status};")
    assert f"deploy.sh {REGISTRY} {SHA} in " in "\n".join(server.calls())


@pytest.mark.parametrize("status", [0, 30, 40])
def test_the_new_release_stays_after_a_status_at_which_the_stack_may_be_new(server, status):
    server.release()

    server.run(status)

    assert server.caddyfile() == NEW
    assert server.releases() == ["release"]


@pytest.mark.parametrize("status", [1, 2, 10, 20])
def test_the_release_that_was_there_is_put_back_when_nothing_was_replaced(server, status):
    server.release()

    result = server.run(status)

    assert server.caddyfile() == OLD
    assert server.releases() == ["release"]
    assert "the release directory holds again the files it held before this run" in result.stdout
    # The deploy script did run from the new commit's files.
    ran = [call for call in server.calls() if call.startswith("deploy.sh ")]
    assert len(ran) == 1 and ran[0].endswith("/release")


def test_a_first_deploy_that_fails_keeps_the_only_release_there_is(server):
    result = server.run(20)

    assert result.returncode == 20
    assert server.caddyfile() == NEW
    assert server.releases() == ["release"]


@pytest.mark.parametrize("failing", ["login", "pull", "create", "cp"])
def test_a_failure_before_the_deploy_script_stops_the_run_with_the_release_as_it_was(
    server, failing
):
    server.release()

    result = server.run(STUB_FAIL=failing)

    assert result.returncode == 1
    assert f"docker: {failing} failed, as the test asked" in result.stdout
    assert server.caddyfile() == OLD
    assert server.releases() == ["release"]
    # Nothing after the step that failed was run, the deploy script least of
    # all. The registry's password is fetched in a pipeline into the login, so
    # `aws` and `docker login` run at once and either may write its line last:
    # the fetch is the login's own half, not a step after it.
    calls = server.calls()
    steps = [call for call in calls if not call.startswith("aws ecr get-login-password ")]
    assert steps[-1].startswith(f"docker {failing} ") or (
        failing in ("create", "cp") and steps[-1].startswith("docker rm ")
    )
    assert not [call for call in calls if call.startswith("deploy.sh ")]


def test_a_carrier_whose_copy_failed_is_removed(server):
    server.release()

    server.run(STUB_FAIL="cp")

    assert server.calls()[-1] == "docker rm -f carrier"


# What an earlier run that was cut off can have left, and what the release
# directory must hold once the next run, whose own deploy changes nothing,
# has ended.
LEFT_OVER = [
    # Cut between the two renames: the release is under its other name.
    (["release.old", "release.new"], "release.old"),
    # Cut while the deploy script ran, or before the status was acted on.
    (["release", "release.old"], "release.old"),
    # Cut while putting the old one back.
    (["release", "release.new"], "release"),
    # A first deploy cut during the copy: there was no release.
    (["release.new"], None),
]


@pytest.mark.parametrize(("left", "kept"), LEFT_OVER)
@pytest.mark.parametrize("ending", ["login fails", "status 20"])
def test_whatever_a_cut_off_run_left_the_next_run_ends_with_one_usable_release(
    server, left, kept, ending
):
    for name in left:
        server.release(name, caddyfile=f"the Caddyfile in {name}\n")

    if ending == "login fails":
        result = server.run(STUB_FAIL="login")
        assert result.returncode == 1
        expected = f"the Caddyfile in {kept}\n" if kept else None
    else:
        result = server.run(20)
        assert result.returncode == 20
        # With no release to put back, the first deploy's own files stay.
        expected = f"the Caddyfile in {kept}\n" if kept else NEW

    assert server.caddyfile() == expected
    assert server.releases() == (["release"] if expected else [])


def test_a_proxy_that_starts_during_a_put_back_does_not_keep_the_old_release_out(server):
    # After a status at which the old release is put back: `release` is
    # renamed away, the proxy starts and Docker makes the path, and the old
    # release must still end up at the path and not inside what Docker made.
    server.release()

    result = server.run(20, STUB_PROXY_STARTS="yes")

    assert result.returncode == 20
    assert "holds again the files it held before this run" in result.stdout
    assert server.caddyfile() == OLD
    assert server.releases() == ["release"]
    assert sorted(path.name for path in (server.root / "release").iterdir()) == ["deploy"]


def test_a_proxy_that_starts_while_a_cut_off_run_is_mended_does_not_keep_it_out(server):
    # The same instant in the put-back a run does at its start, when it finds
    # what a run that was cut off left.
    server.release("release", caddyfile="the Caddyfile of the run that was cut off\n")
    server.release("release.old")

    result = server.run(STUB_FAIL="login", STUB_PROXY_STARTS="yes")

    assert result.returncode == 1
    assert server.caddyfile() == OLD
    assert server.releases() == ["release"]
    assert sorted(path.name for path in (server.root / "release").iterdir()) == ["deploy"]


def test_a_release_set_aside_is_never_removed_where_it_stands(server):
    # `release.old` is whole whenever it exists because it only ever leaves by
    # a rename. Were it removed in place, a run cut off during the removal
    # would have the next run put back half a directory.
    script = server.script.read_text(encoding="utf-8")

    # A line may remove one thing and then rename `release.old`: what is
    # looked for is the name among the arguments of an `rm`.
    removed = [
        line for line in script.splitlines() if re.search(r"\brm\b[^;&|]*release\.old", line)
    ]

    assert removed == []


# ---- What is kept and what is shown ----

ENVIRONMENT = (
    "# a comment=with-an-equals-sign-in-it\n"
    "CP_SITE_ADDRESS=causalpulse.example\n"
    "POSTGRES_PASSWORD=0123456789abcdef0123456789abcdef\n"
    "CP_SESSION_SECRET=\"fedcba9876543210 with a space\"\n"
    "CP_SINGLE='single-quoted-value'\n"
    "CP_HTTP_PORT=8080\n"
)
SECRETS = {
    "CP_SITE_ADDRESS": "causalpulse.example",
    "POSTGRES_PASSWORD": "0123456789abcdef0123456789abcdef",
    "CP_SESSION_SECRET": "fedcba9876543210 with a space",
    "CP_SINGLE": "single-quoted-value",
}


@pytest.mark.parametrize("status", [0, 30])
def test_every_long_value_of_the_environment_file_is_taken_out_of_all_three_outputs(
    server, status
):
    (server.root / ".env").write_text(ENVIRONMENT, encoding="utf-8")
    server.says.write_text(
        "".join(f"api-1  | sent {value} and again {value}\n" for value in SECRETS.values())
        + "api-1  | listening on 8080\n",
        encoding="utf-8",
    )

    result = server.run(status)

    assert result.returncode == status
    outputs = {
        "standard output": result.stdout,
        "the run's log": (server.root / "deploy-run.log").read_text(encoding="utf-8"),
        "the history": (server.root / "deploy.log").read_text(encoding="utf-8"),
    }
    for where, text in outputs.items():
        for name, value in SECRETS.items():
            assert value not in text, (where, name)
            hidden = f"[{name} is not shown]"
            assert f"sent {hidden} and again {hidden}" in text, (where, name)
        # A value shorter than eight characters is not looked for.
        assert "listening on 8080" in text, where


@pytest.mark.parametrize("status", [0, 20])
def test_without_an_environment_file_nothing_of_the_run_is_kept_or_shown(server, status):
    server.release()
    (server.root / ".env").unlink()
    server.says.write_text("api-1  | something a service said\n", encoding="utf-8")

    result = server.run(status)

    assert result.returncode == status
    assert "something a service said" not in result.stdout
    assert "could not be searched for the environment file's values" in result.stdout
    assert f"{server.root}/.env is missing or cannot be read" in result.stdout
    assert not (server.root / "deploy-run.log").exists()
    assert not (server.root / "deploy.log").exists()


def test_standard_output_ends_with_the_last_20000_bytes_of_the_run(server):
    lines = [f"line {number:05d} of what a failing deploy says\n" for number in range(1000)]
    server.says.write_text("".join(lines), encoding="utf-8")

    result = server.run(30)

    run = (server.root / "deploy-run.log").read_text(encoding="utf-8")
    assert len(run) > 30000 and lines[0] in run
    assert result.returncode == 30
    # The end of the run, whole; its beginning, not at all; and no more in
    # all than Systems Manager returns.
    assert run[-20000:] in result.stdout
    assert lines[0] not in result.stdout
    assert "deploy: the stand-in ends with status 30\n" in result.stdout
    assert len(result.stdout) < 24000


def test_each_run_is_added_to_the_history_and_the_logs_are_the_owners_alone(server):
    server.run(20)
    server.run(0)

    history = (server.root / "deploy.log").read_text(encoding="utf-8")
    headings = [line for line in history.splitlines() if line.startswith("==== ")]
    assert [heading.split(" ", 2)[2] for heading in headings] == [
        f"{SHA}, exit status 20",
        f"{SHA}, exit status 0",
    ]
    for name in ("deploy.log", "deploy-run.log", "deploy.lock"):
        mode = stat.S_IMODE((server.root / name).stat().st_mode)
        assert mode & 0o077 == 0, (name, oct(mode))
    # The release is not: the services' user reads the Caddyfile from it.
    mode = stat.S_IMODE((server.root / "release" / "deploy" / "Caddyfile").stat().st_mode)
    assert mode & 0o044 == 0o044, oct(mode)


def test_a_failure_shows_how_full_the_disks_are_and_a_success_does_not(server):
    failed = server.run(20).stdout
    passed = server.run(0).stdout

    def disks(text: str) -> list[str]:
        return [line for line in text.splitlines() if re.match(r"Filesystem\b", line)]

    assert len(disks(failed)) == 1
    assert disks(passed) == []


# ---- The first boot ----


def _function(name: str) -> str:
    """One function of the first-boot template, as text. Terraform's doubled
    dollar is undone, and anything else of Terraform's would be a reason to
    render the template properly."""
    text = FIRST_BOOT.read_text(encoding="utf-8")
    found = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.MULTILINE | re.DOTALL)
    assert found, name
    body = found.group(0).replace("$${", "${")
    assert "%{" not in body
    return body


def _write_environment_file(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the template's `write_environment_file` against a temporary root.
    `chown` is a stand-in: the test is not root."""
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir(exist_ok=True)
    chown = bin_directory / "chown"
    chown.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    chown.chmod(0o755)
    root = tmp_path / "srv"
    root.mkdir(exist_ok=True)
    script = tmp_path / "environment-file.sh"
    script.write_text(
        "set -euo pipefail\n"
        "site_name='causalpulse.example'\n"
        f"root='{root}'\n"
        + _function("say")
        + _function("die")
        + _function("write_environment_file")
        + "write_environment_file\n",
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", str(script)],
        env={"PATH": f"{bin_directory}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_the_environment_file_is_written_once_and_a_second_run_leaves_it_as_it_is(tmp_path):
    first = _write_environment_file(tmp_path)

    assert first.returncode == 0, first.stderr
    written = tmp_path / "srv" / ".env"
    content = written.read_bytes()
    settings = dict(line.split("=", 1) for line in content.decode().splitlines())
    assert list(settings) == ["CP_SITE_ADDRESS", "POSTGRES_PASSWORD", "CP_SESSION_SECRET"]
    assert settings["CP_SITE_ADDRESS"] == "causalpulse.example"
    for name in ("POSTGRES_PASSWORD", "CP_SESSION_SECRET"):
        assert re.fullmatch(r"[0-9a-f]{64}", settings[name]), name
        assert settings[name] not in first.stdout + first.stderr
    assert settings["POSTGRES_PASSWORD"] != settings["CP_SESSION_SECRET"]
    assert stat.S_IMODE(written.stat().st_mode) == 0o600

    # A machine that replaces this one finds the file on the data volume,
    # beside the database that was created with its password.
    second = _write_environment_file(tmp_path)

    assert second.returncode == 0, second.stderr
    assert "exists and is kept" in second.stdout
    assert written.read_bytes() == content
    assert sorted(path.name for path in (tmp_path / "srv").iterdir()) == [".env"]


# From the look at the volume to the end of the decision, whole. The volume is
# formatted in one place: where `blkid` found no file system (status 2, no
# type) and `wipefs` ran to its end and found no signature of any kind. A
# file system that is not ext4 stops the script; so does anything else.
FORMAT_DECISION = """\
    status=0
    fstype=$(blkid -p -s TYPE -o value "$device") || status=$?
    if [ "$status" -eq 0 ] && [ "$fstype" = ext4 ]; then
        say "the data volume holds a file system, which is kept"
    elif [ "$status" -eq 0 ]; then
        die "the data volume holds '$fstype', not ext4; left alone"
    elif [ "$status" -eq 2 ] && [ -z "$fstype" ]; then
        wipe_status=0
        signatures=$(wipefs "$device") || wipe_status=$?
        if [ "$wipe_status" -ne 0 ]; then
            die "wipefs could not read the data volume (status $wipe_status); left alone"
        fi
        if [ -n "$signatures" ]; then
            die "the data volume holds something that is not a file system; left alone"
        fi
        say "the data volume is empty; formatting it"
        mkfs.ext4 -q -L causalpulse-data "$device"
    else
        die "could not tell what the data volume holds (blkid: $status); left alone"
    fi
"""


def test_a_volume_with_any_signature_is_never_formatted():
    text = FIRST_BOOT.read_text(encoding="utf-8")
    code = "".join(
        line for line in text.splitlines(keepends=True) if not line.lstrip().startswith("#")
    )

    # One command in the whole script makes a file system or wipes one, and it
    # stands in the decision held above, whose every other branch stops the
    # script or keeps what is there.
    assert FORMAT_DECISION in code
    destructive = re.findall(
        r"\b(?:mkfs[.\w]*|mke2fs|wipefs -a|wipefs --all|dd|sfdisk|parted)\b", code
    )
    assert destructive == ["mkfs.ext4"]
    # And the volume is only formatted when it is not already mounted: the
    # early return for a mounted volume comes before the decision.
    assert code.index('if mountpoint -q "$root"; then') < code.index(FORMAT_DECISION)


def _install_packages(
    tmp_path: Path, succeeds_at: int
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the template's `install_packages` with a package tool that fails
    until the try numbered `succeeds_at`, and no waiting. What was called is
    returned beside the result."""
    calls = tmp_path / "calls"
    script = tmp_path / "install-packages.sh"
    script.write_text(
        "set -euo pipefail\n"
        f"calls='{calls}'\n"
        "tries=0\n"
        'apt_get() {\n    echo "apt_get $*" >>"$calls"\n'
        '    case $* in *update*) tries=$((tries + 1)) ;; esac\n'
        f'    [ "$tries" -ge {succeeds_at} ]\n}}\n'
        'dpkg() {\n    echo "dpkg $*" >>"$calls"\n    return 1\n}\n'
        'sleep() {\n    echo "sleep $*" >>"$calls"\n}\n'
        + _function("say")
        + _function("die")
        + _function("install_packages")
        + "install_packages one two\n"
        + "say 'the script went on'\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(script)],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result, calls.read_text(encoding="utf-8").splitlines()


def test_packages_that_cannot_be_installed_stop_the_first_boot_after_ten_tries(tmp_path):
    # A retry that gave up and returned would let the script go on to its
    # last lines, and the marker would say that a first boot which installed
    # nothing had finished.
    result, calls = _install_packages(tmp_path, succeeds_at=11)

    assert result.returncode == 1
    assert "the script went on" not in result.stdout
    assert "the packages could not be installed in 10 tries: one two" in result.stderr
    assert calls.count("apt_get -o APT::Update::Error-Mode=any update -q") == 10
    # Between two tries dpkg is asked to finish what it left, and its failure
    # is not the script's.
    assert calls.count("dpkg --configure -a") == 9
    assert calls.count("sleep 30") == 9


def test_packages_that_install_at_a_later_try_let_the_first_boot_go_on(tmp_path):
    result, calls = _install_packages(tmp_path, succeeds_at=5)

    assert result.returncode == 0, result.stderr
    assert "the script went on" in result.stdout
    assert calls.count("apt_get -o APT::Update::Error-Mode=any update -q") == 5
    # The lists are fetched before each install, and the fifth fetch is the
    # first that is followed by one.
    assert calls[-2:] == [
        "apt_get -o APT::Update::Error-Mode=any update -q",
        "apt_get install -y -q one two",
    ]
    assert calls.count("apt_get install -y -q one two") == 1


# Docker's drop-in, whole. `RequiresMountsFor` has Docker wait for the mount
# and `nofail` in the mount's own line lets the machine boot without it, so
# it is `AssertPathIsMountPoint` that stops Docker starting on the bare
# directory, where it would make an empty database on the root disk.
DOCKER_DROP_IN = """\
    cat >/etc/systemd/system/docker.service.d/10-causalpulse-volumes.conf <<UNIT
[Unit]
RequiresMountsFor=$volumes
AssertPathIsMountPoint=$volumes
UNIT
"""


def test_docker_does_not_start_when_its_volumes_directory_is_not_the_data_volume():
    text = FIRST_BOOT.read_text(encoding="utf-8")

    assert DOCKER_DROP_IN in text
    assert "\nvolumes=/var/lib/docker/volumes\n" in text
    # The drop-in is written before Docker is installed and started.
    assert text.index("\n    mount_docker_volumes\n") < text.index("\n    install_docker\n")
