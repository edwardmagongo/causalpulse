"""`pytest --shard I/N` runs one shard of the test files.

The option lives in `tests/conftest.py`. The nested runs below load that file
itself: its source is written into the `pytester` project as the project's own
`conftest.py`, so what is exercised is the real hook, not a restatement of it.
A nested project has no `pyproject.toml`, so the repository's
`filterwarnings = ["error"]` would not reach it; every nested run passes
`-W error` to put it back, and a warning raised by the hook fails the run.

The last test collects the real tree, once unsharded and once per shard, in
subprocesses. It only collects: no slow test is run.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFTEST = Path(__file__).resolve().parent / "conftest.py"

# Five files, sorted a..e, and seven tests: counting files and counting tests
# give different answers, so a split by test would not pass for a split by file.
FILES = {
    "test_a": "def test_one(): pass\ndef test_two(): pass\n",
    "test_b": "def test_one(): pass\n",
    "test_c": "def test_one(): pass\ndef test_two(): pass\n",
    "test_d": "def test_one(): pass\n",
    "test_e": "def test_one(): pass\n",
}
EVERY_TEST = {
    "test_a.py::test_one",
    "test_a.py::test_two",
    "test_b.py::test_one",
    "test_c.py::test_one",
    "test_c.py::test_two",
    "test_d.py::test_one",
    "test_e.py::test_one",
}

SLOW_TIER = "slow and not services and not live"
REAL_SHARDS = 8


@pytest.fixture
def project(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(REAL_CONFTEST.read_text())
    pytester.makepyfile(**FILES)
    return pytester


def _node_ids(output: str) -> set[str]:
    return {line.strip() for line in output.splitlines() if "::" in line}


def _selected(project: pytest.Pytester, *args: str) -> set[str]:
    result = project.runpytest("--collect-only", "-q", "-W", "error", *args)
    assert result.ret == pytest.ExitCode.OK, result.stdout.str() + result.stderr.str()
    return _node_ids(result.stdout.str())


def _files(node_ids: set[str]) -> set[str]:
    return {node_id.split("::")[0] for node_id in node_ids}


def test_two_shards_partition_the_tests(project: pytest.Pytester) -> None:
    first = _selected(project, "--shard", "1/2")
    second = _selected(project, "--shard", "2/2")

    assert first.isdisjoint(second)
    assert first | second == EVERY_TEST


def test_files_are_dealt_round_robin_in_sorted_order(project: pytest.Pytester) -> None:
    first = _selected(project, "--shard", "1/2")
    second = _selected(project, "--shard", "2/2")

    assert _files(first) == {"test_a.py", "test_c.py", "test_e.py"}
    assert _files(second) == {"test_b.py", "test_d.py"}


def test_a_single_shard_is_every_test(project: pytest.Pytester) -> None:
    assert _selected(project, "--shard", "1/1") == EVERY_TEST


def test_no_option_is_every_test(project: pytest.Pytester) -> None:
    assert _selected(project) == EVERY_TEST


@pytest.mark.parametrize("value", ["0/4", "5/4", "a/b", "3"])
def test_a_malformed_shard_is_a_usage_error_naming_it(
    project: pytest.Pytester, value: str
) -> None:
    result = project.runpytest("-W", "error", "--shard", value)

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert f"'{value}'" in result.stderr.str()


def test_the_deselected_tests_are_reported(project: pytest.Pytester) -> None:
    result = project.runpytest("-W", "error", "--shard", "2/2")

    result.assert_outcomes(passed=2, deselected=5)


def _passed(result: pytest.RunResult) -> set[str]:
    return {
        line.removeprefix("PASSED ").strip()
        for line in result.stdout.lines
        if line.startswith("PASSED ")
    }


def test_a_shard_runs_the_same_tests_under_xdist(
    project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every xdist worker collects for itself, and the workers must agree on the
    # shard or the run aborts. Subprocesses, so the nested workers are real
    # ones; without the outer run's worker variables, so the nested run is a
    # controller and not taken for a worker of this one.
    for name in [name for name in os.environ if name.startswith("PYTEST_XDIST_")]:
        monkeypatch.delenv(name)
    args = ["-rA", "-W", "error", "-p", "no:cacheprovider", "--shard", "1/2"]

    serial = project.runpytest_subprocess(*args)
    parallel = project.runpytest_subprocess(*args, "-n", "2")

    assert serial.ret == pytest.ExitCode.OK, serial.stdout.str() + serial.stderr.str()
    assert parallel.ret == pytest.ExitCode.OK, parallel.stdout.str() + parallel.stderr.str()
    assert _passed(serial) == {
        "test_a.py::test_one",
        "test_a.py::test_two",
        "test_c.py::test_one",
        "test_c.py::test_two",
        "test_e.py::test_one",
    }
    assert _passed(parallel) == _passed(serial)


def test_files_are_dealt_after_marker_selection(pytester: pytest.Pytester) -> None:
    # Six files, of which the three even-numbered ones hold a marked test. Dealt
    # before `-m`, shard 1/2 would be files 0, 2, 4 and hold every marked test,
    # leaving shard 2/2 empty; dealt after, the three marked files split 2 and 1.
    pytester.makeconftest(REAL_CONFTEST.read_text())
    pytester.makeini("[pytest]\nmarkers =\n    heavy: for this test\n")
    marked = "import pytest\n@pytest.mark.heavy\ndef test_one(): pass\ndef test_two(): pass\n"
    plain = "def test_one(): pass\n"
    pytester.makepyfile(**{f"test_{k}": marked if k % 2 == 0 else plain for k in range(6)})

    first = _selected(pytester, "-m", "heavy", "--shard", "1/2")
    second = _selected(pytester, "-m", "heavy", "--shard", "2/2")

    assert first == {"test_0.py::test_one", "test_4.py::test_one"}
    assert second == {"test_2.py::test_one"}


def _collect_real(shard: str | None) -> set[str]:
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", SLOW_TIER]
    command += ["-p", "no:cacheprovider"]
    if shard is not None:
        command += ["--shard", shard]
    # Without the outer run's PYTEST_* variables (the xdist worker id, the
    # current test), so the collection is the one a shell would get.
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"--shard {shard}:\n{result.stdout}\n{result.stderr}"
    return _node_ids(result.stdout)


def test_eight_shards_partition_the_real_slow_tier() -> None:
    shards = [f"{index}/{REAL_SHARDS}" for index in range(1, REAL_SHARDS + 1)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        whole, *parts = pool.map(_collect_real, [None, *shards])

    assert whole, "the unsharded slow tier collected nothing"
    assert all(parts), "a shard of the slow tier is empty"
    assert sum(len(part) for part in parts) == len(set().union(*parts)), "shards overlap"
    assert set().union(*parts) == whole
