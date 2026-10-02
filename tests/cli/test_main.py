import pytest
from cp_cli.__main__ import main


def test_no_arguments_reports_usage_and_fails():
    # argparse exits 2 on a usage error; a bare invocation must not silently
    # succeed having done nothing.
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_an_unknown_command_is_refused():
    with pytest.raises(SystemExit) as excinfo:
        main(["frobnicate"])
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["simulate", "--help"],
        ["analyze", "--help"],
        ["interpret", "--help"],
        ["replay", "--help"],
        ["bench", "--help"],
        ["bench", "run", "--help"],
        ["bench", "report", "--help"],
    ],
)
def test_every_command_has_help(argv):
    # --help exits 0. This also pins that the `bench` subparser tree exists,
    # which is the only nested one.
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 0
