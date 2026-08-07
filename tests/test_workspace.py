import subprocess


def test_cp_core_is_importable():
    import cp_core

    assert cp_core.__version__ == "0.1.0"


def test_make_check_target_exists():
    result = subprocess.run(["make", "-n", "check"], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
