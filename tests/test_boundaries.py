import subprocess

import pytest


def test_import_linter_contracts_hold():
    result = subprocess.run(["uv", "run", "lint-imports"], capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr


def test_cp_causal_cannot_import_cp_simulation():
    import cp_causal

    source_root = cp_causal.__path__[0]
    offenders = subprocess.run(
        ["grep", "-rn", "cp_simulation", source_root],
        capture_output=True,
        text=True,
    )

    # grep: 0 = matches found, 1 = no matches, 2+ = error (e.g. an unresolvable
    # path). Without this the empty-stdout assertion passes having scanned nothing.
    assert offenders.returncode in (0, 1), offenders.stderr
    assert offenders.stdout == "", (
        "cp_causal must never reference cp_simulation:\n" + offenders.stdout
    )


@pytest.mark.parametrize("module_name", ["cp_core", "cp_statistics"])
def test_contract_layer_cannot_import_cp_simulation(module_name):
    # cp_core and cp_simulation both sit "below" cp_causal in different senses:
    # cp_core is the shared contract layer, cp_simulation reads ground truth.
    # Neither the contract layer nor the statistics layer may reach into
    # cp_simulation, or estimators built on top could reach ground truth
    # transitively without ever importing cp_simulation directly.
    module = __import__(module_name)

    source_root = module.__path__[0]
    offenders = subprocess.run(
        ["grep", "-rn", "cp_simulation", source_root],
        capture_output=True,
        text=True,
    )

    # grep: 0 = matches found, 1 = no matches, 2+ = error (e.g. an unresolvable
    # path). Without this the empty-stdout assertion passes having scanned nothing.
    assert offenders.returncode in (0, 1), offenders.stderr
    assert offenders.stdout == "", (
        f"{module_name} must never reference cp_simulation:\n" + offenders.stdout
    )


def test_panel_estimators_do_not_reach_panel_ground_truth():
    # The panel DGP carries PanelGroundTruth. If any estimator could import it,
    # every recovery number in this repository would be worthless.
    import cp_causal

    source_root = cp_causal.__path__[0]
    for symbol in ("PanelGroundTruth", "SimulatedPanel", "true_att_gt"):
        offenders = subprocess.run(
            ["grep", "-rn", symbol, source_root], capture_output=True, text=True
        )
        assert offenders.returncode in (0, 1), offenders.stderr
        assert offenders.stdout == "", (
            f"cp_causal must never reference {symbol}:\n" + offenders.stdout
        )


def test_synthetic_control_does_not_reach_donor_pool_ground_truth():
    # cp_causal must have no code path to DonorScenarioSpec, ContaminatedDonorPanel,
    # or build_donor_scenario. These are ground-truth generation utilities in
    # cp_simulation. If any estimator could import them, recovery numbers would be
    # contaminated by knowledge of the donor pool specification.
    import cp_causal

    source_root = cp_causal.__path__[0]
    for symbol in ("DonorScenarioSpec", "ContaminatedDonorPanel", "build_donor_scenario"):
        offenders = subprocess.run(
            ["grep", "-rn", symbol, source_root], capture_output=True, text=True
        )
        assert offenders.returncode in (0, 1), offenders.stderr
        assert offenders.stdout == "", (
            f"cp_causal must never reference {symbol}:\n" + offenders.stdout
        )


def test_cp_core_contains_no_algorithms():
    # cp-core holds contracts only; depending on scipy would signal that
    # estimation logic had leaked into the contract layer.
    import tomllib
    from pathlib import Path

    manifest = Path("packages/cp-core/pyproject.toml").read_bytes()
    dependencies = tomllib.loads(manifest.decode())["project"]["dependencies"]

    assert not any(dep.startswith("scipy") for dep in dependencies)
