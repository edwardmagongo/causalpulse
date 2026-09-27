import pytest
from cp_causal.registry import get_estimator
from cp_core.manifest import FixtureReference
from cp_core.values import EstimatorConfig
from cp_report.pipeline import FitFailure, fit_each, select_estimators

REFERENCE = FixtureReference(path="d.csv", content_sha256="0" * 64)
CONFIG = EstimatorConfig(random_seed=0)


def test_no_request_means_the_routers_eligible_set_with_every_exclusion_kept(cross_section):
    decisions, names = select_estimators(cross_section, None)

    assert names == ("aipw", "causal_forest", "difference_in_means", "ipw", "lin_regression")
    excluded = {d.estimator_name: d.reasons for d in decisions if not d.eligible}
    assert "synthetic_control" in excluded
    assert any("panel" in reason for reason in excluded["synthetic_control"])


def test_a_request_keeps_its_order_and_drops_repeats(cross_section):
    _, names = select_estimators(
        cross_section, ["lin_regression", "difference_in_means", "lin_regression"]
    )

    assert names == ("lin_regression", "difference_in_means")


def test_an_unknown_name_is_refused_before_anything_is_fitted(cross_section):
    with pytest.raises(ValueError, match=r"unknown estimator\(s\): \['nope'\]"):
        select_estimators(cross_section, ["difference_in_means", "nope"])


def test_an_ineligible_request_is_refused_with_the_routers_own_reasons(cross_section):
    with pytest.raises(ValueError, match="synthetic_control: requires panel data"):
        select_estimators(cross_section, ["synthetic_control"])


def test_an_empty_request_is_refused_rather_than_read_as_route_everything(cross_section):
    with pytest.raises(ValueError, match="at least one"):
        select_estimators(cross_section, [])


def test_one_raising_estimator_leaves_the_others_and_is_absent_from_the_manifest(
    cross_section, monkeypatch
):
    def boom(self, dataset, config):
        raise RuntimeError("boom")

    monkeypatch.setattr(get_estimator("lin_regression"), "fit", boom)

    estimates, failures, manifest = fit_each(
        cross_section, REFERENCE, ("difference_in_means", "lin_regression"), CONFIG
    )

    assert [e.method for e in estimates] == ["difference_in_means"]
    assert failures == (FitFailure(method="lin_regression", error="boom"),)
    assert [name for name, _ in manifest.estimators] == ["difference_in_means"]
    assert manifest.dataset == REFERENCE


def test_every_estimator_failing_raises_with_each_error(cross_section, monkeypatch):
    def boom(self, dataset, config):
        raise RuntimeError("boom")

    monkeypatch.setattr(get_estimator("difference_in_means"), "fit", boom)

    with pytest.raises(
        ValueError, match="every requested estimator failed.*difference_in_means: boom"
    ):
        fit_each(cross_section, REFERENCE, ("difference_in_means",), CONFIG)
