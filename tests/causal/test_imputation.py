"""Filling missing outcomes, and refusing to when the fill would be a fiction."""

import cp_causal  # noqa: F401  -- registers the estimators
import numpy as np
import pandas as pd
import pytest
from cp_causal.imputation import DEFAULT_IMPUTATIONS, impute_once, multiple_impute
from cp_causal.registry import all_estimators, get_estimator
from cp_causal.sensitivity import compare_missing_data, sensitivity_report
from cp_core.dataset import CausalDataset, CausalSchema, complete_cases
from cp_core.values import EstimatorConfig
from cp_simulation.components import MissingnessMechanism
from cp_simulation.scenarios import build_scenario


def _dataset(*, missing_rows=(0, 1), auxiliary=False, single_class=False, n=40):
    rng = np.random.default_rng(0)
    outcome = np.zeros(n) if single_class else np.tile([0.0, 1.0], n // 2)
    frame = pd.DataFrame(
        {
            "unit_id": np.arange(n),
            "treated": np.tile([0.0, 1.0], n // 2),
            "x1": rng.normal(size=n),
            "outcome": outcome,
        }
    )
    if auxiliary:
        frame["outcome_proxy"] = frame["outcome"] + rng.normal(0.0, 0.3, size=n)
    frame.loc[list(missing_rows), "outcome"] = np.nan
    schema = CausalSchema(
        unit="unit_id",
        treatment="treated",
        outcome="outcome",
        covariates=("x1",),
        outcome_type="binary",
        auxiliary_columns=("outcome_proxy",) if auxiliary else (),
    )
    return CausalDataset(frame=frame, schema=schema)


def test_it_fills_every_gap_and_leaves_observed_values_alone():
    dataset = _dataset()
    filled = impute_once(dataset, np.random.default_rng(0), draw_parameters=False)
    assert not filled.frame["outcome"].isna().any()
    observed = ~dataset.frame["outcome"].isna()
    np.testing.assert_array_equal(
        filled.frame.loc[observed, "outcome"].to_numpy(),
        dataset.frame.loc[observed, "outcome"].to_numpy(),
    )


def test_the_filled_outcome_stays_binary():
    # A probability would keep the point estimate closer but break the
    # schema's own binary contract and every downstream check that rests on
    # it, so the fill is a Bernoulli draw rather than a predicted mean.
    filled = impute_once(_dataset(), np.random.default_rng(0), draw_parameters=False)
    assert set(np.unique(filled.frame["outcome"].to_numpy())) <= {0.0, 1.0}


def test_a_dataset_with_nothing_missing_comes_back_unchanged():
    dataset = _dataset(missing_rows=())
    filled = impute_once(dataset, np.random.default_rng(0), draw_parameters=False)
    assert filled is dataset


def test_drawing_parameters_changes_the_fill_across_calls():
    # Proper multiple imputation must vary the imputation model itself, not
    # only the Bernoulli draw. Two draws that differed only in the Bernoulli
    # would understate between-imputation variance.
    dataset = _dataset(missing_rows=tuple(range(0, 30)))
    a = impute_once(dataset, np.random.default_rng(1), draw_parameters=True)
    b = impute_once(dataset, np.random.default_rng(2), draw_parameters=True)
    assert not np.array_equal(
        a.frame["outcome"].to_numpy(), b.frame["outcome"].to_numpy()
    )


def test_it_refuses_when_a_declared_auxiliary_has_gaps():
    dataset = _dataset(auxiliary=True)
    frame = dataset.frame.copy()
    frame.loc[5, "outcome_proxy"] = np.nan
    broken = CausalDataset(frame=frame, schema=dataset.schema)
    with pytest.raises(ValueError, match="outcome_proxy"):
        impute_once(broken, np.random.default_rng(0), draw_parameters=False)


def test_it_refuses_by_name_when_a_covariate_has_gaps():
    # The auxiliary was guarded and the rest of the same design matrix was
    # not, so a gap in a covariate reached sklearn and came back as "Input X
    # contains NaN" -- naming no column, from a library the caller never
    # invoked. That is exactly the late, undiagnosable failure
    # `require_no_missing` was built to end, and it is reachable through
    # `sensitivity_report`, which hands `compare_missing_data` the raw
    # dataset by design. Every column the model reads must refuse the same
    # way, naming itself. The treatment column is the one part of the design
    # already covered -- `CausalDataset` refuses a gap there at construction
    # -- so covariates are the hole.
    dataset = _dataset()
    frame = dataset.frame.copy()
    frame.loc[5, "x1"] = np.nan
    broken = CausalDataset(frame=frame, schema=dataset.schema)
    with pytest.raises(ValueError, match="x1"):
        impute_once(broken, np.random.default_rng(0), draw_parameters=False)


def test_it_refuses_when_the_observed_outcomes_are_all_one_class():
    # A logistic model cannot be fitted on one class, and filling every gap
    # with the only value ever seen would manufacture certainty.
    with pytest.raises(ValueError, match="one class"):
        impute_once(
            _dataset(single_class=True), np.random.default_rng(0), draw_parameters=False
        )


def test_the_auxiliary_enters_the_imputation_model():
    # With a near-deterministic proxy the fill should track it. This is what
    # distinguishes an imputation that uses the auxiliary from one that
    # silently ignores it -- the failure mode that would collapse the whole
    # phase's efficiency result into a no-op.
    #
    # 170 observed rows rather than the 10 this started with. The proxy is
    # drawn on a +/-5 scale, and on raw columns that scale bought it an
    # effective exemption from the L2 penalty `LogisticRegression` applies by
    # default -- so ten rows were enough to clear this bar only because the
    # model was reading the proxy's units as evidence. Standardised, the proxy
    # has to earn its coefficient against the same penalty as every other
    # column, and needs a training set to earn it on. Measured across ten rng
    # seeds the correlation here is 0.87 at worst, against 0.61 at n=40.
    dataset = _dataset(n=200, missing_rows=tuple(range(0, 30)), auxiliary=True)
    frame = dataset.frame.copy()
    frame["outcome_proxy"] = np.where(np.arange(len(frame)) % 2 == 0, -5.0, 5.0)
    steered = CausalDataset(frame=frame, schema=dataset.schema)
    filled = impute_once(steered, np.random.default_rng(0), draw_parameters=False)
    imputed = filled.frame.loc[list(range(0, 30)), "outcome"].to_numpy()
    proxy = frame.loc[list(range(0, 30)), "outcome_proxy"].to_numpy()
    assert float(np.corrcoef(imputed, proxy)[0, 1]) > 0.8


CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)


def _simulated(scenario="D", n_units=600, seed=0, proxy=None):
    dgp = build_scenario(
        scenario,
        missingness=MissingnessMechanism(mechanism="MAR", rate=0.25, strength=2.0),
        outcome_proxy=proxy,
    )
    return dgp.generate(n_units=n_units, rng=np.random.default_rng(seed)).dataset


def test_it_reports_rubin_as_its_interval_basis():
    estimate = multiple_impute(_simulated(), "difference_in_means", CONFIG, m=5)
    assert estimate is not None
    assert estimate.ci_basis == "rubin"
    assert estimate.method == "difference_in_means"


def test_it_carries_the_pooling_as_diagnostics():
    estimate = multiple_impute(_simulated(), "difference_in_means", CONFIG, m=5)
    assert estimate is not None
    names = set(estimate.diagnostics)
    assert {
        "imputations",
        "within_variance",
        "between_variance",
        "proportion_variance_missing",
        "fraction_missing_information",
        "barnard_rubin_df",
    } <= names
    assert estimate.diagnostics["imputations"].value == 5.0


def test_it_offers_no_bootstrap_interval():
    # The project asks for a bootstrap interval alongside an analytic one.
    # Bootstrapping a Rubin-pooled quantity would be a bootstrap of a
    # bootstrap, so this is a deliberate exception rather than an omission.
    estimate = multiple_impute(_simulated(), "difference_in_means", CONFIG, m=5)
    assert estimate is not None
    assert estimate.bootstrap_ci is None


def test_a_wider_ci_level_widens_the_pooled_interval():
    # Nothing pinned that the interval actually reads `config.ci_level`
    # rather than a hardcoded 0.95. Same dataset and seed, two configs that
    # differ only in ci_level: the wider level must produce a wider interval
    # (a larger critical t-value at the same pooled se) and each returned
    # `Interval.level` must match the config it came from.
    dataset = _simulated()
    narrow_config = EstimatorConfig(
        bootstrap_reps=25, random_seed=1, ci_level=0.80
    )
    wide_config = EstimatorConfig(bootstrap_reps=25, random_seed=1, ci_level=0.99)

    narrow = multiple_impute(dataset, "difference_in_means", narrow_config, m=5)
    wide = multiple_impute(dataset, "difference_in_means", wide_config, m=5)
    assert narrow is not None
    assert wide is not None

    assert narrow.ci.level == 0.80
    assert wide.ci.level == 0.99
    narrow_width = narrow.ci.upper - narrow.ci.lower
    wide_width = wide.ci.upper - wide.ci.lower
    assert wide_width > narrow_width


def test_it_is_wider_than_a_typical_single_completion():
    # Pin the property the test name claims, not an arithmetic fact
    # `pool_rubin` guarantees for any positive between-variance regardless of
    # whether `multiple_impute` wired the right fits into it (`se**2 >
    # within_variance` is exactly that guarantee, true of `total = within +
    # (1 + 1/m) * between` for any m and any between > 0, so it would pass
    # even if the pooled point/se were quietly swapped for garbage).
    #
    # Compared against the MEAN width across the m completions actually
    # pooled, not one arbitrary completion, because a single completion's
    # width relative to the pooled one is only a probabilistic claim on one
    # seed. The mean is reconstructed by replaying the identical rng sequence
    # `multiple_impute` consumes internally (same seed, same
    # draw_parameters=True resampling), so this is exactly the set of fits
    # that went into the pool, not a fresh independent draw.
    #
    # Verified over 30 seeds (0-29) that pooled_width > mean_width holds with
    # margin ranging from about 0.0106 to 0.075 (mean widths cluster near
    # 0.145-0.153; pooled widths near 0.15-0.22) -- comfortably robust, never
    # close to reversing. See task-6-report.md for the full sweep.
    dataset = _simulated()
    estimate = multiple_impute(dataset, "difference_in_means", CONFIG, m=10)
    assert estimate is not None
    pooled_width = estimate.ci.upper - estimate.ci.lower

    rng = np.random.default_rng(np.random.SeedSequence(0))
    estimator_class = get_estimator("difference_in_means")
    widths = []
    for _ in range(10):
        completed = impute_once(dataset, rng, draw_parameters=True)
        fit = estimator_class().fit(completed, CONFIG)
        widths.append(fit.ci.upper - fit.ci.lower)
    mean_width = float(np.mean(widths))

    assert pooled_width > mean_width


def test_a_panel_estimator_is_refused():
    assert multiple_impute(_simulated(), "twfe", CONFIG, m=5) is None


def test_a_dataset_with_no_missing_outcomes_is_refused():
    complete = build_scenario("D").generate(
        n_units=400, rng=np.random.default_rng(0)
    ).dataset
    assert multiple_impute(complete, "difference_in_means", CONFIG, m=5) is None


@pytest.mark.parametrize(
    "name", ["difference_in_means", "lin_regression", "ipw", "aipw"]
)
def test_every_eligible_cross_sectional_estimator_pools(name):
    # Enumerated rather than predicated. Phase 9's bsts defect and Phase 10's
    # misclassification defect were both a family predicate admitting an
    # estimator whose fields meant something else.
    estimate = multiple_impute(_simulated(), name, CONFIG, m=5)
    assert estimate is not None
    assert estimate.ci_basis == "rubin"


@pytest.mark.slow
def test_an_assumption_that_disagrees_across_completions_is_not_claimed():
    # A check whose verdict depends on which completion it saw has not been
    # tested, and reporting the first one's answer would be the same overclaim
    # a hardcoded WARN status made in Phase 8.
    #
    # Pins the downgrade actually firing end to end, not just that every
    # status is one of the four the type permits (`AssumptionCheck` already
    # guarantees that for any assumptions object, so it would pass even if
    # `_pool_assumptions` never downgraded anything). `causal_forest` at
    # m=20 on scenario D produces `calibration_test == "NOT_TESTABLE"` on
    # seed 1 (also reproduced on seed 5); confirmed by direct measurement
    # rather than assumed. About 19s per call, hence @pytest.mark.slow.
    estimate = multiple_impute(_simulated(seed=1), "causal_forest", CONFIG, m=20, seed=1)
    assert estimate is not None
    calibration = next(
        check for check in estimate.assumptions if check.name == "calibration_test"
    )
    assert calibration.status == "WARN"
    assert "reached different verdicts across the imputed datasets" in (
        calibration.interpretation
    )


def test_the_default_imputation_count_is_reported_in_diagnostics():
    estimate = multiple_impute(_simulated(), "difference_in_means", CONFIG)
    assert estimate is not None
    assert estimate.diagnostics["imputations"].value == float(DEFAULT_IMPUTATIONS)


def test_m_below_two_is_refused_with_a_clear_error():
    with pytest.raises(ValueError, match="at least two"):
        multiple_impute(_simulated(), "difference_in_means", CONFIG, m=1)


def _fake_effect_estimate(assumptions, diagnostics=None):
    from cp_core.dataset import CausalDataset as _CausalDataset  # noqa: F401
    from cp_core.results import EffectEstimate, FitMetadata
    from cp_core.values import Interval

    return EffectEstimate(
        estimand="ATE",
        method="fake",
        point=0.1,
        se=0.02,
        ci=Interval(lower=0.05, upper=0.15),
        ci_basis="analytic",
        bootstrap_ci=None,
        p_value=0.01,
        n_treated=100,
        n_control=100,
        scale="absolute",
        diagnostics={} if diagnostics is None else diagnostics,
        assumptions=assumptions,
        counterfactual=None,
        fit_metadata=FitMetadata.capture(
            estimator="fake", config_hash="deadbeef", runtime_seconds=0.0, random_seed=0
        ),
    )


def _check(name, status):
    from cp_core.results import AssumptionCheck

    return AssumptionCheck(
        name=name,
        kind="identification",
        status=status,
        statistic=None,
        threshold=None,
        interpretation="an interpretation",
        remedy=None,
    )


def test_pool_assumptions_reports_the_worst_verdict_a_disagreeing_check_reached():
    # Reporting NOT_TESTABLE here would be an UPGRADE, not a downgrade:
    # `score_validity` names NOT_TESTABLE checks without moving the band and
    # `_SEVERITY` does not rank it, so a WARN two completions out of three
    # reached would stop counting anywhere downstream. The pooled verdict is
    # the most severe one any completion reached.
    from cp_causal.imputation import _pool_assumptions

    fits = [
        _fake_effect_estimate((_check("positivity", "PASS"),)),
        _fake_effect_estimate((_check("positivity", "WARN"),)),
        _fake_effect_estimate((_check("positivity", "PASS"),)),
    ]
    pooled = _pool_assumptions(fits)
    assert len(pooled) == 1
    assert pooled[0].status == "WARN"
    # No single completion's statistic describes the pooled result.
    assert pooled[0].statistic is None
    assert pooled[0].threshold is None
    assert "most severe verdict" in pooled[0].interpretation


def test_pool_assumptions_carries_a_fail_only_some_completions_reached():
    # The reachable worst case: nineteen identifying FAILs and one PASS must
    # not pool to a status that escapes the WEAK cap entirely.
    from cp_causal.imputation import _pool_assumptions

    fits = [_fake_effect_estimate((_check("positivity", "FAIL"),)) for _ in range(19)]
    fits.append(_fake_effect_estimate((_check("positivity", "PASS"),)))
    pooled = _pool_assumptions(fits)
    assert pooled[0].status == "FAIL"


def test_pool_assumptions_will_not_claim_a_pass_a_completion_could_not_test():
    # Nothing severe is being hidden here, but a check one completion could
    # not answer is not a PASS either. NOT_TESTABLE is the honest verdict,
    # and it is not an upgrade because PASS is already the mildest rank.
    from cp_causal.imputation import _pool_assumptions

    fits = [
        _fake_effect_estimate((_check("positivity", "PASS"),)),
        _fake_effect_estimate((_check("positivity", "NOT_TESTABLE"),)),
    ]
    pooled = _pool_assumptions(fits)
    assert pooled[0].status == "NOT_TESTABLE"


def test_pool_assumptions_prefers_a_warn_to_an_untestable_verdict():
    # The mirror of the case above: once something rankable and worse than
    # PASS was observed, it outranks the completions that could not answer.
    from cp_causal.imputation import _pool_assumptions

    fits = [
        _fake_effect_estimate((_check("positivity", "NOT_TESTABLE"),)),
        _fake_effect_estimate((_check("positivity", "WARN"),)),
    ]
    pooled = _pool_assumptions(fits)
    assert pooled[0].status == "WARN"


def test_a_disagreeing_warn_still_moves_the_validity_band():
    # The consequence the unit tests above exist for, pinned end to end: a
    # WARN that only some completions reached must still cost the estimate a
    # band. Pooling it to NOT_TESTABLE left the band at the observational
    # ceiling, which reported a weaker analysis as a stronger one.
    from cp_causal.imputation import _pool_assumptions
    from cp_causal.validity import score_validity

    fits = [
        _fake_effect_estimate((_check("positivity", "WARN"),)),
        _fake_effect_estimate((_check("positivity", "PASS"),)),
    ]
    estimate = _fake_effect_estimate(_pool_assumptions(fits))
    assert estimate.worst_assumption_status == "WARN"
    assert score_validity(estimate, provenance="observational").band == "WEAK"


def test_pool_assumptions_keeps_a_check_every_completion_agrees_on():
    from cp_causal.imputation import _pool_assumptions

    fits = [
        _fake_effect_estimate((_check("randomised_assignment", "NOT_TESTABLE"),)),
        _fake_effect_estimate((_check("randomised_assignment", "NOT_TESTABLE"),)),
    ]
    pooled = _pool_assumptions(fits)
    assert len(pooled) == 1
    assert pooled[0].status == "NOT_TESTABLE"
    assert pooled[0].interpretation == "an interpretation"


def test_pool_assumptions_will_not_claim_a_pass_from_only_some_completions():
    # A check present in the first completion's list but missing from a
    # later one has not been tested consistently. An all-PASS partial report
    # is still NOT_TESTABLE: PASS is the mildest rank, so nothing severe is
    # hidden, and a check some completions never reported is not a PASS.
    from cp_causal.imputation import _pool_assumptions

    fits = [
        _fake_effect_estimate(
            (_check("positivity", "PASS"), _check("covariate_balance", "PASS"))
        ),
        _fake_effect_estimate((_check("positivity", "PASS"),)),
    ]
    pooled = _pool_assumptions(fits)
    names = {check.name: check.status for check in pooled}
    assert names["positivity"] == "PASS"
    assert names["covariate_balance"] == "NOT_TESTABLE"


@pytest.mark.parametrize("severe", ["FAIL", "WARN"])
def test_pool_assumptions_carries_a_severe_verdict_out_of_a_partial_report(severe):
    # The absence branch's own laundering case, and the twin of the
    # disagreement branch's. NOT_TESTABLE carries no rank in `_SEVERITY` and
    # no weight in `score_validity`, so routing a FAIL into it hides the
    # FAIL -- and "some completions could not even report this check" is
    # weaker evidence of health than "the completions disagreed", not
    # stronger. An absent report votes NOT_TESTABLE; it cannot outvote a
    # verdict some completion actually reached.
    from cp_causal.imputation import _pool_assumptions

    fits = [
        _fake_effect_estimate(
            (_check("positivity", "PASS"), _check("covariate_balance", severe))
        ),
        _fake_effect_estimate((_check("positivity", "PASS"),)),
    ]
    pooled = _pool_assumptions(fits)
    names = {check.name: check.status for check in pooled}
    assert names["covariate_balance"] == severe


def test_pool_assumptions_keeps_a_check_absent_from_the_first_completion():
    # The reverse direction: a check the first completion lacks but a later
    # one carries must not be silently dropped from the pooled result --
    # dropping it would hide the check from a reader entirely. It is kept,
    # and at the WARN the one completion that could test it reached.
    from cp_causal.imputation import _pool_assumptions

    fits = [
        _fake_effect_estimate((_check("positivity", "PASS"),)),
        _fake_effect_estimate(
            (_check("positivity", "PASS"), _check("covariate_balance", "WARN"))
        ),
    ]
    pooled = _pool_assumptions(fits)
    names = {check.name: check.status for check in pooled}
    assert names["positivity"] == "PASS"
    assert names["covariate_balance"] == "WARN"


def _diagnostic(name, value):
    from cp_core.values import Diagnostic

    return Diagnostic(
        name=name, value=value, unit="units", interpretation="How many units."
    )


def test_pooled_diagnostics_average_what_every_completion_reported():
    # §5.3 says the pooled diagnostics "gain" the pooling fields; shipped
    # code replaced the mapping wholesale, leaving the multiple-imputation
    # arm the least inspectable of the three -- no trimming count, no fold
    # count, no overlap -- even though `compare()` may pick it as the
    # highest-validity line. Averaging over completions is the same
    # operation Rubin's rules already apply to the point estimate.
    from cp_causal.imputation import _pool_estimator_diagnostics

    fits = [
        _fake_effect_estimate((), {"n_trimmed": _diagnostic("n_trimmed", 4.0)}),
        _fake_effect_estimate((), {"n_trimmed": _diagnostic("n_trimmed", 10.0)}),
    ]
    pooled = _pool_estimator_diagnostics(fits)
    assert pooled["n_trimmed"].value == 7.0
    assert "mean over the 2" in pooled["n_trimmed"].interpretation


def test_a_diagnostic_missing_from_a_completion_is_dropped_rather_than_averaged():
    # A mean over a subset of the completions would describe neither the
    # subset nor the pooled fit, and there is no honest name for it.
    from cp_causal.imputation import _pool_estimator_diagnostics

    fits = [
        _fake_effect_estimate((), {"n_trimmed": _diagnostic("n_trimmed", 4.0)}),
        _fake_effect_estimate((), {}),
    ]
    assert _pool_estimator_diagnostics(fits) == {}


def test_an_array_valued_diagnostic_is_dropped_rather_than_averaged():
    # Averaging arrays would need every completion to agree on shape and
    # meaning, which nothing guarantees. Reporting nothing beats reporting a
    # number whose definition depends on that.
    from cp_causal.imputation import _pool_estimator_diagnostics

    fits = [
        _fake_effect_estimate((), {"curve": _diagnostic("curve", np.array([1.0, 2.0]))}),
        _fake_effect_estimate((), {"curve": _diagnostic("curve", np.array([3.0, 4.0]))}),
    ]
    assert _pool_estimator_diagnostics(fits) == {}


def test_the_pooled_estimate_still_carries_the_estimators_own_diagnostics():
    estimate = multiple_impute(_simulated(), "aipw", CONFIG, m=3)
    assert estimate is not None
    # The pooling fields, and the estimator's own alongside them.
    assert "proportion_variance_missing" in estimate.diagnostics
    assert "n_trimmed" in estimate.diagnostics
    assert "cross_fit_folds" in estimate.diagnostics


def test_the_pooled_fit_records_the_seed_that_actually_reproduces_it():
    # FitMetadata is the artifact a reader uses to re-run a result. It
    # recorded the ESTIMATOR's seed, while the imputation ran from
    # `multiple_impute`'s own `seed` -- which was recorded nowhere, and which
    # `config_hash` did not see either. Two materially different pooled
    # estimates therefore carried the same configuration fingerprint and a
    # seed reproducing neither.
    first = multiple_impute(_simulated(), "difference_in_means", CONFIG, m=3, seed=3)
    second = multiple_impute(_simulated(), "difference_in_means", CONFIG, m=3, seed=99)
    more = multiple_impute(_simulated(), "difference_in_means", CONFIG, m=5, seed=3)
    assert first is not None and second is not None and more is not None

    assert first.point != second.point
    assert first.fit_metadata.random_seed == 3
    assert second.fit_metadata.random_seed == 99
    # `m` and `seed` both change the pooled result, so both must reach the
    # fingerprint.
    assert first.fit_metadata.config_hash != second.fit_metadata.config_hash
    assert first.fit_metadata.config_hash != more.fit_metadata.config_hash


def test_the_comparison_carries_all_three_arms_and_the_tipping_point():
    analysis = compare_missing_data(_simulated(), "difference_in_means", CONFIG, m=5)
    assert analysis is not None
    assert analysis.complete_case.ci_basis == "analytic"
    assert analysis.single_imputation.ci_basis == "analytic"
    assert analysis.multiple_imputation.ci_basis == "rubin"
    assert analysis.tipping_point is not None
    assert analysis.tipping_point_reason is None
    assert analysis.n_missing_outcomes > 0


def test_the_complete_case_arm_uses_fewer_rows_than_the_imputed_arms():
    # The distinguishing property of the comparison: one arm answers with
    # less data, the other two answer with a model for what is missing.
    analysis = compare_missing_data(_simulated(), "difference_in_means", CONFIG, m=5)
    assert analysis is not None
    complete_case_n = analysis.complete_case.n_treated + analysis.complete_case.n_control
    imputed_n = (
        analysis.multiple_imputation.n_treated + analysis.multiple_imputation.n_control
    )
    assert complete_case_n < imputed_n


def test_a_panel_estimator_gets_no_comparison():
    assert compare_missing_data(_simulated(), "twfe", CONFIG, m=5) is None


def test_the_report_carries_the_comparison_with_no_reason():
    dataset = _simulated()
    reduced, _ = complete_cases(dataset)
    estimate = all_estimators()["difference_in_means"]().fit(reduced, CONFIG)
    report = sensitivity_report(dataset, "difference_in_means", estimate, CONFIG)
    assert report.missing_data is not None
    assert report.missing_data_reason is None
    assert report.missing_data.multiple_imputation.ci_basis == "rubin"


def test_the_single_imputation_arm_shares_a_stream_with_no_other_arm():
    # Third occurrence of one defect family: b753c25 gave the tipping-point
    # fill's two arms their own streams, 79b8729 decoupled this arm from
    # `multiple_impute`'s -- and the arm still landed byte-for-byte on the
    # tipping point's treated fill, because `SeedSequence(seed).spawn(1)[0]`
    # and `SeedSequence(seed).spawn(2)[0]` are the same stream. On the
    # default path they collide outright: `compare_missing_data` defaults to
    # `seed=0` and `_estimate_at_fraction` falls back to 0 when
    # `config.random_seed is None`. A report that presents four arms as
    # independently fitted must not draw two of them from one stream.
    from cp_causal.sensitivity import _single_imputation_seed_sequence

    for seed in range(4):
        single = _single_imputation_seed_sequence(seed).generate_state(8)
        # `multiple_impute` reads the root sequence itself.
        assert not np.array_equal(single, np.random.SeedSequence(seed).generate_state(8))
        # `_estimate_at_fraction` spawns two children of it, one per arm.
        for child in np.random.SeedSequence(seed).spawn(2):
            assert not np.array_equal(single, child.generate_state(8))


def test_the_report_threads_its_seed_into_the_missing_data_arm():
    # `sensitivity_report` passed `seed` to `falsify` and not to
    # `compare_missing_data`, so every call at every seed produced the same
    # imputation arms -- a parameter that looked threaded and was not, which
    # makes the arm impossible to vary for a Monte Carlo check.
    dataset = _simulated()
    reduced, _ = complete_cases(dataset)
    estimate = all_estimators()["difference_in_means"]().fit(reduced, CONFIG)
    first = sensitivity_report(dataset, "difference_in_means", estimate, CONFIG, seed=0)
    second = sensitivity_report(dataset, "difference_in_means", estimate, CONFIG, seed=7)
    assert first.missing_data is not None
    assert second.missing_data is not None
    assert (
        first.missing_data.multiple_imputation.point
        != second.missing_data.multiple_imputation.point
    )


def test_a_complete_dataset_gets_a_reason_and_no_comparison():
    complete = build_scenario("D").generate(
        n_units=400, rng=np.random.default_rng(0)
    ).dataset
    estimate = all_estimators()["difference_in_means"]().fit(complete, CONFIG)
    report = sensitivity_report(complete, "difference_in_means", estimate, CONFIG)
    assert report.missing_data is None
    assert report.missing_data_reason is not None
    assert "nothing to impute" in report.missing_data_reason


def _dollar_dataset(*, scale=1.0, n=800, missing_fraction=0.4, seed=0):
    """A dataset carrying two collinear dollar-denominated covariates.

    `scale` divides both earnings columns, so `scale=1000.0` is the very same
    data recorded in thousands rather than dollars. Every draw upstream of that
    division is identical, so the two datasets differ in nothing but the unit
    the earnings happen to be written in.
    """
    rng = np.random.default_rng(seed)
    sd = 800.0
    treated = rng.binomial(1, 0.5, n).astype(float)
    indicator = rng.binomial(1, 0.4, n).astype(float)
    earlier = rng.normal(2.5 * sd, sd, n)
    later = 0.87 * earlier + 0.493 * rng.normal(2.5 * sd, sd, n)
    logit = (
        0.8 * treated
        + 1.5 * indicator
        + (4.0 / sd) * (earlier - 2.5 * sd)
        + (4.0 / sd) * (later - 2.5 * sd)
    )
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit))).astype(float)
    outcome[rng.permutation(n)[: int(n * missing_fraction)]] = np.nan
    frame = pd.DataFrame(
        {
            "unit_id": np.arange(n),
            "treated": treated,
            "indicator": indicator,
            "earnings_earlier": earlier / scale,
            "earnings_later": later / scale,
            "outcome": outcome,
        }
    )
    schema = CausalSchema(
        unit="unit_id",
        treatment="treated",
        outcome="outcome",
        covariates=("indicator", "earnings_earlier", "earnings_later"),
        outcome_type="binary",
    )
    return CausalDataset(frame=frame, schema=schema)


def test_imputed_values_do_not_depend_on_a_covariate_s_units():
    """Re-expressing a covariate in different units must not move the fills.

    Whether earnings are recorded in dollars or in thousands of dollars is a
    fact about the file, not about what the missing outcomes were. sklearn's
    LogisticRegression applies an L2 penalty by default (C=1.0), which falls on
    each coefficient in inverse proportion to its column's scale: a
    dollar-denominated column's coefficient is small enough to be effectively
    unpenalised, while a 0/1 indicator's is penalised in full. On raw columns
    the imputation model therefore reads the file's units as evidence, and the
    Bernoulli draws it feeds move with them.
    """
    in_dollars = impute_once(
        _dollar_dataset(), np.random.default_rng(11), draw_parameters=False
    )
    in_thousands = impute_once(
        _dollar_dataset(scale=1000.0), np.random.default_rng(11), draw_parameters=False
    )

    difference = np.abs(
        in_dollars.frame["outcome"].to_numpy()
        - in_thousands.frame["outcome"].to_numpy()
    )
    assert difference.max() < 1e-6


def test_drawn_parameter_imputations_do_not_depend_on_a_covariate_s_units():
    # The `draw_parameters=True` path fits on a bootstrap resample of the
    # observed rows rather than on all of them, so the invariance has to hold
    # against a training set that is not the whole sample -- which is also
    # where a scaler fitted across every row rather than across the fitted rows
    # would start leaking.
    in_dollars = impute_once(
        _dollar_dataset(), np.random.default_rng(11), draw_parameters=True
    )
    in_thousands = impute_once(
        _dollar_dataset(scale=1000.0), np.random.default_rng(11), draw_parameters=True
    )

    difference = np.abs(
        in_dollars.frame["outcome"].to_numpy()
        - in_thousands.frame["outcome"].to_numpy()
    )
    assert difference.max() < 1e-6


def test_pooled_imputation_estimate_does_not_depend_on_a_covariate_s_units():
    # The single-fill tests above read one draw each. This reads what a caller
    # actually receives: twenty completions pooled through Rubin's rules. The
    # point estimate is continuous, so it registers a shift in the imputation
    # model that individual 0/1 fills are too coarse to show.
    in_dollars = multiple_impute(
        _dollar_dataset(), "difference_in_means", CONFIG, seed=5
    )
    in_thousands = multiple_impute(
        _dollar_dataset(scale=1000.0), "difference_in_means", CONFIG, seed=5
    )

    assert in_dollars is not None and in_thousands is not None
    assert abs(in_dollars.point - in_thousands.point) < 1e-6
