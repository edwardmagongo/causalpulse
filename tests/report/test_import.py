def test_the_report_package_may_import_the_result_types():
    import cp_report  # noqa: F401
    from cp_core.results import CausalValidityScore  # noqa: F401


def test_the_report_package_may_import_the_figures():
    # The report embeds figures, so this import is the whole reason the
    # layers contract places cp_report above cp_figures.
    import cp_report  # noqa: F401
    from cp_figures.guards import require_interval  # noqa: F401
