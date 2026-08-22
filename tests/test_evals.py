"""Runs the policy evaluation suites as part of the normal test run.

The harness exists so policy quality is *measured* rather than assumed (F13).
Wiring it into pytest means a change that degrades routing, screening or
escalation fails the build the same way a broken unit test would — the suites
are deterministic and need no model calls, so they are safe in CI.
"""

import pytest

from evals.runner import format_report, load_suites, run_all, run_suite


@pytest.mark.parametrize("suite", load_suites(), ids=lambda s: s["suite"])
def test_policy_suite_has_no_regressions(suite):
    result = run_suite(suite)

    failures = [
        f"{f.case_id}: expected {f.expected!r}, got {f.actual!r}"
        for f in result.failures
    ]
    assert (
        not failures
    ), f"{result.suite} regressed ({result.passed}/{result.total}):\n  " + "\n  ".join(
        failures
    )


def test_known_gaps_are_still_gaps():
    """Fails when a documented limitation is quietly fixed.

    A closed gap is good news, but it has to be recorded — otherwise the
    dataset and docs keep describing a weakness that no longer exists, which
    is how a project ends up misrepresenting its own coverage.
    """
    closed = [
        (suite.suite, case.case_id) for suite in run_all() for case in suite.closed_gaps
    ]
    assert not closed, (
        "These known gaps now pass — remove `known_gap` from the dataset and "
        f"update docs/AI_ML_FLAWS.md: {closed}"
    )


def test_every_suite_grades_false_positives():
    """Screening that fires on everything scores perfectly on recall.

    Each suite must therefore contain cases that are expected *not* to
    trigger, or its score is not meaningful.
    """
    negative_expectations = {"allow", "simple", "keep"}
    for suite in load_suites():
        has_negative = any(
            case.get("expect") in negative_expectations
            or case.get("expect_labels") == []
            for case in suite["cases"]
        )
        assert has_negative, (
            f"suite {suite['suite']!r} has no negative cases, so its score "
            "cannot distinguish precision from over-triggering"
        )


def test_report_renders():
    assert "OVERALL" in format_report(run_all())
