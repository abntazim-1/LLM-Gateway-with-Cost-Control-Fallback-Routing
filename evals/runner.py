"""Evaluation harness for the gateway's decision layer.

Every routing, screening and escalation decision in this gateway is a policy
judgement that can be wrong in ways tests don't catch: unit tests assert that
a specific input behaves a specific way, but they can't tell you whether the
policy is getting *better or worse overall*. This runner scores each policy
against a labelled dataset so that question has an answer.

Two properties matter:

* **Known gaps are tracked, not hidden.** Cases the current approach provably
  cannot handle are marked `known_gap` in the dataset. They are reported
  separately — excluded from the pass rate so they don't block CI, but counted
  and printed so the limitation stays visible and any improvement shows up
  immediately. A gap that silently disappears from the docs is how a system
  ends up claiming coverage it doesn't have.
* **False positives are graded too.** Every suite includes cases that must
  *not* fire. Screening that blocks everything scores perfectly on recall and
  is useless.

These suites are deterministic and need no model calls, so they run in CI.
Judging answer *correctness* — as opposed to answer *shape* — needs either a
judge model or real user feedback; see F13 in docs/AI_ML_FLAWS.md.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import yaml

from gateway.adapters.base import NormalizedMessage, NormalizedResponse
from gateway.policy.guardrails import GuardrailsPipeline, GuardrailViolationException
from gateway.policy.pii import PiiVault
from gateway.policy.router import Router

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    known_gap: bool
    expected: Any
    actual: Any
    note: str = ""


@dataclass
class SuiteResult:
    suite: str
    description: str
    results: List[CaseResult] = field(default_factory=list)

    @property
    def graded(self) -> List[CaseResult]:
        """Cases that count toward the score (known gaps excluded)."""
        return [r for r in self.results if not r.known_gap]

    @property
    def gaps(self) -> List[CaseResult]:
        return [r for r in self.results if r.known_gap]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.graded if r.passed)

    @property
    def total(self) -> int:
        return len(self.graded)

    @property
    def failures(self) -> List[CaseResult]:
        return [r for r in self.graded if not r.passed]

    @property
    def closed_gaps(self) -> List[CaseResult]:
        """Known gaps that now pass — the dataset and docs should be updated."""
        return [r for r in self.gaps if r.passed]

    @property
    def score(self) -> float:
        return (self.passed / self.total) if self.total else 1.0


# ── Policy adapters under evaluation ─────────────────────────────────────


def _eval_guardrails(case: Dict[str, Any]) -> tuple:
    pipeline = GuardrailsPipeline()
    try:
        pipeline.validate_messages([{"role": "user", "content": case["text"]}])
        actual = "allow"
    except GuardrailViolationException:
        actual = "block"
    return actual, case["expect"]


def _eval_pii(case: Dict[str, Any]) -> tuple:
    _, mapping = PiiVault().mask_text(case["text"])
    found = sorted({token.strip("[]").rsplit("_", 1)[0] for token in mapping})
    expected = sorted(case.get("expect_labels", []))
    return found, expected


def _make_router(strategy: str) -> Router:
    # Imported lazily so the harness stays importable without a live ledger.
    from gateway.ledger.store import LedgerStore
    from gateway.policy.circuit_breaker import CircuitBreakerRegistry

    return Router(
        adapters=[],
        circuit_registry=CircuitBreakerRegistry(ledger=LedgerStore(":memory:")),
        strategy=strategy,
    )


def _eval_routing(case: Dict[str, Any]) -> tuple:
    router = _make_router("complexity")
    is_complex = router._is_complex_request([{"role": "user", "content": case["text"]}])
    return ("complex" if is_complex else "simple"), case["expect"]


def _eval_adequacy(case: Dict[str, Any]) -> tuple:
    response = NormalizedResponse(
        id="eval",
        backend_id="eval",
        model="eval",
        messages=[NormalizedMessage(role="assistant", content=case["response"])],
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=0.0,
        latency_ms=1.0,
    )
    inadequate = Router._is_response_inadequate(
        response,
        [{"role": "user", "content": case["prompt"]}],
        finish_reason=case.get("finish_reason"),
    )
    return ("escalate" if inadequate else "keep"), case["expect"]


EVALUATORS: Dict[str, Callable[[Dict[str, Any]], tuple]] = {
    "guardrails": _eval_guardrails,
    "pii": _eval_pii,
    "routing": _eval_routing,
    "adequacy": _eval_adequacy,
}


# ── Runner ───────────────────────────────────────────────────────────────


def load_suites(dataset_dir: str = DATASET_DIR) -> List[Dict[str, Any]]:
    suites = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "*.yaml"))):
        with open(path, encoding="utf-8") as f:
            suites.append(yaml.safe_load(f))
    return suites


def run_suite(suite: Dict[str, Any]) -> SuiteResult:
    name = suite["suite"]
    evaluator = EVALUATORS[name]
    result = SuiteResult(suite=name, description=suite.get("description", ""))

    for case in suite["cases"]:
        actual, expected = evaluator(case)
        result.results.append(
            CaseResult(
                case_id=case["id"],
                passed=actual == expected,
                known_gap=bool(case.get("known_gap")),
                expected=expected,
                actual=actual,
                note=(case.get("note") or "").strip(),
            )
        )
    return result


def run_all(dataset_dir: str = DATASET_DIR) -> List[SuiteResult]:
    return [run_suite(s) for s in load_suites(dataset_dir)]


def format_report(results: List[SuiteResult]) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("GATEWAY POLICY EVALUATION")
    lines.append("=" * 72)

    for suite in results:
        pct = suite.score * 100
        lines.append("")
        lines.append(f"{suite.suite}  —  {suite.passed}/{suite.total} ({pct:.0f}%)")
        lines.append(f"  {suite.description}")

        for failure in suite.failures:
            lines.append(
                f"    FAIL  {failure.case_id}: "
                f"expected {failure.expected!r}, got {failure.actual!r}"
            )
        if suite.gaps:
            still_open = [g for g in suite.gaps if not g.passed]
            lines.append(
                f"    ({len(still_open)} known gap"
                f"{'s' if len(still_open) != 1 else ''} not counted)"
            )
            for gap in still_open:
                first_line = gap.note.splitlines()[0] if gap.note else ""
                lines.append(f"      gap   {gap.case_id}: {first_line}")
        for closed in suite.closed_gaps:
            lines.append(
                f"    CLOSED {closed.case_id} now passes — "
                f"remove `known_gap` and update docs/AI_ML_FLAWS.md"
            )

    graded_total = sum(s.total for s in results)
    graded_passed = sum(s.passed for s in results)
    open_gaps = sum(len([g for g in s.gaps if not g.passed]) for s in results)
    overall = (graded_passed / graded_total * 100) if graded_total else 100.0

    lines.append("")
    lines.append("-" * 72)
    lines.append(
        f"OVERALL  {graded_passed}/{graded_total} ({overall:.0f}%)  "
        f"· {open_gaps} known gap{'s' if open_gaps != 1 else ''} tracked"
    )
    lines.append("-" * 72)
    return "\n".join(lines)
