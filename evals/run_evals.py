"""CLI entry point for the policy evaluation harness.

    python evals/run_evals.py            # report, exit non-zero on any failure
    python evals/run_evals.py --min 0.9  # allow up to 10% failures per suite

Exits non-zero when a suite falls below the threshold, so it can gate CI.
Known gaps never fail the run — they are reported and counted separately.
"""

import argparse
import sys

from evals.runner import format_report, run_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min",
        type=float,
        default=1.0,
        help="Minimum per-suite pass rate, 0-1 (default 1.0).",
    )
    args = parser.parse_args()

    results = run_all()
    print(format_report(results))

    below = [s for s in results if s.score < args.min]
    if below:
        print()
        for suite in below:
            print(
                f"FAILED: {suite.suite} scored {suite.score:.0%}, "
                f"below the {args.min:.0%} threshold"
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
