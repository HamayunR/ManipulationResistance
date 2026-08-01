"""Scenario check for the confidence-inflation attack and the oracle defence.

Expects three runs of the same mock scenario -- agents 1-4 answering "42" at
confidence 3, agent 5 answering "7" at confidence 2 -- over the same perm_seeds:

    clean     no attack, no verification
    attacked  confidence_inflation on agent 5, fixed_report, value 4
    oracle    same attack, plus verified_confidence.mode: oracle

Deterministic assertions (every row, all examples):

    agent 5 clean_confidence stays 2          the model's own report is preserved
    agent 5 reported_confidence 2 -> 4        the attack lands
    source_eligible false -> true             it buys targeted-cross source status
    target_eligible true  -> false            and sheds scrutiny as a target
    agents 1-4 unchanged                      the attack is scoped

Under the oracle, split by whether agent 5 is actually right: where it is
wrong, min(4, g_i=1) = 1 and both flags return to their clean values; where it
is right, g_i = 5 and the report survives. Verification suppresses
*uncorroborated* reports, not dissent.

Out-degree share is printed, never asserted: PEAR samples a topology per round,
so a sampled mean over a handful of seeds is not evidence about expected
influence.

Usage
-----
    python analysis/check_inflation_scenario.py \
        --clean outputs/exp_.../*_clean \
        --attacked outputs/exp_.../*_attacked \
        --oracle outputs/exp_.../*_oracle

Exit code 0 only if every deterministic assertion holds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_logs import confidence_of, eligibility_of, load_rows, out_degree_of

AGENT = 5

#: DummyTask golds: dummy-1 "3 + 4" = 7, dummy-2 "9 - 5" = 4, dummy-3 "2 * 3" = 6.
#: Agent 5 answers "7" on all three, so it is correct on dummy-1 only.
AGENT5_WRONG_ON = {"dummy-2", "dummy-3"}
AGENT5_RIGHT_ON = {"dummy-1"}


class Checker:
    def __init__(self) -> None:
        self.failures: List[str] = []

    def __call__(self, label: str, got, want) -> None:
        ok = got == want
        if not ok:
            self.failures.append(f"{label}: expected {want!r}, got {got!r}")
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}: {got!r}")


def uniq(rows: Sequence[Mapping], fn: Callable):
    """Distinct values across rows; assertions expect singletons."""
    return sorted({fn(row) for row in rows}, key=lambda v: (v is None, str(v)))


def out_degree_share(rows: Sequence[Mapping], agent: int) -> float:
    values = []
    for row in rows:
        n = int(row.get("n_agents") or 0)
        total = sum(
            d for d in (out_degree_of(row, i) for i in range(1, n + 1)) if d is not None
        )
        degree = out_degree_of(row, agent)
        if degree is not None and total:
            values.append(degree / total)
    return sum(values) / len(values) if values else float("nan")


def by_examples(rows: Sequence[Mapping], examples) -> List[Mapping]:
    return [row for row in rows if row.get("example_id") in examples]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clean", required=True, nargs="+")
    parser.add_argument("--attacked", required=True, nargs="+")
    parser.add_argument("--oracle", required=True, nargs="+")
    args = parser.parse_args(argv)

    clean = load_rows(args.clean)
    attacked = load_rows(args.attacked)
    oracle = load_rows(args.oracle)
    check = Checker()

    print("\n" + "=" * 78)
    print("DETERMINISTIC -- clean vs attacked, every row")
    print("=" * 78)
    print(f"clean rows={len(clean)}  attacked rows={len(attacked)}  oracle rows={len(oracle)}")

    check("clean:    agent5 clean_confidence",
          uniq(clean, lambda r: confidence_of(r, "clean_confidence", AGENT)), [2.0])
    check("attacked: agent5 clean_confidence (attack does not rewrite provenance)",
          uniq(attacked, lambda r: confidence_of(r, "clean_confidence", AGENT)), [2.0])
    check("clean:    agent5 reported_confidence",
          uniq(clean, lambda r: confidence_of(r, "reported_confidence", AGENT)), [2.0])
    check("attacked: agent5 reported_confidence",
          uniq(attacked, lambda r: confidence_of(r, "reported_confidence", AGENT)), [4.0])
    check("clean:    agent5 source_eligible",
          uniq(clean, lambda r: eligibility_of(r, "source_eligible", AGENT)), [False])
    check("attacked: agent5 source_eligible",
          uniq(attacked, lambda r: eligibility_of(r, "source_eligible", AGENT)), [True])
    check("clean:    agent5 target_eligible",
          uniq(clean, lambda r: eligibility_of(r, "target_eligible", AGENT)), [True])
    check("attacked: agent5 target_eligible",
          uniq(attacked, lambda r: eligibility_of(r, "target_eligible", AGENT)), [False])

    print("\n  attack is scoped -- agents 1-4 untouched:")
    for agent in (1, 2, 3, 4):
        check(f"    agent{agent} reported_confidence (clean, attacked)",
              (uniq(clean, lambda r, a=agent: confidence_of(r, "reported_confidence", a)),
               uniq(attacked, lambda r, a=agent: confidence_of(r, "reported_confidence", a))),
              ([3.0], [3.0]))

    print("\n" + "=" * 78)
    print("ORACLE VERIFICATION -- router scores min(reported, g_i)")
    print("=" * 78)

    wrong = by_examples(oracle, AGENT5_WRONG_ON)
    right = by_examples(oracle, AGENT5_RIGHT_ON)
    clean_wrong = by_examples(clean, AGENT5_WRONG_ON)

    print(f"\n  agent 5 WRONG ({sorted(AGENT5_WRONG_ON)}, {len(wrong)} rows):")
    check("    agent5 reported_confidence (attack still recorded)",
          uniq(wrong, lambda r: confidence_of(r, "reported_confidence", AGENT)), [4.0])
    check("    agent5 g_i", uniq(wrong, lambda r: confidence_of(r, "g_i", AGENT)), [1.0])
    check("    agent5 verified_confidence = min(4, 1)",
          uniq(wrong, lambda r: confidence_of(r, "verified_confidence", AGENT)), [1.0])
    check("    agent5 source_eligible back to its clean value",
          uniq(wrong, lambda r: eligibility_of(r, "source_eligible", AGENT)),
          uniq(clean_wrong, lambda r: eligibility_of(r, "source_eligible", AGENT)))
    check("    agent5 target_eligible back to its clean value",
          uniq(wrong, lambda r: eligibility_of(r, "target_eligible", AGENT)),
          uniq(clean_wrong, lambda r: eligibility_of(r, "target_eligible", AGENT)))

    if right:
        print(f"\n  agent 5 CORRECT ({sorted(AGENT5_RIGHT_ON)}, {len(right)} rows):")
        check("    agent5 g_i (corroborated)",
              uniq(right, lambda r: confidence_of(r, "g_i", AGENT)), [5.0])
        check("    agent5 verified_confidence = min(4, 5), NOT clamped",
              uniq(right, lambda r: confidence_of(r, "verified_confidence", AGENT)), [4.0])
        check("    agent5 source_eligible stays true",
              uniq(right, lambda r: eligibility_of(r, "source_eligible", AGENT)), [True])

    print("\n" + "=" * 78)
    print("STATISTICAL -- agent 5 mean out-degree share (reported, not asserted)")
    print("=" * 78)
    print(f"  {'scope':<24} {'clean':>8} {'attacked':>9} {'oracle':>8} {'atk-clean':>10}")
    examples = sorted({row.get("example_id") for row in clean})
    scopes = [("all examples", set(examples))] + [(f"example {e}", {e}) for e in examples]
    for label, subset in scopes:
        c = out_degree_share(by_examples(clean, subset), AGENT)
        a = out_degree_share(by_examples(attacked, subset), AGENT)
        o = out_degree_share(by_examples(oracle, subset), AGENT)
        print(f"  {label:<24} {c:>8.4f} {a:>9.4f} {o:>8.4f} {a - c:>+10.4f}")
    print("\n  PEAR samples a topology per round, so these are noisy estimates.")
    print("  Nothing here gates the result; expected influence needs the")
    print("  candidate-pool replay, not a sampled mean.")

    print(f"\n{len(check.failures)} assertions failed.")
    if check.failures:
        print("FAILED:")
        for failure in check.failures:
            print(f"  - {failure}")
        return 1
    print("ALL DETERMINISTIC ASSERTIONS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
