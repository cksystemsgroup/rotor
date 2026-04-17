"""Example: use IC3 to obtain an unbounded safety proof."""

from __future__ import annotations

import sys

from rotor.api import RotorAPI


def main(binary_path: str, function: str, invariant: str) -> int:
    api = RotorAPI(binary_path, default_solver="ic3", default_bound=1000)
    result = api.verify(
        function=function, invariant=invariant, bound=1000, unbounded=True
    )
    print(f"verdict: {result.verdict}  unbounded={result.unbounded}")
    if result.proof is not None:
        print("inductive invariant:")
        print(result.proof)
    if result.counterexample is not None:
        print("counterexample trace:")
        print(result.counterexample.as_markdown())
    return 0 if result.verdict != "unknown" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
