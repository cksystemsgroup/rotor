"""Example: check that two implementations produce the same output.

Usage: ``python equivalence_check.py a.elf b.elf <function_name>``
"""

from __future__ import annotations

import sys

from rotor.api import RotorAPI


def main(binary_a: str, binary_b: str, function: str) -> int:
    api = RotorAPI(binary_a, default_solver="bitwuzla", default_bound=500)
    result = api.are_equivalent(
        other_binary=binary_b, function=function, bound=500, unbounded=False
    )
    print(f"verdict: {result.verdict}")
    if result.trace_a is not None:
        print("diverging trace (A):")
        print(result.trace_a.as_markdown())
    return 0 if result.verdict != "unknown" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
