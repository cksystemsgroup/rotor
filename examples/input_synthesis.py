"""Example: synthesize an input that produces a target output condition."""

from __future__ import annotations

import sys

from rotor.api import RotorAPI


def main(binary_path: str, function: str, condition: str) -> int:
    api = RotorAPI(binary_path, default_solver="bitwuzla", default_bound=500)
    result = api.find_input(function=function, output_condition=condition)
    print(f"verdict: {result.verdict}")
    if result.input_bytes is not None:
        print(f"input ({len(result.input_bytes)} bytes): {result.input_bytes!r}")
    if result.trace is not None:
        print(result.trace.as_markdown())
    return 0 if result.verdict != "unknown" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
