"""Example: find an input that triggers a buffer overflow.

Given a RISC-V ELF containing a function ``copy`` that writes user input
into a fixed-size buffer, ask the API whether an input exists that reaches a
known bad PC. On SAT, print the source-level trace.
"""

from __future__ import annotations

import sys

from rotor.api import RotorAPI


def main(binary_path: str) -> int:
    api = RotorAPI(binary_path, default_solver="bitwuzla", default_bound=200)
    try:
        result = api.can_reach(
            function="copy",
            condition="pc == 0x10140",  # example bad PC
            bound=200,
        )
    except NotImplementedError as err:
        print(f"(condition compiler not configured: {err})", file=sys.stderr)
        return 2
    print(f"verdict: {result.verdict}")
    if result.trace is not None:
        print(result.trace.as_markdown())
    return 0 if result.verdict != "unknown" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1]))
