"""Track C integration: non-leaf reach via `include_fns`.

`nonleaf.elf` contains:

  double_square(x)  —  stack setup + `jal ra, square` + slliw +
                       stack teardown + ret
  square(x)         —  mulw a0, a0, a0 + ret

Without `include_fns`, the `jal` target is outside the analyzed
set so PC gets stuck after the call — double_square's ret is
unreachable. With `include_fns=["square"]`, rotor decodes both
functions into one dispatch: `jal` lands on square.start, square's
ret restores ra to double_square's post-jal PC, and execution
resumes. This exercises the full Track C machinery: cycle-0
ra-constraint (so the jal's intra-set ra write isn't rejected),
multi-function dispatch, and scope-aware ret enumeration.
"""

from __future__ import annotations

import io
from pathlib import Path

from rotor import RotorAPI
from rotor.cli import main

FIXTURE = str((Path(__file__).resolve().parents[1] / "fixtures" / "nonleaf.elf"))
DOUBLE_SQUARE_RET_OFFSET = 0x10                    # fn.start + 0x10 is the ret


def _run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


# ---- API ----

def test_api_can_reach_ret_requires_include_fns() -> None:
    # Without include_fns: square is outside the analyzed set, so the
    # jal's target PC is unreachable and the post-call ret never
    # fires.
    with RotorAPI(FIXTURE) as api:
        fn = api.binary.function("double_square")
        target = fn.start + DOUBLE_SQUARE_RET_OFFSET
        r = api.can_reach(function="double_square", target_pc=target, bound=20)
    assert r.verdict == "unreachable"


def test_api_can_reach_ret_with_include_fns() -> None:
    with RotorAPI(FIXTURE) as api:
        fn = api.binary.function("double_square")
        target = fn.start + DOUBLE_SQUARE_RET_OFFSET
        r = api.can_reach(
            function="double_square", target_pc=target, bound=20,
            include_fns=["square"],
        )
    assert r.verdict == "reachable"
    # Initial ra must be outside the analyzed set (both functions).
    ra = r.initial_regs.get("x1", 0)
    fn_a = api.binary.function("double_square")
    fn_b = api.binary.function("square")
    assert not (fn_a.start <= ra < fn_a.end)
    assert not (fn_b.start <= ra < fn_b.end)


def test_api_intra_set_jal_target_is_reachable() -> None:
    # Track C's specific claim: a `jal` whose target lies inside the
    # analyzed set lands on a real instruction, not on a "stuck" PC.
    # The first instruction of `square` (mulw) must be reachable from
    # `double_square`'s entry when `include_fns=["square"]`.
    with RotorAPI(FIXTURE) as api:
        square = api.binary.function("square")
        r = api.can_reach(
            function="double_square", target_pc=square.start, bound=6,
            include_fns=["square"],
        )
    assert r.verdict == "reachable"


# ---- CLI ----

def test_cli_reach_with_include_fn_flag() -> None:
    code, out, err = _run(
        "reach", FIXTURE,
        "--function", "double_square",
        "--target", "0x11168",
        "--bound", "20",
        "--include-fn", "square",
    )
    assert code == 1                              # reachable → EXIT_FOUND
    assert "verdict  : reachable" in out


def test_cli_reach_without_include_is_unreachable() -> None:
    code, out, err = _run(
        "reach", FIXTURE,
        "--function", "double_square",
        "--target", "0x11168",
        "--bound", "20",
    )
    assert code == 0                              # unreachable → EXIT_OK
    assert "verdict  : unreachable" in out
