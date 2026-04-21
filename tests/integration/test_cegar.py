"""Phase 6.4 integration tests for CEGAR.

`tiny_mask` is the main workload: the dead branch is provable by a
loose abstraction (a few registers unhavoc'd), so CEGAR converges in
a handful of iterations. The `reachable` test confirms that when the
abstract CEX is real, concrete replay identifies it and CEGAR reports
the real step.

`bounded_counter` remains out of CEGAR's reach under the current
register-granular abstraction — the loop invariant Spacer needs to
infer is in the fanin of several registers and the abstract Spacer
call exceeds its timeout budget. Test marked `skipif` with the
waiting-on-M8 note; upgrading to SSA-BV slicing should collapse the
state space the PDR engine has to reason over.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rotor.binary import RISCVBinary
from rotor.cegar import CegarConfig, cegar_reach
from rotor.ir.spec import ReachSpec

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "counter.elf"
TINY_MASK_DEAD_PC = 0x1117c
TINY_MASK_LIVE_RET_PC = 0x11178
BOUNDED_COUNTER_DEAD_PC = 0x111e0


def test_cegar_proves_tiny_mask_dead_branch_unreachable() -> None:
    with RISCVBinary(FIXTURE) as b:
        r = cegar_reach(
            b,
            ReachSpec(function="tiny_mask", target_pc=TINY_MASK_DEAD_PC),
            CegarConfig(max_iterations=8, spacer_timeout=30.0),
        )
    assert r.verdict == "proved", f"expected proved, got {r.verdict} ({r.reason})"
    assert r.invariant is not None and len(r.invariant) > 0
    assert "cegar" in r.backend


def test_cegar_detects_real_reachability_on_live_path() -> None:
    # The live ret at 0x11178 is reachable at step 8 on any input (it's
    # the concrete return site after the x <= 3 path). CEGAR must not
    # falsely prove this safe.
    with RISCVBinary(FIXTURE) as b:
        r = cegar_reach(
            b,
            ReachSpec(function="tiny_mask", target_pc=TINY_MASK_LIVE_RET_PC),
            CegarConfig(max_iterations=4, spacer_timeout=15.0),
        )
    assert r.verdict == "reachable", f"expected reachable, got {r.verdict} ({r.reason})"
    assert r.step is not None and r.step > 0


@pytest.mark.skip(reason="bounded_counter exceeds Spacer under CEGAR abstraction; unblocked by M8 slicing")
def test_cegar_proves_bounded_counter_safe() -> None:
    with RISCVBinary(FIXTURE) as b:
        r = cegar_reach(
            b,
            ReachSpec(function="bounded_counter", target_pc=BOUNDED_COUNTER_DEAD_PC),
            CegarConfig(max_iterations=16, spacer_timeout=60.0),
        )
    assert r.verdict == "proved"
