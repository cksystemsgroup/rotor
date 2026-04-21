"""Phase 6.4 integration tests for CEGAR.

`tiny_mask` is the main workload: the dead branch is provable by a
loose abstraction (a few registers unhavoc'd), so CEGAR converges in
a handful of iterations. The `reachable` test confirms that when the
abstract CEX is real, concrete replay identifies it and CEGAR reports
the real step.

`bounded_counter` remains beyond Spacer even after M8's L2 slicing
collapsed the PDR state to {pc, ra, a0, a1}. The hurdle is loop-
invariant inference over the bitvector loop body, which neither
slicing nor register-level abstraction can shrink. Test stays
skipped with a note pointing at BVDD (M9), external IC3 bridges,
or predicate abstraction as the next candidates.
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


@pytest.mark.skip(reason=(
    "Spacer exceeds timeout on bounded_counter even with M8's L2 "
    "slicing (pc + 3 regs). The remaining hurdle is loop-invariant "
    "inference over bitvector arithmetic; slicing removes state "
    "dimensions but cannot shrink the loop body. Candidate "
    "follow-ups: BVDD frame representation (M9), external IC3 "
    "bridges (rIC3), or predicate abstraction in CEGAR."
))
def test_cegar_proves_bounded_counter_safe() -> None:
    with RISCVBinary(FIXTURE) as b:
        r = cegar_reach(
            b,
            ReachSpec(function="bounded_counter", target_pc=BOUNDED_COUNTER_DEAD_PC),
            CegarConfig(max_iterations=16, spacer_timeout=60.0),
        )
    assert r.verdict == "proved"
