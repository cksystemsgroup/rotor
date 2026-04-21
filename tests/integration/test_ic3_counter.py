"""Phase 6.3 integration: Z3Spacer proves unbounded safety on a
real RISC-V fixture where Z3BMC can only answer `unreachable` up to
a given bound.

`tiny_mask` is a small arithmetic function whose dead branch (`return
x ^ 0xdeadbeef` guarded by `x > 10`) is unreachable — the input is
masked to 2 bits at entry, so the invariant `x in [0, 3]` forbids the
dead PC. BMC at any finite bound answers `unreachable` (safe up to k);
Spacer answers `proved` with an inductive invariant. This is the
BMC/IC3 contrast rotor's portfolio exists to exploit.

`bounded_counter` is a loop-carried version of the same pattern.
Spacer currently exceeds reasonable timeouts on it — the PDR engine
scales poorly in the number of state variables, and rotor's L0
encoding declares a state for every architectural register even when
the property only reads a handful. M8 (L2 SSA-BV with goal-directed
slicing) is the designed fix. Until then this file only asserts
BMC's `unreachable-within-bound` verdict on `bounded_counter`,
documenting the gap the IR layers will close.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rotor.binary import RISCVBinary
from rotor.ir.emitter import IdentityEmitter
from rotor.ir.spec import ReachSpec
from rotor.solvers import Portfolio, Z3BMC, Z3Spacer

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "counter.elf"
TINY_MASK_DEAD_PC = 0x1117c
BOUNDED_COUNTER_DEAD_PC = 0x111e0


@pytest.fixture(scope="module")
def tiny_mask_model():
    with RISCVBinary(FIXTURE) as b:
        yield IdentityEmitter(b).emit(
            ReachSpec(function="tiny_mask", target_pc=TINY_MASK_DEAD_PC)
        )


@pytest.mark.parametrize("bound", [5, 10, 20])
def test_bmc_says_unreachable_at_every_bound(tiny_mask_model, bound: int) -> None:
    r = Z3BMC().check_reach(tiny_mask_model, bound=bound, timeout=30.0)
    assert r.verdict == "unreachable"
    assert r.bound == bound


def test_spacer_proves_safety_with_invariant(tiny_mask_model) -> None:
    r = Z3Spacer().check_reach(tiny_mask_model, bound=0, timeout=60.0)
    assert r.verdict == "proved", f"expected proved, got {r.verdict} ({r.reason})"
    assert r.invariant is not None
    assert len(r.invariant) > 0


def test_portfolio_prefers_spacer_global_over_bmc_bounded(tiny_mask_model) -> None:
    # With both a BMC entry (bounded `unreachable`) and Spacer (unbounded
    # `proved`) in the race, the portfolio must return the globally-
    # conclusive verdict (`proved`) rather than the weaker `unreachable`.
    portfolio = (
        Portfolio()
        .add(Z3BMC(), bound=10, timeout=30.0)
        .add(Z3Spacer(), bound=0, timeout=60.0)
    )
    r = portfolio.check_reach(tiny_mask_model)
    assert r.verdict == "proved"
    assert r.backend == "z3-spacer"


def test_bounded_counter_bmc_unreachable_at_small_bound() -> None:
    # The loop-carried variant is too heavy for Spacer under the current
    # L0 encoding; M8 slicing is the fix. BMC still gives the expected
    # `unreachable` answer within a modest bound.
    with RISCVBinary(FIXTURE) as b:
        model = IdentityEmitter(b).emit(
            ReachSpec(function="bounded_counter", target_pc=BOUNDED_COUNTER_DEAD_PC)
        )
        r = Z3BMC().check_reach(model, bound=10, timeout=30.0)
        assert r.verdict == "unreachable"
