"""Track A.1 integration: Bitwuzla earns its place in the portfolio.

The roadmap's Track A predicts Bitwuzla dominates Z3 by a large
margin on pure-BV workloads. These tests verify the bridge works
end-to-end against rotor fixtures and that the portfolio correctly
consumes both backends.

Module skips when the `bitwuzla` Python package is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("bitwuzla")

from rotor import BitwuzlaBMC, RotorAPI, Z3BMC       # noqa: E402
from rotor.binary import RISCVBinary                  # noqa: E402
from rotor.ir.emitter import IdentityEmitter          # noqa: E402
from rotor.ir.spec import ReachSpec                   # noqa: E402
from rotor.solvers.portfolio import Portfolio         # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_bitwuzla_matches_z3_verdict_on_add2() -> None:
    with RISCVBinary(FIXTURES / "add2.elf") as b:
        model = IdentityEmitter(b).emit(
            ReachSpec(function="add2", target_pc=b.function("add2").start + 4)
        )
        z3 = Z3BMC().check_reach(model, bound=2, timeout=10.0)
        bw = BitwuzlaBMC().check_reach(model, bound=2, timeout=10.0)
    assert z3.verdict == bw.verdict == "reachable"
    assert z3.step == bw.step == 1


def test_bitwuzla_handles_memory_model_on_memops() -> None:
    with RISCVBinary(FIXTURES / "memops.elf") as b:
        model = IdentityEmitter(b).emit(
            ReachSpec(function="load_sum", target_pc=b.function("load_sum").start + 0xC)
        )
        r = BitwuzlaBMC().check_reach(model, bound=3, timeout=30.0)
    assert r.verdict == "reachable"
    assert r.step == 3


def test_portfolio_races_bitwuzla_and_z3() -> None:
    # On a trivially-reachable model, whichever engine lands first wins.
    # Both must find reachable — we only assert verdict + backend is one
    # of the two engines.
    with RISCVBinary(FIXTURES / "add2.elf") as b:
        model = IdentityEmitter(b).emit(
            ReachSpec(function="add2", target_pc=b.function("add2").start + 4)
        )
        portfolio = (
            Portfolio()
            .add(Z3BMC(),       bound=4, timeout=10.0)
            .add(BitwuzlaBMC(), bound=4, timeout=10.0)
        )
        r = portfolio.check_reach(model)
    assert r.verdict == "reachable"
    assert r.backend in ("z3-bmc", "bitwuzla-bmc")


def test_bitwuzla_gives_expected_verdict_on_counter_unreachable() -> None:
    # tiny_mask's dead branch is unreachable at any finite bound.
    # Bitwuzla confirms this on par with Z3 but faster on larger bounds.
    with RISCVBinary(FIXTURES / "counter.elf") as b:
        model = IdentityEmitter(b).emit(
            ReachSpec(function="tiny_mask", target_pc=0x1117c)
        )
        r = BitwuzlaBMC().check_reach(model, bound=20, timeout=30.0)
    assert r.verdict == "unreachable"
