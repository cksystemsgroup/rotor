"""Integration: CVC5 BMC earns its place in the portfolio.

Exercises the CVC5 bridge end-to-end on real rotor fixtures and
verifies the portfolio correctly consumes it alongside the other
in-process engines.

Module skips when the `cvc5` pip package is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cvc5")

from rotor import RotorAPI, Z3BMC                      # noqa: E402
from rotor.binary import RISCVBinary                   # noqa: E402
from rotor.ir.emitter import IdentityEmitter           # noqa: E402
from rotor.ir.spec import ReachSpec                    # noqa: E402
from rotor.solvers.cvc5bmc import CVC5BMC              # noqa: E402
from rotor.solvers.portfolio import Portfolio          # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_cvc5_matches_z3_verdict_on_add2() -> None:
    with RISCVBinary(FIXTURES / "add2.elf") as b:
        model = IdentityEmitter(b).emit(
            ReachSpec(function="add2", target_pc=b.function("add2").start + 4)
        )
        z3 = Z3BMC().check_reach(model, bound=2, timeout=10.0)
        cv = CVC5BMC().check_reach(model, bound=2, timeout=10.0)
    assert z3.verdict == cv.verdict == "reachable"
    assert z3.step == cv.step == 1


def test_cvc5_handles_memory_model_on_memops() -> None:
    with RISCVBinary(FIXTURES / "memops.elf") as b:
        fn = b.function("load_sum")
        model = IdentityEmitter(b).emit(
            ReachSpec(function="load_sum", target_pc=fn.start + 0xC)
        )
        r = CVC5BMC().check_reach(model, bound=3, timeout=30.0)
    assert r.verdict == "reachable"
    assert r.step == 3


def test_cvc5_handles_rv64m_fixture() -> None:
    # mult.elf uses mul / divuw / mulw; exercises the M-extension
    # kinds in _apply_op.
    with RISCVBinary(FIXTURES / "mult.elf") as b:
        fn = b.function("divmod")
        model = IdentityEmitter(b).emit(
            ReachSpec(function="divmod", target_pc=fn.start + 0x10)
        )
        r = CVC5BMC().check_reach(model, bound=5, timeout=10.0)
    assert r.verdict == "reachable"
    assert r.step == 4


def test_portfolio_race_includes_cvc5() -> None:
    with RISCVBinary(FIXTURES / "add2.elf") as b:
        model = IdentityEmitter(b).emit(
            ReachSpec(function="add2", target_pc=b.function("add2").start + 4)
        )
        portfolio = (
            Portfolio()
            .add(Z3BMC(),   bound=4, timeout=10.0)
            .add(CVC5BMC(), bound=4, timeout=10.0)
        )
        r = portfolio.check_reach(model)
    assert r.verdict == "reachable"
    assert r.backend in ("z3-bmc", "cvc5-bmc")


def test_default_portfolio_picks_up_cvc5() -> None:
    # When cvc5 is importable, default_portfolio must include it.
    from rotor.solvers import default_portfolio, CVC5BMC
    p = default_portfolio()
    types = [type(e.backend) for e in p.entries]
    assert CVC5BMC in types
