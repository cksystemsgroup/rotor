"""Tests for solver backends that don't require external binaries."""

from __future__ import annotations

from rotor.solvers import make_solver
from rotor.solvers.base import CheckResult, SolverBackend
from rotor.solvers.portfolio import PortfolioSolver


class _StubSolver(SolverBackend):
    name = "stub"

    def __init__(self, verdict: str = "unknown", elapsed: float = 0.0) -> None:
        self._verdict = verdict
        self._elapsed = elapsed

    def check(self, btor2: str, bound: int) -> CheckResult:  # noqa: D401
        return CheckResult(
            verdict=self._verdict, solver=self.name, elapsed=self._elapsed
        )


def test_make_solver_dispatch() -> None:
    solver = make_solver("btormc", binary="nope")
    assert solver.name == "btormc"


def test_check_result_is_conclusive() -> None:
    assert CheckResult(verdict="sat").is_conclusive()
    assert CheckResult(verdict="unsat").is_conclusive()
    assert not CheckResult(verdict="unknown").is_conclusive()


def test_portfolio_empty() -> None:
    result = PortfolioSolver().check("", 10)
    assert result.verdict == "unknown"
    assert "no backends" in result.stderr


def test_portfolio_first_conclusive() -> None:
    unsat = _StubSolver("unsat")
    unknown = _StubSolver("unknown")
    pool = PortfolioSolver(backends=[unknown, unsat])
    result = pool.check("", 10)
    assert result.verdict == "unsat"


def test_portfolio_all_unknown() -> None:
    pool = PortfolioSolver(backends=[_StubSolver("unknown"), _StubSolver("unknown")])
    result = pool.check("", 10)
    # Portfolio only returns 'unknown' when no backend is conclusive; since
    # both return unknown via the as_completed iteration, we still return
    # the last one processed or aggregate — accept either outcome.
    assert result.verdict in ("unknown",)


def test_btormc_missing_binary() -> None:
    solver = make_solver("btormc", binary="definitely-not-installed-xyz")
    result = solver.check("1 sort bitvec 1\n", 1)
    assert result.verdict == "unknown"
    assert "not found" in result.stderr
