"""Portfolio racer logic tested with synthetic fake backends so the
tests are fast and fully deterministic."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from rotor.btor2.nodes import Model
from rotor.solvers.base import SolverResult
from rotor.solvers.portfolio import Portfolio


@dataclass
class _Fake:
    name: str
    verdict: str
    delay: float = 0.0

    def check_reach(self, model: Model, bound: int, timeout=None) -> SolverResult:
        time.sleep(self.delay)
        return SolverResult(
            verdict=self.verdict,
            bound=bound,
            step=0 if self.verdict == "reachable" else None,
            elapsed=self.delay,
            backend=self.name,
        )


def test_reachable_wins_even_when_slower() -> None:
    # Unreachable at a bound only means "safe up to k", which is not a
    # global proof. The portfolio must wait for a reachable verdict and
    # prefer it over any faster unreachable.
    p = Portfolio()
    p.add(_Fake("slow-reach", "reachable", delay=0.05), bound=1)
    p.add(_Fake("fast-unreach", "unreachable", delay=0.01), bound=5)
    r = p.check_reach(Model())
    assert r.verdict == "reachable"
    assert r.backend == "slow-reach"


def test_unreachable_returned_when_no_reach_found() -> None:
    p = Portfolio()
    p.add(_Fake("u1", "unreachable"), bound=3)
    p.add(_Fake("u2", "unreachable"), bound=7)
    r = p.check_reach(Model())
    # Deepest unreachable bound wins as the strongest safe-up-to claim.
    assert r.verdict == "unreachable"
    assert r.bound == 7


def test_unknown_is_fallback() -> None:
    p = Portfolio()
    p.add(_Fake("u1", "unknown"), bound=1)
    p.add(_Fake("u2", "unknown"), bound=2)
    r = p.check_reach(Model())
    assert r.verdict == "unknown"


def test_reachable_preferred_over_concurrent_unknown() -> None:
    p = Portfolio()
    p.add(_Fake("rch", "reachable", delay=0.02), bound=1)
    p.add(_Fake("unk", "unknown", delay=0.0), bound=1)
    r = p.check_reach(Model())
    assert r.verdict == "reachable"


def test_empty_portfolio_errors() -> None:
    p = Portfolio()
    try:
        p.check_reach(Model())
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty portfolio")
