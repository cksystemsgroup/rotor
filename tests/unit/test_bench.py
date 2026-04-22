"""Unit tests for the rotor benchmark harness.

Uses synthetic in-memory BTOR2 models so tests don't depend on the
L0-equivalence corpus or the larger shootout run. The harness's
classification logic (SOLVED / UNSOLVED, verdict matching,
proved-covers-unreachable, PAR-2 scoring) is covered here; the
full multi-engine integration is in tests/integration/test_bench.py.
"""

from __future__ import annotations

from rotor.bench import (
    BenchEntry, RunOutcome, ShootoutResult, _cell, _is_solved,
    format_markdown, run_shootout,
)
from rotor.btor2.nodes import Model, Sort
from rotor.solvers.base import SolverResult
from rotor.solvers.portfolio import Portfolio

BV1 = Sort(1)
BV8 = Sort(8)


def _counter(bad_at: int) -> Model:
    m = Model()
    x = m.state(BV8, "x")
    m.init(x, m.const(BV8, 0))
    m.next(x, m.op("add", BV8, x, m.const(BV8, 1)))
    m.bad(m.op("ugte", BV1, x, m.const(BV8, bad_at)))
    return m


class _FakeEngine:
    """A synthetic backend that returns a prescribed verdict after
    an optional delay. Lets us test the harness without the cost
    of real solvers."""
    def __init__(self, name: str, verdict: str, elapsed: float = 0.0,
                 reason: str | None = None):
        self.name = name
        self._verdict = verdict
        self._elapsed = elapsed
        self._reason = reason

    def check_reach(self, model, bound, timeout=None):
        return SolverResult(
            verdict=self._verdict, bound=bound, step=0,
            elapsed=self._elapsed, backend=self.name, reason=self._reason,
        )


def test_is_solved_accepts_matching_verdict() -> None:
    assert _is_solved("reachable", "reachable")
    assert _is_solved("unreachable", "unreachable")
    assert _is_solved("proved", "proved")


def test_is_solved_treats_unknown_as_unsolved() -> None:
    assert not _is_solved("unknown", "reachable")
    assert not _is_solved("unknown", None)
    assert not _is_solved("error", "reachable")


def test_is_solved_accepts_proved_for_unreachable() -> None:
    # Unbounded engines produce strictly stronger verdicts; proved
    # must count when the expectation was the weaker unreachable.
    assert _is_solved("proved", "unreachable")


def test_is_solved_rejects_verdict_mismatch() -> None:
    assert not _is_solved("reachable", "unreachable")
    assert not _is_solved("unreachable", "reachable")


def test_is_solved_without_expectation_accepts_conclusive() -> None:
    assert _is_solved("reachable", None)
    assert _is_solved("proved", None)
    assert _is_solved("unreachable", None)


def test_run_shootout_classifies_outcomes() -> None:
    entry_sat = BenchEntry(
        name="sat-entry", model_factory=lambda: _counter(3),
        expected_verdict="reachable",
    )
    entry_unsat = BenchEntry(
        name="unsat-entry", model_factory=lambda: _counter(100),
        expected_verdict="unreachable",
    )
    engines = [
        ("fast-sat",  lambda: _FakeEngine("fast-sat",  "reachable",   elapsed=0.01)),
        ("slow-unsat", lambda: _FakeEngine("slow-unsat", "unreachable", elapsed=1.0)),
        ("stuck",     lambda: _FakeEngine("stuck",     "unknown",     elapsed=30.0)),
    ]
    result = run_shootout([entry_sat, entry_unsat], engines, bound=5, timeout=30.0)

    grouped = result.grouped()
    # fast-sat on sat-entry: solved.
    assert grouped[("sat-entry", "fast-sat")].solved
    # fast-sat on unsat-entry: verdict mismatch → unsolved.
    assert not grouped[("unsat-entry", "fast-sat")].solved
    # slow-unsat on sat-entry: verdict mismatch.
    assert not grouped[("sat-entry", "slow-unsat")].solved
    # stuck: never solved.
    assert not grouped[("sat-entry", "stuck")].solved


def test_par2_penalizes_unsolved_with_double_timeout() -> None:
    # One solved at 1.0s, one unsolved → PAR-2 = 1.0 + 2*30 = 61.0.
    entries = [
        BenchEntry("a", lambda: _counter(3), expected_verdict="reachable"),
        BenchEntry("b", lambda: _counter(3), expected_verdict="reachable"),
    ]
    engines = [
        ("mixed", lambda entry=None: (
            _FakeEngine("mixed", "reachable", elapsed=1.0)
        )),
    ]
    # Force "b" to be unsolved by making mixed return unknown on second call.
    # Simpler: use two engines.
    engines = [
        ("mixed-a", lambda: _FakeEngine("mixed-a", "reachable", elapsed=1.0)),
        ("mixed-b", lambda: _FakeEngine("mixed-b", "unknown",   elapsed=30.0)),
    ]
    result = run_shootout(entries, engines, bound=5, timeout=30.0)
    # mixed-a solved both entries at 1.0s each → PAR-2 = 2.0.
    assert result.par2("mixed-a") == 2.0
    # mixed-b never solved → PAR-2 = 2 * 2 * 30 = 120.
    assert result.par2("mixed-b") == 120.0


def test_format_markdown_renders_expected_sections() -> None:
    entries = [BenchEntry("only", lambda: _counter(3), expected_verdict="reachable")]
    engines = [
        ("solver1", lambda: _FakeEngine("solver1", "reachable", elapsed=0.01)),
        ("solver2", lambda: _FakeEngine("solver2", "unknown",   elapsed=30.0)),
    ]
    result = run_shootout(entries, engines, bound=5, timeout=30.0)
    md = format_markdown(result)
    assert "# Solver shootout" in md
    assert "solver1" in md and "solver2" in md
    assert "PAR-2" in md
    assert "Unique solves" in md


def test_format_markdown_unique_solves_named_correctly() -> None:
    entries = [
        BenchEntry("shared", lambda: _counter(3), expected_verdict="reachable"),
        BenchEntry("solo", lambda: _counter(3), expected_verdict="reachable"),
    ]
    # solver-a solves both; solver-b solves only 'solo'.
    # Nothing is uniquely solved by solver-a (shared also solvable by
    # solver-b? no — set up so only solver-a solves 'shared').
    engines = [
        ("solver-a", lambda: _FakeEngine("solver-a", "reachable", elapsed=0.01)),
        ("solver-b", lambda: _FakeEngine("solver-b", "unknown",   elapsed=30.0)),
    ]
    result = run_shootout(entries, engines, bound=5, timeout=30.0)
    md = format_markdown(result)
    # solver-a uniquely solves both entries (since solver-b never solves).
    assert "solver-a" in md
    assert "shared" in md and "solo" in md


def test_cell_compact_formatting() -> None:
    fast = RunOutcome("e", "z", "reachable", 0.0123, True)
    slow = RunOutcome("e", "z", "unreachable", 5.67, True)
    timeout = RunOutcome("e", "z", "unknown", 30.0, False)
    assert "ms" in _cell(fast)
    assert "5.67" in _cell(slow)
    assert "30" in _cell(timeout)
