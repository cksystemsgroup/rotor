"""Unit tests for the Pono subprocess bridge.

Two kinds of tests:

  1. Output-parser tests. Exercise `_parse_pono` directly against
     synthetic tool outputs for every shape Pono produces across
     versions. These run regardless of whether the binary is
     installed; they guard the parser against drift.
  2. Live subprocess tests. Skipped when `pono` is not on PATH.
     Exercise the full bridge end-to-end against a hand-built
     synthetic BTOR2 model.

Parser tests are the primary CI safety net — they catch
regressions without requiring any external install.
"""

from __future__ import annotations

import shutil

import pytest

from rotor.btor2.nodes import Model, Sort
from rotor.solvers.pono import Pono, _parse_pono

BV1 = Sort(1)
BV8 = Sort(8)


def _counter(bad_at: int) -> Model:
    m = Model()
    x = m.state(BV8, "x")
    m.init(x, m.const(BV8, 0))
    m.next(x, m.op("add", BV8, x, m.const(BV8, 1)))
    m.bad(m.op("ugte", BV1, x, m.const(BV8, bad_at)))
    return m


# ---- parser tests --------------------------------------------------------

def test_parser_classifies_unsat_as_proved_for_unbounded_engine() -> None:
    r = _parse_pono(stdout="Running IC3IA engine...\nunsat\n", stderr="",
                    returncode=0, name="pono-ic3ia", bound=10, elapsed=0.5,
                    unbounded=True)
    assert r.verdict == "proved"


def test_parser_classifies_unsat_as_unreachable_for_bmc_engine() -> None:
    r = _parse_pono(stdout="Running BMC at k=5\nunsat\n", stderr="",
                    returncode=0, name="pono-bmc", bound=5, elapsed=0.1,
                    unbounded=False)
    assert r.verdict == "unreachable"
    assert r.bound == 5


def test_parser_classifies_sat_as_reachable() -> None:
    r = _parse_pono(stdout="k=3\nsat\n", stderr="", returncode=0,
                    name="pono-bmc", bound=5, elapsed=0.1, unbounded=False)
    assert r.verdict == "reachable"


def test_parser_accepts_safe_unsafe_aliases() -> None:
    safe = _parse_pono("...\nsafe\n", "", 0, "pono-ic3ia", 0, 0.0, unbounded=True)
    unsafe = _parse_pono("...\nunsafe\n", "", 0, "pono-bmc", 5, 0.0, unbounded=False)
    assert safe.verdict == "proved"
    assert unsafe.verdict == "reachable"


def test_parser_accepts_true_false_aliases() -> None:
    t = _parse_pono("true\n", "", 0, "pono-ind", 0, 0.0, unbounded=True)
    f = _parse_pono("false\n", "", 0, "pono-bmc", 5, 0.0, unbounded=False)
    assert t.verdict == "proved"
    assert f.verdict == "reachable"


def test_parser_extracts_invariant_block_when_present() -> None:
    stdout = (
        "Running ic3ia\n"
        "unsat\n"
        "invariant: (and (<= x 5) (>= x 0))\n"
    )
    r = _parse_pono(stdout, "", 0, "pono-ic3ia", 0, 0.1, unbounded=True)
    assert r.verdict == "proved"
    assert r.invariant is not None
    assert "x 5" in r.invariant


def test_parser_returns_unknown_for_garbage() -> None:
    r = _parse_pono("nonsense\nweird garbage\n", "", 1,
                    "pono-bmc", 5, 0.1, unbounded=False)
    assert r.verdict == "unknown"
    assert "could not classify" in (r.reason or "")


# ---- bridge tests --------------------------------------------------------

def test_bridge_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError) as exc_info:
        Pono(mode="not-a-real-engine")
    assert "not-a-real-engine" in str(exc_info.value)


def test_bridge_name_includes_mode() -> None:
    assert Pono(mode="bmc").name == "pono-bmc"
    assert Pono(mode="ic3ia").name == "pono-ic3ia"
    assert Pono(mode="mbic3").name == "pono-mbic3"


def test_bridge_returns_unknown_when_binary_missing() -> None:
    r = Pono(mode="bmc", binary="definitely-not-installed-pono").check_reach(
        _counter(5), bound=3,
    )
    assert r.verdict == "unknown"
    assert "not found in PATH" in (r.reason or "")
    # Name threads through even in the error case so the portfolio
    # can report which racer gave up.
    assert r.backend == "pono-bmc"


def test_bridge_passes_bound_to_pono() -> None:
    # When the binary is missing we can't observe the argv list
    # directly, but the early-return verdict preserves `bound` in
    # the SolverResult so downstream portfolio logic sees the
    # right value.
    r = Pono(mode="bmc", binary="nope-not-here").check_reach(
        _counter(5), bound=42,
    )
    assert r.bound == 42


# ---- live tool test (skipped without pono) -------------------------------

@pytest.mark.skipif(shutil.which("pono") is None, reason="pono binary not installed")
def test_pono_live_on_simple_counter() -> None:
    # Bounded counter — bmc mode should find the bug at step 5.
    r = Pono(mode="bmc").check_reach(_counter(5), bound=10, timeout=30.0)
    assert r.verdict in ("reachable", "unknown")
