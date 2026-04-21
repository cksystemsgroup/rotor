"""Unit tests for the subprocess backend bridges (rIC3, BtorMC).

Two kinds of tests:

  1. Output-parser tests. Exercise `_parse_ric3` / `_parse_btormc`
     directly against synthetic tool outputs. These run regardless
     of whether the tools are installed; they guard the parser
     against drift in the verdict-extraction logic.

  2. Live subprocess tests. Skipped when the tool binary is not on
     PATH. These exercise the full bridge end-to-end against a
     handcrafted synthetic BTOR2 model.

The parser tests are the primary CI safety net — they catch
regressions without requiring any external install.
"""

from __future__ import annotations

import shutil

import pytest

from rotor.btor2.nodes import Model, Sort
from rotor.solvers.btormc import BtorMC, _parse_btormc
from rotor.solvers.ric3 import Ric3, _parse_ric3

BV1 = Sort(1)
BV8 = Sort(8)


def _counter(bad_at: int) -> Model:
    m = Model()
    x = m.state(BV8, "x")
    m.init(x, m.const(BV8, 0))
    m.next(x, m.op("add", BV8, x, m.const(BV8, 1)))
    m.bad(m.op("ugte", BV1, x, m.const(BV8, bad_at)))
    return m


# ---- rIC3 parser tests ------------------------------------------------------

def test_ric3_parser_classifies_unsat_as_proved() -> None:
    r = _parse_ric3(stdout="... some log ...\nunsat\n", stderr="", returncode=0,
                    name="rIC3", elapsed=0.1)
    assert r.verdict == "proved"
    assert r.invariant is not None


def test_ric3_parser_classifies_sat_as_reachable() -> None:
    r = _parse_ric3(stdout="searching...\nsat\n", stderr="", returncode=0,
                    name="rIC3", elapsed=0.1)
    assert r.verdict == "reachable"


def test_ric3_parser_accepts_true_false_tokens() -> None:
    # Older rIC3 versions print 'true' / 'false'.
    safe = _parse_ric3("true\n", "", 0, "rIC3", 0.0)
    buggy = _parse_ric3("false\n", "", 0, "rIC3", 0.0)
    assert safe.verdict == "proved"
    assert buggy.verdict == "reachable"


def test_ric3_parser_accepts_safe_unsafe_tokens() -> None:
    safe = _parse_ric3("...\nsafe\n", "", 0, "rIC3", 0.0)
    buggy = _parse_ric3("...\nunsafe\n", "", 0, "rIC3", 0.0)
    assert safe.verdict == "proved"
    assert buggy.verdict == "reachable"


def test_ric3_parser_returns_unknown_for_garbage() -> None:
    r = _parse_ric3("no verdict here\nweird garbage\n", "", 1, "rIC3", 0.0)
    assert r.verdict == "unknown"
    assert "could not classify" in (r.reason or "")


def test_ric3_bridge_returns_unknown_when_binary_missing() -> None:
    r = Ric3(binary="definitely-not-installed-ric3").check_reach(_counter(5))
    assert r.verdict == "unknown"
    assert "not found in PATH" in (r.reason or "")


# ---- BtorMC parser tests ----------------------------------------------------

def test_btormc_parser_classifies_unsat_as_proved() -> None:
    r = _parse_btormc("... k-ind log ...\nunsat\n", "", 0, "btormc", 10, 0.1)
    assert r.verdict == "proved"
    assert r.bound == 10


def test_btormc_parser_classifies_sat_as_reachable() -> None:
    r = _parse_btormc("checking...\nsat\n", "", 0, "btormc", 5, 0.1)
    assert r.verdict == "reachable"


def test_btormc_parser_returns_unknown_for_garbage() -> None:
    r = _parse_btormc("hello world", "", 1, "btormc", 5, 0.1)
    assert r.verdict == "unknown"


def test_btormc_bridge_returns_unknown_when_binary_missing() -> None:
    r = BtorMC(binary="definitely-not-installed-btormc").check_reach(_counter(5), bound=3)
    assert r.verdict == "unknown"
    assert "not found in PATH" in (r.reason or "")


def test_btormc_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError):
        BtorMC(mode="invalid")


# ---- live tool tests --------------------------------------------------------

@pytest.mark.skipif(shutil.which("rIC3") is None, reason="rIC3 binary not installed")
def test_ric3_live_on_safe_counter() -> None:
    # Counter bounded at 5; property `x >= 6` is unreachable for all
    # time — rIC3 should close it as proved.
    m = Model()
    x = m.state(BV8, "x")
    m.init(x, m.const(BV8, 0))
    cap = m.const(BV8, 5)
    keep = m.op("ult", BV1, x, cap)
    m.next(x, m.ite(keep, m.op("add", BV8, x, m.const(BV8, 1)), x))
    m.bad(m.op("ugt", BV1, x, cap))
    r = Ric3().check_reach(m, timeout=30.0)
    assert r.verdict in ("proved", "unknown")


@pytest.mark.skipif(shutil.which("btormc") is None, reason="btormc binary not installed")
def test_btormc_live_on_unsafe_counter() -> None:
    r = BtorMC(mode="bmc").check_reach(_counter(5), bound=10, timeout=30.0)
    assert r.verdict in ("reachable", "unknown")
