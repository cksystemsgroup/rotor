"""M2 shipping gate: trace markdown lands for BMC counterexamples."""

from pathlib import Path

from rotor import Portfolio, RotorAPI, Z3BMC

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "add2.elf"


def test_reachable_verdict_produces_markdown_trace() -> None:
    with RotorAPI(FIXTURE, default_bound=2) as api:
        fn = api.binary.function("add2")
        r = api.can_reach(function="add2", target_pc=fn.start + 4)

    assert r.verdict == "reachable"
    assert r.trace is not None
    md = r.trace.to_markdown()
    assert "# Counterexample:" in md
    assert "addw a0, a0, a1" in md
    assert "ret" in md
    assert "add2.c" in md                         # DWARF-lifted source column


def test_unreachable_has_no_trace() -> None:
    with RotorAPI(FIXTURE, default_bound=1) as api:
        fn = api.binary.function("add2")
        r = api.can_reach(function="add2", target_pc=fn.start + 100)
    assert r.verdict == "unreachable"
    assert r.trace is None


def test_portfolio_produces_trace_on_reachable() -> None:
    # Two configs racing: a low bound (unreachable) and a high bound
    # (reachable). The high-bound config should win and still produce
    # a trace.
    portfolio = (
        Portfolio()
        .add(Z3BMC(), bound=0)
        .add(Z3BMC(), bound=4)
    )
    with RotorAPI(FIXTURE, portfolio=portfolio) as api:
        fn = api.binary.function("sign")
        # ret2 of sign is at fn.start + 0x14 (two steps after blt-taken).
        r = api.can_reach(function="sign", target_pc=fn.start + 0x14)
    assert r.verdict == "reachable"
    assert r.trace is not None
    assert "ret" in r.trace.to_markdown()
