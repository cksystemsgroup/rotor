"""Track D.2 integration: `rotor find-input` end-to-end.

Exercises the synthesis verb at both API and CLI level. The
predicate polarity is the opposite of verify — `reachable` here
means "a witness input was found" and `initial_regs` is the
synthesized input.
"""

from __future__ import annotations

import io
from pathlib import Path

from rotor import RotorAPI
from rotor.cli import main

ADD2 = str((Path(__file__).resolve().parents[1] / "fixtures" / "add2.elf"))
BITOPS = str((Path(__file__).resolve().parents[1] / "fixtures" / "bitops.elf"))


def _run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_api_find_input_synthesizes_witness_for_add2_eq_42() -> None:
    # add2(a, b) returns a+b. Any inputs summing to 42 (in 32-bit
    # arithmetic) satisfy the predicate.
    with RotorAPI(ADD2, default_bound=3) as api:
        r = api.find_input(function="add2", register=10, comparison="eq", rhs=42)
    assert r.verdict == "reachable"
    assert r.initial_regs                           # witness populated
    # The solver picked x10 + x11 = 42 mod 2^32 (since add2 is addw).
    x10 = r.initial_regs.get("x10", 0)
    x11 = r.initial_regs.get("x11", 0)
    # Mirror the addw semantics: sum low 32 bits, sign-extend to 64.
    low = (x10 + x11) & 0xFFFFFFFF
    if low & (1 << 31):
        low |= 0xFFFFFFFF00000000
    assert low == 42


def test_api_find_input_no_witness_within_bound_zero() -> None:
    # Bound 0 means PC hasn't moved from entry; no ret has executed.
    # The predicate can never hold, so no witness.
    with RotorAPI(ADD2, default_bound=0) as api:
        r = api.find_input(function="add2", register=10, comparison="eq", rhs=42)
    assert r.verdict == "unreachable"


def test_cli_find_input_reports_witness() -> None:
    code, out, err = _run(
        "find-input", ADD2,
        "--function", "add2",
        "--register", "a0",
        "--op", "eq",
        "--value", "0",
        "--bound", "3",
    )
    assert code == 1                              # EXIT_FOUND
    assert "verdict  : reachable" in out
    assert "witness  :" in out


def test_cli_find_input_on_bitops_popcount() -> None:
    # Find an input that makes popcount's return value equal to 1.
    # The predicate must be satisfiable (any power-of-two input
    # works), so the solver should synthesize a witness.
    code, out, err = _run(
        "find-input", BITOPS,
        "--function", "popcount",
        "--register", "a0",
        "--op", "eq",
        "--value", "1",
        "--bound", "12",
    )
    assert code == 1                              # reachable
    assert "verdict  : reachable" in out


def test_find_input_matches_negated_verify() -> None:
    # Equivalence claim: find_input(cmp, rhs) is satisfiable iff
    # verify(negated cmp, rhs) says reachable. Spot-check with
    # eq / neq on add2 at a small bound.
    with RotorAPI(ADD2, default_bound=3) as api:
        fi = api.find_input(function="add2", register=10, comparison="eq", rhs=0)
        vf = api.verify(function="add2", register=10, comparison="neq", rhs=0)
    assert fi.verdict == "reachable"
    assert vf.verdict == "reachable"
