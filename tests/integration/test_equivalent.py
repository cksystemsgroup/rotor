"""Track D.3 integration: `rotor equivalent` end-to-end.

The equivalence verb compares two functions via a product
construction: both sides run in the same BTOR2 model with shared
initial registers, and `bad` fires when their output registers
disagree at their respective return sites.
"""

from __future__ import annotations

import io
from pathlib import Path

from rotor import RotorAPI
from rotor.cli import main

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ADD2 = str(FIXTURES / "add2.elf")
MULT = str(FIXTURES / "mult.elf")


def _run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


# ---- API ----

def test_api_add2_equivalent_to_itself() -> None:
    with RotorAPI(ADD2) as api:
        r = api.are_equivalent(
            other_binary_path=ADD2, function="add2", bound=4,
        )
    assert r.verdict == "unreachable"


def test_api_add2_not_equivalent_to_sign() -> None:
    # add2(a, b) = a + b; sign(x) = 1 / 0 / -1. Different outputs on
    # many inputs — BMC must find a witness pair.
    with RotorAPI(ADD2) as api:
        r = api.are_equivalent(
            other_binary_path=ADD2,
            function="add2", function_b="sign",
            bound=6,
        )
    assert r.verdict == "reachable"


def test_api_different_output_registers() -> None:
    # Compare mul64 to itself on a register the function doesn't write
    # (say ra = x1). Since ra is shared at init and neither side writes
    # it (mul64 is a leaf: mul a0,a1,a0; ret), captured_a = captured_b =
    # shared initial ra → unreachable.
    with RotorAPI(MULT) as api:
        r = api.are_equivalent(
            other_binary_path=MULT, function="mul64",
            output_register=1,          # compare ra
            bound=3,
        )
    assert r.verdict == "unreachable"


# ---- CLI ----

def test_cli_equivalent_exits_ok_for_equivalent() -> None:
    code, out, err = _run(
        "equivalent", ADD2, ADD2,
        "--function", "add2", "--bound", "4",
    )
    assert code == 0                              # EXIT_OK (no disagreement)
    assert "verdict  : unreachable" in out


def test_cli_equivalent_exits_found_for_divergence() -> None:
    code, out, err = _run(
        "equivalent", ADD2, ADD2,
        "--function", "add2",
        "--function-b", "sign",
        "--bound", "6",
    )
    assert code == 1                              # EXIT_FOUND (divergence)
    assert "verdict  : reachable" in out


def test_cli_equivalent_accepts_abi_output_register() -> None:
    # Compare add2 vs sign on a1 (x11). sign doesn't modify a1 (it's
    # not used; a0 is the input/output). add2 also doesn't modify a1.
    # Both sides' captured_a1 = shared initial a1 → unreachable.
    code, out, err = _run(
        "equivalent", ADD2, ADD2,
        "--function", "add2", "--function-b", "sign",
        "--output-register", "a1",
        "--bound", "6",
    )
    assert code == 0
    assert "verdict  : unreachable" in out
