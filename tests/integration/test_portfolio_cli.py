"""Track A.4 integration: `rotor reach --portfolio` races available
backends end-to-end.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from rotor.cli import main

ADD2 = str((Path(__file__).resolve().parents[1] / "fixtures" / "add2.elf"))
COUNTER = str((Path(__file__).resolve().parents[1] / "fixtures" / "counter.elf"))


def _run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_portfolio_reaches_ret_of_add2() -> None:
    code, out, err = _run(
        "reach", ADD2, "--function", "add2", "--target", "0x100b4", "--portfolio",
    )
    assert code == 1                              # reachable → EXIT_FOUND
    assert "verdict  : reachable" in out
    # The fastest racer wins; it could be Z3, Bitwuzla, CVC5, or a
    # Pono engine if the binary is on PATH. Any bounded BMC is fine.
    assert any(b in out for b in (
        "z3-bmc", "bitwuzla-bmc", "cvc5-bmc", "pono-bmc",
    ))


def test_portfolio_proves_tiny_mask_safe() -> None:
    # This is the Track A headline test: the portfolio must return a
    # globally-conclusive `proved` verdict on tiny_mask's dead branch,
    # not just the bounded `unreachable` that Z3BMC alone would report.
    code, out, err = _run(
        "reach", COUNTER, "--function", "tiny_mask", "--target", "0x1117c",
        "--portfolio",
    )
    assert code == 0                              # proved → EXIT_OK
    assert "verdict  : proved" in out
    assert "invariant:" in out


def test_portfolio_mutually_exclusive_with_unbounded() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["reach", ADD2, "--function", "add2", "--target", "0x100b4",
              "--portfolio", "--unbounded"])
    assert exc_info.value.code == 2


def test_portfolio_mutually_exclusive_with_cegar() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["reach", ADD2, "--function", "add2", "--target", "0x100b4",
              "--portfolio", "--cegar"])
    assert exc_info.value.code == 2
