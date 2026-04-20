"""CLI behavior tests driven through rotor.cli.main(argv, out, err)."""

from __future__ import annotations

import io
from pathlib import Path

from rotor.cli import main

FIXTURE = str((Path(__file__).resolve().parents[1] / "fixtures" / "add2.elf"))


def _run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_info_without_functions_flag() -> None:
    code, out, err = _run("info", FIXTURE)
    assert code == 0
    assert "path      :" in out
    assert "is_64bit  : True" in out
    assert "entry     :" in out
    assert "functions :" not in out


def test_info_with_functions_flag() -> None:
    code, out, err = _run("info", FIXTURE, "--functions")
    assert code == 0
    assert "functions :" in out
    assert " add2" in out
    assert " sign" in out


def test_disasm_add2() -> None:
    code, out, err = _run("disasm", FIXTURE, "--function", "add2")
    assert code == 0
    assert "addw a0, a0, a1" in out
    assert "ret" in out


def test_reach_reachable_exit_code() -> None:
    code, out, err = _run(
        "reach", FIXTURE,
        "--function", "add2",
        "--target", "0x100b4",
        "--bound", "2",
    )
    assert code == 1                              # reached
    assert "verdict  : reachable" in out
    assert "step     : 1" in out
    # Trace markdown lands on stderr by default.
    assert "# Counterexample" in err


def test_reach_unreachable_exit_code() -> None:
    code, out, err = _run(
        "reach", FIXTURE,
        "--function", "add2",
        "--target", "0x100c0",
        "--bound", "1",
    )
    assert code == 0                              # safe up to bound
    assert "verdict  : unreachable" in out
    assert err == ""                              # no counterexample


def test_reach_writes_trace_to_file(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.md"
    code, out, err = _run(
        "reach", FIXTURE,
        "--function", "add2",
        "--target", "0x100b4",
        "--bound", "2",
        "--trace", str(trace_path),
    )
    assert code == 1
    assert trace_path.exists()
    assert "# Counterexample" in trace_path.read_text()


# ---------------------------------------------------------------- BTOR2 subcommands


_COUNTER_BTOR2 = (
    "; counter that reaches 100 at step 100\n"
    "1 sort bitvec 8\n"
    "2 sort bitvec 1\n"
    "3 state 1 counter\n"
    "4 zero 1\n"
    "5 init 1 3 4\n"
    "6 one 1\n"
    "7 add 1 3 6\n"
    "8 next 1 3 7\n"
    "9 constd 1 100\n"
    "10 eq 2 3 9\n"
    "11 bad 10\n"
)


def test_btor2_roundtrip_prints_normalized_model(tmp_path: Path) -> None:
    f = tmp_path / "m.btor2"
    f.write_text(_COUNTER_BTOR2)
    code, out, err = _run("btor2-roundtrip", str(f))
    assert code == 0
    assert err == ""
    # zero/one get normalized to constd on re-emit.
    assert " zero " not in out and " one " not in out
    assert out.count("constd ") == 3


def test_btor2_roundtrip_surfaces_diagnostics_on_stderr(tmp_path: Path) -> None:
    f = tmp_path / "m.btor2"
    f.write_text("1 sort bitvec 8\n2 frobnicate 1\n")
    code, out, err = _run("btor2-roundtrip", str(f))
    assert code == 2
    assert "error:" in err and "frobnicate" in err
    # Valid line still gets re-emitted.
    assert "sort bitvec 8" in out


def test_solve_btor2_unreachable_below_threshold(tmp_path: Path) -> None:
    f = tmp_path / "m.btor2"
    f.write_text(_COUNTER_BTOR2)
    code, out, err = _run("solve-btor2", str(f), "--bound", "5")
    assert code == 0
    assert "verdict  : unreachable" in out
    assert "bound    : 5" in out


def test_solve_btor2_reachable_above_threshold(tmp_path: Path) -> None:
    f = tmp_path / "m.btor2"
    f.write_text(_COUNTER_BTOR2)
    code, out, err = _run("solve-btor2", str(f), "--bound", "150")
    assert code == 1
    assert "verdict  : reachable" in out
    assert "step     : 100" in out


def test_solve_btor2_races_multiple_bounds(tmp_path: Path) -> None:
    """Two bounds (one short, one long): portfolio must see the reachable
    verdict from the longer bound and short-circuit on it."""
    f = tmp_path / "m.btor2"
    f.write_text(_COUNTER_BTOR2)
    code, out, err = _run(
        "solve-btor2", str(f),
        "--bound", "5",
        "--bound", "150",
    )
    assert code == 1
    assert "verdict  : reachable" in out


def test_solve_btor2_exits_unknown_on_parse_error(tmp_path: Path) -> None:
    f = tmp_path / "m.btor2"
    f.write_text("1 sort bitvec 8\n2 frobnicate 1\n3 bad 2\n")
    code, out, err = _run("solve-btor2", str(f))
    assert code == 2
    assert "frobnicate" in err
    # No verdict line: parse error short-circuits before solving.
    assert "verdict" not in out


def test_unknown_function_error() -> None:
    code, out, err = _run("disasm", FIXTURE, "--function", "nope")
    assert code == 2
    assert "error" in err
