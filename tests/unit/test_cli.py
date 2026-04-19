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
    assert f"trace    : {trace_path}" in out


def test_unknown_function_error() -> None:
    code, out, err = _run("disasm", FIXTURE, "--function", "nope")
    assert code == 2
    assert "error" in err
