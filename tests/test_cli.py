"""Tests for the rotor command-line interface.

Each test invokes ``rotor.cli.main`` with a small argv, capturing stdout/
stderr via ``capsys`` and checking both the exit code and a distinctive
substring of the output. Fixtures are built by tests/fixtures/build.sh.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("elftools")

from rotor.cli import main

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture(name: str) -> str:
    path = os.path.join(FIXTURES, name)
    if not os.path.exists(path):
        pytest.skip(f"missing fixture {name}")
    return path


# ──────────────────────────────────────────────────────────────────────────
# info
# ──────────────────────────────────────────────────────────────────────────


def test_info_prints_headers(capsys) -> None:
    rc = main(["info", _fixture("add2.elf")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "is_64bit" in out
    assert "entry" in out


def test_info_functions_flag(capsys) -> None:
    rc = main(["info", _fixture("add2.elf"), "--functions"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "add2" in out


# ──────────────────────────────────────────────────────────────────────────
# disasm
# ──────────────────────────────────────────────────────────────────────────


def test_disasm_function(capsys) -> None:
    rc = main(["disasm", _fixture("add2.elf"), "--function", "add2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "addi sp" in out
    assert "addw" in out
    assert "ret" in out


# ──────────────────────────────────────────────────────────────────────────
# analyze
# ──────────────────────────────────────────────────────────────────────────


def test_analyze_clean_function(capsys) -> None:
    rc = main(["analyze", _fixture("add2.elf"), "--function", "add2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "clean" in out


# ──────────────────────────────────────────────────────────────────────────
# btor2
# ──────────────────────────────────────────────────────────────────────────


def test_btor2_writes_to_file(tmp_path, capsys) -> None:
    out_path = tmp_path / "out.btor2"
    rc = main([
        "btor2", _fixture("add2.elf"),
        "--function", "add2",
        "-o", str(out_path),
    ])
    assert rc == 0
    text = out_path.read_text()
    assert "sort bitvec" in text
    assert "state" in text


# ──────────────────────────────────────────────────────────────────────────
# check / reach / verify (need Bitwuzla)
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _bitwuzla():
    pytest.importorskip("bitwuzla")


def test_check_bounded_unsat(capsys, _bitwuzla) -> None:
    rc = main([
        "check", _fixture("add2.elf"),
        "--function", "add2",
        "--bound", "8",
    ])
    out = capsys.readouterr().out
    assert "verdict  : unsat" in out
    assert rc == 0


def test_reach_finds_reachable_pc(capsys, _bitwuzla) -> None:
    rc = main([
        "reach", _fixture("add2.elf"),
        "--function", "add2",
        "--condition", "pc == 0x11178",
        "--bound", "10",
    ])
    out = capsys.readouterr().out
    assert "verdict  : reachable" in out
    # Exit code 1 for a reachable condition (something found).
    assert rc == 1


def test_reach_unreachable(capsys, _bitwuzla) -> None:
    rc = main([
        "reach", _fixture("add2.elf"),
        "--function", "add2",
        "--condition", "pc == 0x99999",  # way outside function
        "--bound", "6",
    ])
    out = capsys.readouterr().out
    assert "verdict  : unreachable" in out
    assert rc == 0


def test_verify_unbounded_trivial_invariant(capsys, _bitwuzla) -> None:
    rc = main([
        "verify", _fixture("add2.elf"),
        "--function", "add2",
        "--invariant", "0 == 0",
        "--unbounded",
        "--bound", "4",
    ])
    out = capsys.readouterr().out
    assert "verdict  : holds" in out
    assert "unbounded: True" in out
    assert "inductive invariant" in out
    assert rc == 0


# ──────────────────────────────────────────────────────────────────────────
# crotor-check
# ──────────────────────────────────────────────────────────────────────────


def test_crotor_check_missing_binary(capsys) -> None:
    rc = main(["crotor-check", "--binary", "definitely-not-installed-zzz"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err
