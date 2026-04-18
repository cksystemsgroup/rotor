"""Tests for rotor.riscv.support — unsupported-instruction detection."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("elftools")

from rotor.riscv.support import (
    SUPPORTED_OPCODES,
    UnsupportedInstruction,
    format_issues,
    scan_unsupported_instructions,
)


def test_supported_opcodes_includes_core_rv64i() -> None:
    for opcode in (0x03, 0x13, 0x17, 0x1B, 0x23, 0x33, 0x37, 0x3B,
                   0x63, 0x67, 0x6F, 0x73):
        assert opcode in SUPPORTED_OPCODES


def test_scan_fixture_is_clean() -> None:
    from rotor.binary import RISCVBinary

    fixture = os.path.join(
        os.path.dirname(__file__), "fixtures", "add2.elf"
    )
    if not os.path.exists(fixture):
        pytest.skip("add2.elf fixture not built")
    with RISCVBinary(fixture) as binary:
        low, high = binary.function_bounds("add2")
        issues = scan_unsupported_instructions(binary, low, high)
    assert issues == []


def test_scan_detects_compressed_instructions() -> None:
    """A synthetic 16-bit instruction should be flagged as RVC."""
    import types

    # Construct a minimal binary-like object that satisfies the scanner.
    binary = types.SimpleNamespace()
    seg = types.SimpleNamespace(
        start=0x1000,
        size=4,
        # Two bytes that form a compressed instruction (low 2 bits = 01).
        data=b"\x01\x00\x01\x00",
    )
    binary.code = seg
    issues = scan_unsupported_instructions(binary, 0x1000, 0x1004)
    assert len(issues) == 2
    assert all("compressed" in i.reason for i in issues)


def test_scan_detects_unsupported_opcode() -> None:
    """A 32-bit instruction with an opcode not in our set should flag."""
    import types

    binary = types.SimpleNamespace()
    # Opcode 0x0f = MISC-MEM (fence); low 2 bits = 11 so it's not compressed.
    word = 0x0000000F.to_bytes(4, "little")
    seg = types.SimpleNamespace(start=0x2000, size=4, data=word)
    binary.code = seg
    issues = scan_unsupported_instructions(binary, 0x2000, 0x2004)
    assert len(issues) == 1
    assert issues[0].opcode == 0x0F
    assert "fence" in format_issues(issues).lower()


def test_format_issues_empty() -> None:
    assert format_issues([]) == ""


def test_format_issues_nonempty() -> None:
    issue = UnsupportedInstruction(
        pc=0x1000, word=0x0f, opcode=0x0f, reason="test",
    )
    out = format_issues([issue])
    assert "0x1000" in out
    assert "BMC will trip" in out
    assert "Suggestions" in out
