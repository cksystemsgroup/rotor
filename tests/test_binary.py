"""Tests for :mod:`rotor.binary`.

These exercise the stand-alone bits (helpers, data classes) without needing
an ELF fixture; ELF-based tests live in ``test_binary_integration.py`` and
are skipped when pyelftools or fixtures are unavailable.
"""

from __future__ import annotations

import pytest

from rotor.binary import (
    DWARFLocation,
    FunctionInfo,
    Segment,
    SourceLocation,
    Symbol,
    VariableInfo,
    _read_word,
)


def test_segment_dataclass() -> None:
    seg = Segment(name=".text", start=0x1000, size=4, data=b"\x13\x00\x00\x00")
    assert seg.size == 4
    assert seg.data == b"\x13\x00\x00\x00"


def test_source_location_str() -> None:
    loc = SourceLocation(file="foo.c", line=12, column=7)
    assert str(loc) == "foo.c:12:7"


def test_function_info_contains() -> None:
    fi = FunctionInfo(name="main", low_pc=0x1000, high_pc=0x1040)
    assert fi.contains(0x1000)
    assert fi.contains(0x1020)
    assert not fi.contains(0x1040)
    assert not fi.contains(0x0FFF)


def test_symbol_kinds() -> None:
    sym = Symbol(name="main", address=0x1000, size=64, kind="func")
    assert sym.kind == "func"


def test_dwarf_location_defaults() -> None:
    loc = DWARFLocation(kind="register", register=10)
    assert loc.offset is None
    assert loc.expr is None


def test_variable_info_roundtrip() -> None:
    loc = DWARFLocation(kind="frame_offset", offset=-8)
    vi = VariableInfo(name="x", type_name="int", byte_size=4, location=loc)
    assert vi.name == "x"
    assert vi.location.offset == -8


def test_read_word_little_endian() -> None:
    mem = {0x2000 + i: (0xAABBCCDD >> (8 * i)) & 0xFF for i in range(4)}
    assert _read_word(mem, 0x2000, 4) == 0xAABBCCDD


def test_read_word_missing_byte() -> None:
    mem = {0x2000: 0x01, 0x2001: 0x02}
    assert _read_word(mem, 0x2000, 4) is None


@pytest.mark.skipif(
    pytest.importorskip("elftools", reason="pyelftools not installed") is None,
    reason="pyelftools not available",
)
def test_binary_requires_elf(tmp_path) -> None:
    from rotor.binary import RISCVBinary

    bogus = tmp_path / "empty"
    bogus.write_bytes(b"not-an-elf")
    with pytest.raises(Exception):
        RISCVBinary(str(bogus))
