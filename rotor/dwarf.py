"""Minimal DWARF .debug_line access: PC -> (file, line, column).

M1 only needs enough to label traces. We keep this intentionally small.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from elftools.elf.elffile import ELFFile


@dataclass(frozen=True)
class SourceLocation:
    file: str
    line: int
    column: int


class DwarfLineMap:
    """PC -> SourceLocation via the DWARF line-number program."""

    def __init__(self, elf_path: str | Path) -> None:
        self._entries: list[tuple[int, SourceLocation]] = []
        with open(elf_path, "rb") as fh:
            elf = ELFFile(fh)
            if not elf.has_dwarf_info():
                return
            dw = elf.get_dwarf_info()
            for cu in dw.iter_CUs():
                lineprog = dw.line_program_for_CU(cu)
                if lineprog is None:
                    continue
                file_entries = lineprog["file_entry"]
                for entry in lineprog.get_entries():
                    s = entry.state
                    if s is None or s.end_sequence:
                        continue
                    # DWARF file indices are 1-based for DWARF <= 4; 0-based for DWARF 5.
                    idx = s.file
                    if lineprog.header.version < 5:
                        idx -= 1
                    try:
                        name = file_entries[idx].name.decode("utf-8", "replace")
                    except (IndexError, AttributeError):
                        name = "<unknown>"
                    self._entries.append(
                        (s.address, SourceLocation(file=name, line=s.line, column=s.column))
                    )
        self._entries.sort(key=lambda t: t[0])

    def lookup(self, pc: int) -> SourceLocation | None:
        """Greatest entry whose address <= pc."""
        if not self._entries:
            return None
        lo, hi = 0, len(self._entries)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._entries[mid][0] <= pc:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return None
        return self._entries[lo - 1][1]
