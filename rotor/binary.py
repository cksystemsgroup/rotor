"""ELF loading and per-function instruction enumeration.

Minimal M1 surface:
- Load a RISC-V ELF via pyelftools.
- Enumerate symbols; report function (start, end) ranges.
- Yield 32-bit instruction words at each PC in a function.

Compressed (RVC) instructions are out of scope for M1 — fixtures are
built with -march=rv64im so the stream is pure 32-bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection


@dataclass(frozen=True)
class Function:
    name: str
    start: int   # inclusive
    end: int     # exclusive

    def contains(self, pc: int) -> bool:
        return self.start <= pc < self.end


@dataclass(frozen=True)
class Instruction:
    pc: int
    word: int            # 32-bit RISC-V instruction word (for RVC:
                         # expanded equivalent — see rotor.btor2.riscv.rvc)
    size: int = 4        # 2 for compressed (RVC), 4 for RV64I/M


class RISCVBinary:
    """Read-only view of a RISC-V ELF binary."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        self._elf = ELFFile(self._fh)
        if self._elf.get_machine_arch() != "RISC-V":
            raise ValueError(f"not a RISC-V ELF: {self.path}")
        self.is_64bit = self._elf.elfclass == 64
        self.entry = self._elf["e_entry"]
        self._functions: dict[str, Function] | None = None
        self._text_cache: dict[int, bytes] = {}

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "RISCVBinary":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def functions(self) -> dict[str, Function]:
        if self._functions is None:
            self._functions = dict(self._enumerate_functions())
        return self._functions

    def function(self, name: str) -> Function:
        try:
            return self.functions[name]
        except KeyError as exc:
            raise KeyError(f"function {name!r} not found in {self.path}") from exc

    def instructions(self, fn: Function) -> Iterator[Instruction]:
        """Yield instructions in [fn.start, fn.end), handling RVC.

        RISC-V uses a variable-length encoding: if the low two bits of
        the first halfword are `0b11` the instruction is 32-bit;
        otherwise it is a 16-bit compressed (RVC) instruction and we
        emit its expanded 32-bit RV64I/M equivalent via
        `rotor.btor2.riscv.rvc.expand_rvc`. `Instruction.size` tracks
        the real byte size (2 or 4) so the lowering pipeline computes
        correct fall-through PCs.
        """
        from rotor.btor2.riscv.rvc import expand_rvc
        data = self._read_range(fn.start, fn.end)
        offset = 0
        end = len(data)
        while offset < end:
            lo = int.from_bytes(data[offset:offset + 2], "little")
            if (lo & 0b11) == 0b11:
                if offset + 4 > end:
                    break                          # truncated 32-bit inst
                word = int.from_bytes(data[offset:offset + 4], "little")
                yield Instruction(pc=fn.start + offset, word=word, size=4)
                offset += 4
            else:
                expanded = expand_rvc(lo)
                if expanded is None:
                    # Let the decoder signal unsupported-instruction on the
                    # raw 16-bit word; preserves the existing error path.
                    yield Instruction(pc=fn.start + offset, word=lo, size=2)
                else:
                    yield Instruction(pc=fn.start + offset, word=expanded, size=2)
                offset += 2

    def loadable_bytes(self) -> Iterator[tuple[int, int]]:
        """Yield (vaddr, byte) pairs for every file-backed byte in PT_LOAD.

        These are the bytes the ELF loader would place in memory at process
        start — `.text`, `.rodata`, `.data`, and anything else covered by
        `p_filesz`. Zero-initialized `.bss` bytes (inside `p_memsz` but
        outside `p_filesz`) are omitted here; rotor lets the memory model
        treat them as free and the verifier learns bounds from use.
        """
        seen: set[int] = set()
        for seg in self._elf.iter_segments():
            if seg["p_type"] != "PT_LOAD":
                continue
            vbeg = seg["p_vaddr"]
            data = seg.data()
            filesz = seg["p_filesz"]
            for offset in range(filesz):
                addr = vbeg + offset
                if addr in seen:
                    continue                         # overlapping segment: first wins
                seen.add(addr)
                yield addr, data[offset]

    def _enumerate_functions(self) -> Iterator[tuple[str, Function]]:
        for section in self._elf.iter_sections():
            if not isinstance(section, SymbolTableSection):
                continue
            for sym in section.iter_symbols():
                if sym["st_info"]["type"] != "STT_FUNC":
                    continue
                size = sym["st_size"]
                if size == 0:
                    continue
                start = sym["st_value"]
                yield sym.name, Function(name=sym.name, start=start, end=start + size)

    def _read_range(self, start: int, end: int) -> bytes:
        """Read raw bytes spanning [start, end) from loadable segments."""
        # Simple path: find the PT_LOAD segment containing `start` and slice.
        for seg in self._elf.iter_segments():
            if seg["p_type"] != "PT_LOAD":
                continue
            vbeg = seg["p_vaddr"]
            vend = vbeg + seg["p_filesz"]
            if vbeg <= start and end <= vend:
                data = seg.data()
                return data[start - vbeg:end - vbeg]
        raise ValueError(f"range 0x{start:x}..0x{end:x} not covered by any PT_LOAD segment")
