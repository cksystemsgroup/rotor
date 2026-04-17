"""Phase 1: ELF and DWARF parsing.

Given a RISC-V ELF binary, extract everything needed to build a BTOR2 model
instance and to present solver results back in source terms:

* segments (code / data / rodata),
* symbol table (function and data symbols),
* PC → source (file:line:column) via the DWARF line program,
* function DIEs (bounds, parameters, locals, return type),
* variable locations at a given PC, with a minimal DWARF expression
  evaluator to resolve them against concrete register/memory state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection
except ImportError:  # pragma: no cover - dependency is runtime-only
    ELFFile = None  # type: ignore[assignment]
    SymbolTableSection = None  # type: ignore[assignment]


# ──────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Segment:
    """One contiguous region of the binary loaded at a known virtual address."""

    name: str
    start: int
    size: int
    data: bytes


@dataclass
class Symbol:
    """A symbol table entry (function or data)."""

    name: str
    address: int
    size: int
    kind: str  # 'func', 'object', 'section', 'notype', ...


@dataclass
class SourceLocation:
    """A source-code coordinate derived from the DWARF line program."""

    file: str
    line: int
    column: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


@dataclass
class DWARFLocation:
    """Describes *where* a variable lives at a given PC.

    The evaluator in :class:`DWARFExprEvaluator` turns this into a concrete
    integer when combined with a :class:`MachineState`. ``kind`` is one of
    ``'register'``, ``'frame_offset'``, ``'address'``, ``'expr'`` (raw DWARF
    expression bytes), or ``'composite'`` (future).
    """

    kind: str
    register: int | None = None
    offset: int | None = None
    address: int | None = None
    expr: bytes | None = None


@dataclass
class VariableInfo:
    """A DWARF-level variable (parameter or local)."""

    name: str
    type_name: str
    byte_size: int
    location: DWARFLocation


@dataclass
class FunctionInfo:
    """A DWARF-level function DIE summary."""

    name: str
    low_pc: int
    high_pc: int
    return_type: str = ""
    parameters: list[VariableInfo] = field(default_factory=list)
    locals: list[VariableInfo] = field(default_factory=list)

    def contains(self, pc: int) -> bool:
        return self.low_pc <= pc < self.high_pc


# ──────────────────────────────────────────────────────────────────────────
# Binary loader
# ──────────────────────────────────────────────────────────────────────────


class RISCVBinary:
    """A loaded RISC-V ELF binary with DWARF debug information.

    All heavy parsing (line map, DIE walk) is performed lazily on the first
    query that needs it, so cheap construction is possible.
    """

    def __init__(self, path: str) -> None:
        if ELFFile is None:
            raise ImportError(
                "pyelftools is required for RISCVBinary; install `pyelftools`."
            )
        self._path = path
        self._fp = open(path, "rb")
        self._elf = ELFFile(self._fp)
        self._check_arch()

        self.code = self._load_segment(".text")
        self.data = self._load_segment(".data")
        self.rodata = self._load_segment(".rodata")
        self.bss = self._load_segment(".bss")
        self.symbols: dict[str, Symbol] = self._load_symbols()

        self._has_dwarf = self._elf.has_dwarf_info()
        self._dwarf = self._elf.get_dwarf_info() if self._has_dwarf else None

        self._line_map: dict[int, SourceLocation] | None = None
        self._functions: list[FunctionInfo] | None = None
        self._functions_by_name: dict[str, FunctionInfo] = {}

    # ---------------------------------------------------------------- dunder

    def __repr__(self) -> str:
        bits = 64 if self.is_64bit else 32
        return f"RISCVBinary({self._path!r}, rv{bits}, entry=0x{self.entry:x})"

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> RISCVBinary:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --------------------------------------------------------------- headers

    @property
    def entry(self) -> int:
        return int(self._elf.header["e_entry"])

    def _check_arch(self) -> None:
        machine = self._elf.header["e_machine"]
        if machine != "EM_RISCV":
            raise ValueError(f"Expected RISC-V ELF, got {machine}")
        self.is_64bit = self._elf.elfclass == 64

    # -------------------------------------------------------------- segments

    def _load_segment(self, name: str) -> Segment | None:
        section = self._elf.get_section_by_name(name)
        if section is None:
            return None
        return Segment(
            name=name,
            start=int(section["sh_addr"]),
            size=int(section["sh_size"]),
            data=bytes(section.data()),
        )

    # --------------------------------------------------------------- symbols

    def _load_symbols(self) -> dict[str, Symbol]:
        result: dict[str, Symbol] = {}
        for section in self._elf.iter_sections():
            if not isinstance(section, SymbolTableSection):
                continue
            for sym in section.iter_symbols():
                name = sym.name
                if not name:
                    continue
                kind = sym["st_info"]["type"]
                kind_map = {
                    "STT_FUNC": "func",
                    "STT_OBJECT": "object",
                    "STT_SECTION": "section",
                    "STT_NOTYPE": "notype",
                }
                result[name] = Symbol(
                    name=name,
                    address=int(sym["st_value"]),
                    size=int(sym["st_size"]),
                    kind=kind_map.get(kind, kind.lower()),
                )
        return result

    # -------------------------------------------------------- pc → source map

    def pc_to_source(self, pc: int) -> SourceLocation | None:
        """Map a program counter to its source file:line:column, if known."""
        if not self._has_dwarf:
            return None
        if self._line_map is None:
            self._line_map = self._build_line_map()
        if pc in self._line_map:
            return self._line_map[pc]
        # Fall back to nearest lower PC (the row corresponding to the
        # sequence containing `pc`).
        best: SourceLocation | None = None
        best_pc = -1
        for addr, loc in self._line_map.items():
            if addr <= pc > best_pc:
                best_pc = addr
                best = loc
        return best

    def _build_line_map(self) -> dict[int, SourceLocation]:
        result: dict[int, SourceLocation] = {}
        assert self._dwarf is not None
        for cu in self._dwarf.iter_CUs():
            lineprog = self._dwarf.line_program_for_CU(cu)
            if lineprog is None:
                continue
            file_entries = lineprog["file_entry"]
            for entry in lineprog.get_entries():
                state = entry.state
                if state is None or state.end_sequence:
                    continue
                file_idx = state.file - 1
                if 0 <= file_idx < len(file_entries):
                    raw = file_entries[file_idx].name
                    fname = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                else:
                    fname = "?"
                result[state.address] = SourceLocation(
                    file=fname,
                    line=state.line,
                    column=getattr(state, "column", 0) or 0,
                )
        return result

    # ------------------------------------------------------------- functions

    def _ensure_functions(self) -> None:
        if self._functions is not None:
            return
        self._functions = []
        if not self._has_dwarf:
            # Seed from symbol table so function_by_name still works.
            for name, sym in self.symbols.items():
                if sym.kind == "func" and sym.size > 0:
                    fi = FunctionInfo(
                        name=name,
                        low_pc=sym.address,
                        high_pc=sym.address + sym.size,
                    )
                    self._functions.append(fi)
                    self._functions_by_name[name] = fi
            return

        assert self._dwarf is not None
        for cu in self._dwarf.iter_CUs():
            top = cu.get_top_DIE()
            self._walk_dies_for_functions(cu, top)

    def _walk_dies_for_functions(self, cu: Any, die: Any) -> None:
        if die.tag == "DW_TAG_subprogram":
            fi = self._function_from_die(cu, die)
            if fi is not None:
                assert self._functions is not None
                self._functions.append(fi)
                self._functions_by_name[fi.name] = fi
        for child in die.iter_children():
            self._walk_dies_for_functions(cu, child)

    def _function_from_die(self, cu: Any, die: Any) -> FunctionInfo | None:
        attrs = die.attributes
        if "DW_AT_low_pc" not in attrs:
            return None
        low_pc = int(attrs["DW_AT_low_pc"].value)
        high_pc_attr = attrs.get("DW_AT_high_pc")
        if high_pc_attr is None:
            return None
        hv = high_pc_attr.value
        high_pc = int(hv) if high_pc_attr.form == "DW_FORM_addr" else low_pc + int(hv)
        name_attr = attrs.get("DW_AT_name")
        if name_attr is None:
            # Could be an inlined instance; skip.
            return None
        raw = name_attr.value
        name = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)

        params: list[VariableInfo] = []
        locals_: list[VariableInfo] = []
        for child in die.iter_children():
            if child.tag == "DW_TAG_formal_parameter":
                vi = self._variable_from_die(cu, child)
                if vi is not None:
                    params.append(vi)
            elif child.tag == "DW_TAG_variable":
                vi = self._variable_from_die(cu, child)
                if vi is not None:
                    locals_.append(vi)

        return FunctionInfo(
            name=name,
            low_pc=low_pc,
            high_pc=high_pc,
            return_type=self._type_name(cu, attrs.get("DW_AT_type")),
            parameters=params,
            locals=locals_,
        )

    def _variable_from_die(self, cu: Any, die: Any) -> VariableInfo | None:
        attrs = die.attributes
        name_attr = attrs.get("DW_AT_name")
        if name_attr is None:
            return None
        raw = name_attr.value
        name = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)

        type_name = self._type_name(cu, attrs.get("DW_AT_type"))
        byte_size = self._byte_size(cu, attrs.get("DW_AT_type"))

        location = DWARFLocation(kind="unknown")
        loc_attr = attrs.get("DW_AT_location")
        if loc_attr is not None and isinstance(loc_attr.value, (bytes, list)):
            if isinstance(loc_attr.value, bytes):
                location = DWARFLocation(kind="expr", expr=loc_attr.value)

        return VariableInfo(
            name=name,
            type_name=type_name,
            byte_size=byte_size,
            location=location,
        )

    def _type_name(self, cu: Any, type_attr: Any) -> str:
        if type_attr is None:
            return "void"
        try:
            die = cu.get_DIE_from_refaddr(type_attr.value + cu.cu_offset)
        except Exception:
            return "?"
        if "DW_AT_name" in die.attributes:
            raw = die.attributes["DW_AT_name"].value
            return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        return die.tag.replace("DW_TAG_", "")

    def _byte_size(self, cu: Any, type_attr: Any) -> int:
        if type_attr is None:
            return 0
        try:
            die = cu.get_DIE_from_refaddr(type_attr.value + cu.cu_offset)
        except Exception:
            return 0
        size_attr = die.attributes.get("DW_AT_byte_size")
        if size_attr is not None:
            return int(size_attr.value)
        return 0

    def function_at(self, pc: int) -> FunctionInfo | None:
        """Return the function containing `pc`, or ``None``."""
        self._ensure_functions()
        assert self._functions is not None
        for fi in self._functions:
            if fi.contains(pc):
                return fi
        return None

    def function_by_name(self, name: str) -> FunctionInfo | None:
        self._ensure_functions()
        return self._functions_by_name.get(name)

    def function_bounds(self, name: str) -> tuple[int, int]:
        """Return ``(low_pc, high_pc)`` for the named function.

        Falls back to the symbol table when DWARF is unavailable.
        """
        fi = self.function_by_name(name)
        if fi is not None:
            return (fi.low_pc, fi.high_pc)
        sym = self.symbols.get(name)
        if sym is not None and sym.size > 0:
            return (sym.address, sym.address + sym.size)
        raise KeyError(f"Function not found: {name}")

    # -------------------------------------------------------------- variables

    def live_variables_at(self, pc: int) -> list[VariableInfo]:
        """Return all variables in scope at ``pc``.

        This uses the function DIE's parameter and local lists. A fuller
        implementation would narrow locals by scope (lexical block DIEs) and
        by location-list validity ranges.
        """
        fi = self.function_at(pc)
        if fi is None:
            return []
        return list(fi.parameters) + list(fi.locals)

    def resolve_variable(
        self,
        var: VariableInfo,
        registers: dict[int, int],
        memory: dict[int, int],
        frame_base: int = 0,
    ) -> int | None:
        """Evaluate ``var.location`` against concrete register + memory state.

        Returns the variable's current value or ``None`` if the location
        cannot be evaluated in the provided state.
        """
        from rotor.dwarf import DWARFExprEvaluator

        loc = var.location
        word_bytes = 8 if self.is_64bit else 4

        if loc.kind == "register" and loc.register is not None:
            return registers.get(loc.register)
        if loc.kind == "frame_offset" and loc.offset is not None:
            addr = frame_base + loc.offset
            return _read_word(memory, addr, var.byte_size or word_bytes)
        if loc.kind == "address" and loc.address is not None:
            return _read_word(memory, loc.address, var.byte_size or word_bytes)
        if loc.kind == "expr" and loc.expr is not None:
            ev = DWARFExprEvaluator(
                registers=registers,
                memory=memory,
                frame_base=frame_base,
                word_bytes=word_bytes,
            )
            result = ev.evaluate(loc.expr)
            if result is None:
                return None
            if ev.is_value:
                return result
            return _read_word(memory, result, var.byte_size or word_bytes)
        return None

    # -------------------------------------------------------- disassembly hook

    def disassemble(self, pc: int) -> str:
        """Return a textual disassembly of the instruction at ``pc``.

        This is a simple stub: the Python package defers full RISC-V
        disassembly to an optional helper; when unavailable we render the
        raw bytes.
        """
        if self.code is None:
            return f"<no code at 0x{pc:x}>"
        offset = pc - self.code.start
        if offset < 0 or offset >= self.code.size:
            return f"<pc 0x{pc:x} out of .text>"
        # RV64 instructions are either 32 bits or 16 bits (RVC).
        raw = self.code.data[offset:offset + 4]
        if len(raw) < 2:
            return f"<pc 0x{pc:x} truncated>"
        word = int.from_bytes(raw[:4] if len(raw) >= 4 else raw + b"\x00\x00", "little")
        return f"0x{pc:08x}: {word:08x}"


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _read_word(memory: dict[int, int], addr: int, size: int) -> int | None:
    """Assemble a little-endian integer of ``size`` bytes from a byte map."""
    if size <= 0:
        return None
    value = 0
    for i in range(size):
        byte = memory.get(addr + i)
        if byte is None:
            return None
        value |= (byte & 0xFF) << (8 * i)
    return value
