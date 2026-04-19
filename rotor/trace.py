"""Source-level traces: MachineSteps lifted through DWARF, rendered.

M2 gate: markdown rendering of counterexamples for BMC-reachable
verdicts. Other formats (JSON, SARIF, GDB) are future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rotor.binary import Function, RISCVBinary
from rotor.btor2.riscv.decoder import Decoded
from rotor.dwarf import DwarfLineMap, SourceLocation
from rotor.witness import MachineStep, simulate

ABI = (
    "zero", "ra", "sp", "gp", "tp",
    "t0", "t1", "t2",
    "s0", "s1",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    "t3", "t4", "t5", "t6",
)


@dataclass(frozen=True)
class Trace:
    function: str
    binary_path: Path
    target_pc: int
    verdict: str
    bound: int
    reached_at: Optional[int]
    elapsed: float
    backend: str
    steps: tuple[MachineStep, ...]
    source_by_pc: dict[int, SourceLocation]
    initial_regs: dict[str, int]

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# Counterexample: can_reach({self.function}, 0x{self.target_pc:x})")
        lines.append("")
        reached_note = f" at step {self.reached_at}" if self.reached_at is not None else ""
        lines.append(
            f"**verdict**: {self.verdict}{reached_note} "
            f"(bound {self.bound}, {self.elapsed * 1000:.1f} ms, {self.backend})"
        )
        lines.append("")
        lines.append(f"**binary**: `{self.binary_path}`")
        lines.append("")
        lines.append("## Execution trace")
        lines.append("")
        lines.append("| step | pc       | instruction            | source               |")
        lines.append("|-----:|:---------|:-----------------------|:---------------------|")
        for st in self.steps:
            disasm = _disasm(st.decoded) if st.decoded is not None else "<no instruction>"
            src = self.source_by_pc.get(st.pc)
            src_cell = f"{Path(src.file).name}:{src.line}" if src else "-"
            lines.append(f"| {st.step:>4} | 0x{st.pc:06x} | {disasm:<22} | {src_cell:<20} |")
        lines.append("")
        lines.append("## Initial register values (witness)")
        lines.append("")
        lines.append("| register | value                  |")
        lines.append("|:---------|-----------------------:|")
        for i in range(1, 32):
            v = self.initial_regs.get(f"x{i}", 0)
            if v == 0:
                continue
            lines.append(f"| {ABI[i]:<8} | 0x{v:016x} |")
        if all(self.initial_regs.get(f"x{i}", 0) == 0 for i in range(1, 32)):
            lines.append("| *all zero* |                      0 |")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------

def build_trace(
    binary: RISCVBinary,
    function: str,
    target_pc: int,
    verdict: str,
    bound: int,
    reached_at: Optional[int],
    elapsed: float,
    backend: str,
    initial_regs: dict[str, int],
    dwarf: Optional[DwarfLineMap] = None,
) -> Trace:
    fn: Function = binary.function(function)
    if reached_at is None:
        steps: tuple[MachineStep, ...] = tuple()
    else:
        steps = tuple(simulate(binary, fn, initial_regs, reached_at))

    source_by_pc: dict[int, SourceLocation] = {}
    if dwarf is not None:
        for st in steps:
            loc = dwarf.lookup(st.pc)
            if loc is not None:
                source_by_pc[st.pc] = loc

    return Trace(
        function=function,
        binary_path=binary.path,
        target_pc=target_pc,
        verdict=verdict,
        bound=bound,
        reached_at=reached_at,
        elapsed=elapsed,
        backend=backend,
        steps=steps,
        source_by_pc=source_by_pc,
        initial_regs=dict(initial_regs),
    )


# ---------------------------------------------------------------------------

def _disasm(d: Decoded) -> str:
    """Terse assembler rendering, recognizing common pseudo-instructions."""
    r = ABI
    if d.mnem == "jalr" and d.rd == 0 and d.rs1 == 1 and d.imm == 0:
        return "ret"
    if d.mnem == "addi" and d.rs1 == 0:
        return f"li {r[d.rd]}, {d.imm}"
    if d.mnem == "addi" and d.imm == 0:
        return f"mv {r[d.rd]}, {r[d.rs1]}"
    if d.mnem == "sub" and d.rs1 == 0:
        return f"neg {r[d.rd]}, {r[d.rs2]}"
    if d.mnem == "sltu" and d.rs1 == 0:
        return f"snez {r[d.rd]}, {r[d.rs2]}"
    if d.mnem == "blt" and d.rs1 == 0:
        return f"bgtz {r[d.rs2]}, {d.imm:+d}"
    if d.mnem == "blt" and d.rs2 == 0:
        return f"bltz {r[d.rs1]}, {d.imm:+d}"
    if d.mnem in ("addi",):
        return f"{d.mnem} {r[d.rd]}, {r[d.rs1]}, {d.imm}"
    if d.mnem in ("addw", "sub", "sltu"):
        return f"{d.mnem} {r[d.rd]}, {r[d.rs1]}, {r[d.rs2]}"
    if d.mnem == "blt":
        return f"blt {r[d.rs1]}, {r[d.rs2]}, {d.imm:+d}"
    if d.mnem == "jalr":
        return f"jalr {r[d.rd]}, {d.imm}({r[d.rs1]})"
    return d.mnem  # pragma: no cover
