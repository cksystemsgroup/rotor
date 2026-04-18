"""Classification of RISC-V opcodes against the native builder's coverage.

When a real ELF contains instructions outside the native Python builder's
subset (e.g. compressed RVC, fence, CSR), BMC still runs — it just trips
``illegal-instruction`` the moment the unsupported PC is fetched. Users
get ``verdict=sat`` with a trace that ends at the offending instruction,
which is confusing in context.

This module helps the CLI and RotorAPI surface that situation early:
scan the code segment, identify any instruction outside the supported
opcode set, and emit a warning listing the offending PCs plus suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from rotor.binary import RISCVBinary


# Top-level opcodes the native builder recognizes. Top-level only — we
# can't disambiguate illegal f3/f7 combinations at this level, but ~99%
# of cases where the native builder will trip illegal-instruction are
# caught by "opcode not in this set".
SUPPORTED_OPCODES: frozenset[int] = frozenset([
    0x03,  # OP_LOAD
    0x13,  # OP_OP_IMM
    0x17,  # OP_AUIPC
    0x1B,  # OP_OP_IMM_32
    0x23,  # OP_STORE
    0x33,  # OP_OP (includes RV64M via funct7)
    0x37,  # OP_LUI
    0x3B,  # OP_OP_32 (includes RV64M 32-bit)
    0x63,  # OP_BRANCH
    0x67,  # OP_JALR
    0x6F,  # OP_JAL
    0x73,  # OP_SYSTEM (ECALL only; CSR* not modeled)
])

OPCODE_NAMES: dict[int, str] = {
    0x03: "LOAD",
    0x13: "OP-IMM",
    0x17: "AUIPC",
    0x1B: "OP-IMM-32",
    0x23: "STORE",
    0x33: "OP (R-type)",
    0x37: "LUI",
    0x3B: "OP-32 (R-type 32-bit)",
    0x63: "BRANCH",
    0x67: "JALR",
    0x6F: "JAL",
    0x73: "SYSTEM",
    # Common unsupported families (for messages):
    0x0F: "MISC-MEM (fence)",
    0x07: "LOAD-FP",
    0x27: "STORE-FP",
    0x43: "FMADD",
    0x47: "FMSUB",
    0x4B: "FNMSUB",
    0x4F: "FNMADD",
    0x53: "OP-FP",
    0x2F: "AMO",
}


@dataclass
class UnsupportedInstruction:
    pc: int
    word: int
    opcode: int
    reason: str

    def __str__(self) -> str:
        op_name = OPCODE_NAMES.get(self.opcode, f"0x{self.opcode:02x}")
        return (
            f"0x{self.pc:x}: 0x{self.word:08x}  opcode={op_name}  {self.reason}"
        )


def scan_unsupported_instructions(
    binary: "RISCVBinary",
    start: int | None = None,
    end: int | None = None,
) -> list[UnsupportedInstruction]:
    """Walk the code segment and return every instruction the native
    builder would reject.

    Detects:
      * Compressed (RVC) 16-bit instructions — low two bits != 0b11.
      * Any 32-bit instruction whose opcode field is not in
        :data:`SUPPORTED_OPCODES`.
    """
    if binary.code is None:
        return []
    low = start if start is not None else binary.code.start
    high = end if end is not None else binary.code.start + binary.code.size
    low = max(low, binary.code.start)
    high = min(high, binary.code.start + binary.code.size)
    issues: list[UnsupportedInstruction] = []
    data = binary.code.data
    pc = low
    while pc + 2 <= high:
        offset = pc - binary.code.start
        half = int.from_bytes(data[offset:offset + 2], "little")
        if (half & 0x3) != 0x3:
            issues.append(
                UnsupportedInstruction(
                    pc=pc, word=half, opcode=half & 0x3,
                    reason="compressed (RVC) 16-bit instruction not modeled",
                )
            )
            pc += 2
            continue
        if offset + 4 > len(data):
            break
        word = int.from_bytes(data[offset:offset + 4], "little")
        opcode = word & 0x7F
        if opcode not in SUPPORTED_OPCODES:
            issues.append(
                UnsupportedInstruction(
                    pc=pc, word=word, opcode=opcode,
                    reason="opcode not in native builder's subset",
                )
            )
        pc += 4
    return issues


def format_issues(issues: list[UnsupportedInstruction]) -> str:
    """Render a list of issues as a human-readable warning block."""
    if not issues:
        return ""
    lines = [
        f"Found {len(issues)} unsupported instruction(s) in the selected "
        "code range. BMC will trip illegal-instruction at the first one."
    ]
    for issue in issues:
        lines.append(f"  {issue}")
    lines.append(
        "Suggestions: reduce the code range with --start/--end, or use "
        "model_backend='crotor' if you have C Rotor installed."
    )
    return "\n".join(lines)
