"""Top-level RISC-V helpers shared by rotor modules.

So far this namespace exposes the disassembler only. The native BTOR2
machine builder continues to live at :mod:`rotor.btor2.riscv` since it is
tightly coupled to the BTOR2 node layer.
"""

from __future__ import annotations

from rotor.riscv.disasm import disassemble, reg_name, ABI_REGISTERS
from rotor.riscv.support import (
    SUPPORTED_OPCODES,
    UnsupportedInstruction,
    scan_unsupported_instructions,
    format_issues,
)

__all__ = [
    "disassemble",
    "reg_name",
    "ABI_REGISTERS",
    "SUPPORTED_OPCODES",
    "UnsupportedInstruction",
    "scan_unsupported_instructions",
    "format_issues",
]
