"""Native RISC-V instruction semantics for the Python BTOR2 builder.

The :func:`build_fetch_decode_execute` entry point wires up the transition
relation for the program counter and register file of each core. The scope
covers a representative subset of RV64I (LUI/AUIPC/JAL/JALR, branch set,
I-type arithmetic, R-type arithmetic). Full RV64I/M/A/F/D can be layered on
top by extending :mod:`rotor.btor2.riscv.isa` with additional instruction
definitions — the dispatch framework in this module is generic over them.

This module does *not* claim to reproduce every corner of C Rotor. It exists
to make the native Python backend produce a valid, executable BTOR2 model
for binaries that stay within the supported subset; everything outside that
subset traps to an ``illegal-instruction`` bad property.
"""

from __future__ import annotations

from rotor.btor2.riscv.isa import build_fetch_decode_execute

__all__ = ["build_fetch_decode_execute"]
