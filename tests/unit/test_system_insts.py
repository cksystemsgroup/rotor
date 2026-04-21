"""ECALL / EBREAK decoder + witness + lowering coverage.

These instructions are modeled as "halt": the PC self-loops at the
instruction, no register writes, no memory writes. That is the most
honest semantics without a proper syscall / debug-trap model, and it
keeps reachability of any post-halt PC correctly `unreachable`.
"""

from __future__ import annotations

from pathlib import Path

from rotor.btor2.builder import build_reach
from rotor.btor2.riscv.decoder import decode
from rotor.ir.spec import ReachSpec
from rotor.witness import _STEP


ECALL  = 0b000000000000_00000_000_00000_1110011
EBREAK = 0b000000000001_00000_000_00000_1110011


def test_decode_ecall() -> None:
    d = decode(ECALL)
    assert d is not None and d.mnem == "ecall"
    assert d.rd == 0 and d.rs1 == 0 and d.rs2 == 0 and d.imm == 0


def test_decode_ebreak() -> None:
    d = decode(EBREAK)
    assert d is not None and d.mnem == "ebreak"


def test_decode_rejects_nonzero_rd_or_rs1_as_system() -> None:
    # Defensive: CSR and other SYSTEM-opcode instructions must not
    # decode as ecall/ebreak just because funct3 == 0.
    with_rd = ECALL | (5 << 7)
    with_rs1 = ECALL | (5 << 15)
    assert decode(with_rd) is None
    assert decode(with_rs1) is None


def test_decode_rejects_unsupported_system_imms() -> None:
    # Only imm=0 (ecall) and imm=1 (ebreak) are supported; everything
    # else (CSR instructions, sret, mret, wfi) must return None.
    for bad_imm in (2, 0x302, 0x100):
        word = (bad_imm << 20) | (0b000 << 12) | 0b1110011
        assert decode(word) is None


def test_witness_ecall_self_loops() -> None:
    d = decode(ECALL)
    regs = [0] * 32
    next_pc = _STEP["ecall"](d, 0x1000, regs, {})
    assert next_pc == 0x1000                       # stuck at the ecall


def test_witness_ebreak_self_loops() -> None:
    d = decode(EBREAK)
    regs = [0] * 32
    next_pc = _STEP["ebreak"](d, 0x2000, regs, {})
    assert next_pc == 0x2000
