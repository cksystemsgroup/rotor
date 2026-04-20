"""Concrete RISC-V simulator for witness reconstruction.

Given the initial register values that the solver produced for a
reachable verdict, the simulator steps the function deterministically
(M5's machine model has no inputs beyond the free initial registers,
so with those fixed the trajectory is unique) and records a
MachineStep per cycle. The caller lifts those steps into a source
trace.

The semantics here must match rotor/btor2/riscv/isa.py exactly — any
divergence would invalidate the witness with respect to the BTOR2
model. Kept side-by-side deliberately; cross-checked by tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from rotor.binary import Function, Instruction, RISCVBinary
from rotor.btor2.riscv.decoder import Decoded, decode

XLEN = 64
MASK = (1 << XLEN) - 1
MASK32 = (1 << 32) - 1
SIGN_BIT = 1 << (XLEN - 1)


@dataclass(frozen=True)
class MachineStep:
    step: int
    pc: int
    word: Optional[int]            # None when pc has no instruction in the function
    decoded: Optional[Decoded]
    registers: tuple[int, ...]     # x0..x31 values at this step (pre-execution)


def simulate(
    binary: RISCVBinary,
    function: Function,
    initial_regs: dict[str, int],
    max_steps: int,
) -> list[MachineStep]:
    """Run the machine from fn.start for up to max_steps cycles."""
    regs = [0] * 32
    for i in range(1, 32):
        regs[i] = initial_regs.get(f"x{i}", 0) & MASK

    instructions: dict[int, Instruction] = {i.pc: i for i in binary.instructions(function)}
    pc = function.start
    steps: list[MachineStep] = []

    for s in range(max_steps + 1):
        inst = instructions.get(pc)
        d = decode(inst.word) if inst is not None else None
        steps.append(
            MachineStep(
                step=s, pc=pc,
                word=inst.word if inst is not None else None,
                decoded=d,
                registers=tuple(regs),
            )
        )
        if inst is None or d is None:
            break                                  # machine left the function
        pc = _step(d, pc, regs)

    return steps


# ---------------------------------------------------------------------------
# Step semantics: dispatch on mnemonic.
# ---------------------------------------------------------------------------

def _step(d: Decoded, pc: int, regs: list[int]) -> int:
    handler = _STEP.get(d.mnem)
    if handler is None:                            # pragma: no cover — decoder filters
        raise AssertionError(f"simulate: unsupported mnem {d.mnem!r}")
    return handler(d, pc, regs)


def _write(regs: list[int], rd: int, value: int) -> None:
    if rd != 0:
        regs[rd] = value & MASK


def _signed64(v: int) -> int:
    v &= MASK
    return v - (1 << XLEN) if v & SIGN_BIT else v


def _signed32(v: int) -> int:
    v &= MASK32
    return v - (1 << 32) if v & 0x80000000 else v


def _sext32(v: int) -> int:
    v &= MASK32
    return v | 0xFFFFFFFF_00000000 if v & 0x80000000 else v


# I-type
def _h_addi(d, pc, regs):
    _write(regs, d.rd, regs[d.rs1] + d.imm)
    return (pc + 4) & MASK


def _h_xori(d, pc, regs):
    _write(regs, d.rd, regs[d.rs1] ^ (d.imm & MASK))
    return (pc + 4) & MASK


def _h_ori(d, pc, regs):
    _write(regs, d.rd, regs[d.rs1] | (d.imm & MASK))
    return (pc + 4) & MASK


def _h_andi(d, pc, regs):
    _write(regs, d.rd, regs[d.rs1] & (d.imm & MASK))
    return (pc + 4) & MASK


def _h_slti(d, pc, regs):
    _write(regs, d.rd, 1 if _signed64(regs[d.rs1]) < d.imm else 0)
    return (pc + 4) & MASK


def _h_sltiu(d, pc, regs):
    _write(regs, d.rd, 1 if regs[d.rs1] < (d.imm & MASK) else 0)
    return (pc + 4) & MASK


def _h_slli(d, pc, regs):
    _write(regs, d.rd, regs[d.rs1] << (d.imm & 63))
    return (pc + 4) & MASK


def _h_srli(d, pc, regs):
    _write(regs, d.rd, regs[d.rs1] >> (d.imm & 63))
    return (pc + 4) & MASK


def _h_srai(d, pc, regs):
    _write(regs, d.rd, _signed64(regs[d.rs1]) >> (d.imm & 63))
    return (pc + 4) & MASK


def _h_addiw(d, pc, regs):
    _write(regs, d.rd, _sext32((regs[d.rs1] + d.imm) & MASK32))
    return (pc + 4) & MASK


def _h_slliw(d, pc, regs):
    _write(regs, d.rd, _sext32((regs[d.rs1] << (d.imm & 31)) & MASK32))
    return (pc + 4) & MASK


def _h_srliw(d, pc, regs):
    _write(regs, d.rd, _sext32(((regs[d.rs1] & MASK32) >> (d.imm & 31)) & MASK32))
    return (pc + 4) & MASK


def _h_sraiw(d, pc, regs):
    lo = _signed32(regs[d.rs1])
    _write(regs, d.rd, _sext32((lo >> (d.imm & 31)) & MASK32))
    return (pc + 4) & MASK


# R-type
def _h_add(d, pc, regs):
    _write(regs, d.rd, regs[d.rs1] + regs[d.rs2])
    return (pc + 4) & MASK


def _h_sub(d, pc, regs):
    _write(regs, d.rd, regs[d.rs1] - regs[d.rs2])
    return (pc + 4) & MASK


def _h_and(d, pc, regs):
    _write(regs, d.rd, regs[d.rs1] & regs[d.rs2])
    return (pc + 4) & MASK


def _h_or(d, pc, regs):
    _write(regs, d.rd, regs[d.rs1] | regs[d.rs2])
    return (pc + 4) & MASK


def _h_xor(d, pc, regs):
    _write(regs, d.rd, regs[d.rs1] ^ regs[d.rs2])
    return (pc + 4) & MASK


def _h_slt(d, pc, regs):
    _write(regs, d.rd, 1 if _signed64(regs[d.rs1]) < _signed64(regs[d.rs2]) else 0)
    return (pc + 4) & MASK


def _h_sltu(d, pc, regs):
    _write(regs, d.rd, 1 if regs[d.rs1] < regs[d.rs2] else 0)
    return (pc + 4) & MASK


def _h_sll(d, pc, regs):
    _write(regs, d.rd, regs[d.rs1] << (regs[d.rs2] & 63))
    return (pc + 4) & MASK


def _h_srl(d, pc, regs):
    _write(regs, d.rd, regs[d.rs1] >> (regs[d.rs2] & 63))
    return (pc + 4) & MASK


def _h_sra(d, pc, regs):
    _write(regs, d.rd, _signed64(regs[d.rs1]) >> (regs[d.rs2] & 63))
    return (pc + 4) & MASK


# OP-32
def _h_addw(d, pc, regs):
    _write(regs, d.rd, _sext32((regs[d.rs1] + regs[d.rs2]) & MASK32))
    return (pc + 4) & MASK


def _h_subw(d, pc, regs):
    _write(regs, d.rd, _sext32((regs[d.rs1] - regs[d.rs2]) & MASK32))
    return (pc + 4) & MASK


def _h_sllw(d, pc, regs):
    _write(regs, d.rd, _sext32((regs[d.rs1] << (regs[d.rs2] & 31)) & MASK32))
    return (pc + 4) & MASK


def _h_srlw(d, pc, regs):
    _write(regs, d.rd, _sext32(((regs[d.rs1] & MASK32) >> (regs[d.rs2] & 31)) & MASK32))
    return (pc + 4) & MASK


def _h_sraw(d, pc, regs):
    lo = _signed32(regs[d.rs1])
    _write(regs, d.rd, _sext32((lo >> (regs[d.rs2] & 31)) & MASK32))
    return (pc + 4) & MASK


# Branches
def _h_beq(d, pc, regs):
    return (pc + d.imm) & MASK if regs[d.rs1] == regs[d.rs2] else (pc + 4) & MASK


def _h_bne(d, pc, regs):
    return (pc + d.imm) & MASK if regs[d.rs1] != regs[d.rs2] else (pc + 4) & MASK


def _h_blt(d, pc, regs):
    return (pc + d.imm) & MASK if _signed64(regs[d.rs1]) < _signed64(regs[d.rs2]) else (pc + 4) & MASK


def _h_bge(d, pc, regs):
    return (pc + d.imm) & MASK if _signed64(regs[d.rs1]) >= _signed64(regs[d.rs2]) else (pc + 4) & MASK


def _h_bltu(d, pc, regs):
    return (pc + d.imm) & MASK if regs[d.rs1] < regs[d.rs2] else (pc + 4) & MASK


def _h_bgeu(d, pc, regs):
    return (pc + d.imm) & MASK if regs[d.rs1] >= regs[d.rs2] else (pc + 4) & MASK


# U / J
def _h_lui(d, pc, regs):
    _write(regs, d.rd, d.imm)
    return (pc + 4) & MASK


def _h_auipc(d, pc, regs):
    _write(regs, d.rd, (pc + d.imm) & MASK)
    return (pc + 4) & MASK


def _h_jal(d, pc, regs):
    _write(regs, d.rd, (pc + 4) & MASK)
    return (pc + d.imm) & MASK


def _h_jalr(d, pc, regs):
    target = ((regs[d.rs1] + d.imm) & MASK) & ~1
    _write(regs, d.rd, (pc + 4) & MASK)
    return target


# Misc
def _h_fence(d, pc, regs):
    return (pc + 4) & MASK


_STEP: dict[str, Callable[[Decoded, int, list[int]], int]] = {
    "addi": _h_addi, "slti": _h_slti, "sltiu": _h_sltiu,
    "xori": _h_xori, "ori": _h_ori, "andi": _h_andi,
    "slli": _h_slli, "srli": _h_srli, "srai": _h_srai,
    "addiw": _h_addiw, "slliw": _h_slliw, "srliw": _h_srliw, "sraiw": _h_sraiw,
    "add": _h_add, "sub": _h_sub, "and": _h_and, "or": _h_or, "xor": _h_xor,
    "slt": _h_slt, "sltu": _h_sltu,
    "sll": _h_sll, "srl": _h_srl, "sra": _h_sra,
    "addw": _h_addw, "subw": _h_subw,
    "sllw": _h_sllw, "srlw": _h_srlw, "sraw": _h_sraw,
    "beq": _h_beq, "bne": _h_bne, "blt": _h_blt,
    "bge": _h_bge, "bltu": _h_bltu, "bgeu": _h_bgeu,
    "lui": _h_lui, "auipc": _h_auipc, "jal": _h_jal, "jalr": _h_jalr,
    "fence": _h_fence,
}
