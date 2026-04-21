"""Concrete RISC-V simulator for witness reconstruction.

Given the initial register values that the solver produced for a
reachable verdict, the simulator steps the function deterministically
and records a MachineStep per cycle. The caller lifts those steps
into a source trace.

Memory (M6) is a sparse `dict[int, int]` byte mirror. It is seeded
with every PT_LOAD byte from the ELF; reads to uninitialized bytes
return 0, which matches the common case for freshly-mapped BSS or
stack pages once the SMT solver has picked concrete values for the
trajectory.

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
MASK8 = 0xFF
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
    initial_mem: Optional[dict[int, int]] = None,
) -> list[MachineStep]:
    """Run the machine from fn.start for up to max_steps cycles."""
    regs = [0] * 32
    for i in range(1, 32):
        regs[i] = initial_regs.get(f"x{i}", 0) & MASK

    mem: dict[int, int] = {}
    for vaddr, byte in binary.loadable_bytes():
        mem[vaddr] = byte & MASK8
    if initial_mem:
        for addr, byte in initial_mem.items():
            mem[addr & MASK] = byte & MASK8

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
        pc = _step(d, pc, regs, mem)

    return steps


# ---------------------------------------------------------------------------
# Step semantics: dispatch on mnemonic.
# ---------------------------------------------------------------------------

def _step(d: Decoded, pc: int, regs: list[int], mem: Optional[dict[int, int]] = None) -> int:
    handler = _STEP.get(d.mnem)
    if handler is None:                            # pragma: no cover — decoder filters
        raise AssertionError(f"simulate: unsupported mnem {d.mnem!r}")
    return handler(d, pc, regs, mem if mem is not None else {})


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
def _h_addi(d, pc, regs, mem):
    _write(regs, d.rd, regs[d.rs1] + d.imm)
    return (pc + 4) & MASK


def _h_xori(d, pc, regs, mem):
    _write(regs, d.rd, regs[d.rs1] ^ (d.imm & MASK))
    return (pc + 4) & MASK


def _h_ori(d, pc, regs, mem):
    _write(regs, d.rd, regs[d.rs1] | (d.imm & MASK))
    return (pc + 4) & MASK


def _h_andi(d, pc, regs, mem):
    _write(regs, d.rd, regs[d.rs1] & (d.imm & MASK))
    return (pc + 4) & MASK


def _h_slti(d, pc, regs, mem):
    _write(regs, d.rd, 1 if _signed64(regs[d.rs1]) < d.imm else 0)
    return (pc + 4) & MASK


def _h_sltiu(d, pc, regs, mem):
    _write(regs, d.rd, 1 if regs[d.rs1] < (d.imm & MASK) else 0)
    return (pc + 4) & MASK


def _h_slli(d, pc, regs, mem):
    _write(regs, d.rd, regs[d.rs1] << (d.imm & 63))
    return (pc + 4) & MASK


def _h_srli(d, pc, regs, mem):
    _write(regs, d.rd, regs[d.rs1] >> (d.imm & 63))
    return (pc + 4) & MASK


def _h_srai(d, pc, regs, mem):
    _write(regs, d.rd, _signed64(regs[d.rs1]) >> (d.imm & 63))
    return (pc + 4) & MASK


def _h_addiw(d, pc, regs, mem):
    _write(regs, d.rd, _sext32((regs[d.rs1] + d.imm) & MASK32))
    return (pc + 4) & MASK


def _h_slliw(d, pc, regs, mem):
    _write(regs, d.rd, _sext32((regs[d.rs1] << (d.imm & 31)) & MASK32))
    return (pc + 4) & MASK


def _h_srliw(d, pc, regs, mem):
    _write(regs, d.rd, _sext32(((regs[d.rs1] & MASK32) >> (d.imm & 31)) & MASK32))
    return (pc + 4) & MASK


def _h_sraiw(d, pc, regs, mem):
    lo = _signed32(regs[d.rs1])
    _write(regs, d.rd, _sext32((lo >> (d.imm & 31)) & MASK32))
    return (pc + 4) & MASK


# R-type
def _h_add(d, pc, regs, mem):
    _write(regs, d.rd, regs[d.rs1] + regs[d.rs2])
    return (pc + 4) & MASK


def _h_sub(d, pc, regs, mem):
    _write(regs, d.rd, regs[d.rs1] - regs[d.rs2])
    return (pc + 4) & MASK


def _h_and(d, pc, regs, mem):
    _write(regs, d.rd, regs[d.rs1] & regs[d.rs2])
    return (pc + 4) & MASK


def _h_or(d, pc, regs, mem):
    _write(regs, d.rd, regs[d.rs1] | regs[d.rs2])
    return (pc + 4) & MASK


def _h_xor(d, pc, regs, mem):
    _write(regs, d.rd, regs[d.rs1] ^ regs[d.rs2])
    return (pc + 4) & MASK


def _h_slt(d, pc, regs, mem):
    _write(regs, d.rd, 1 if _signed64(regs[d.rs1]) < _signed64(regs[d.rs2]) else 0)
    return (pc + 4) & MASK


def _h_sltu(d, pc, regs, mem):
    _write(regs, d.rd, 1 if regs[d.rs1] < regs[d.rs2] else 0)
    return (pc + 4) & MASK


def _h_sll(d, pc, regs, mem):
    _write(regs, d.rd, regs[d.rs1] << (regs[d.rs2] & 63))
    return (pc + 4) & MASK


def _h_srl(d, pc, regs, mem):
    _write(regs, d.rd, regs[d.rs1] >> (regs[d.rs2] & 63))
    return (pc + 4) & MASK


def _h_sra(d, pc, regs, mem):
    _write(regs, d.rd, _signed64(regs[d.rs1]) >> (regs[d.rs2] & 63))
    return (pc + 4) & MASK


# OP-32
def _h_addw(d, pc, regs, mem):
    _write(regs, d.rd, _sext32((regs[d.rs1] + regs[d.rs2]) & MASK32))
    return (pc + 4) & MASK


def _h_subw(d, pc, regs, mem):
    _write(regs, d.rd, _sext32((regs[d.rs1] - regs[d.rs2]) & MASK32))
    return (pc + 4) & MASK


def _h_sllw(d, pc, regs, mem):
    _write(regs, d.rd, _sext32((regs[d.rs1] << (regs[d.rs2] & 31)) & MASK32))
    return (pc + 4) & MASK


def _h_srlw(d, pc, regs, mem):
    _write(regs, d.rd, _sext32(((regs[d.rs1] & MASK32) >> (regs[d.rs2] & 31)) & MASK32))
    return (pc + 4) & MASK


def _h_sraw(d, pc, regs, mem):
    lo = _signed32(regs[d.rs1])
    _write(regs, d.rd, _sext32((lo >> (regs[d.rs2] & 31)) & MASK32))
    return (pc + 4) & MASK


# M extension — mirrors rotor/btor2/riscv/isa.py's RISC-V spec edge
# cases. Must stay in lockstep with the BTOR2 lowering; any divergence
# invalidates witness replay against the model.

def _trunc_div(a: int, b: int) -> int:
    """Truncated signed division (rounds toward zero)."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _h_mul(d, pc, regs, mem):
    _write(regs, d.rd, (regs[d.rs1] * regs[d.rs2]) & MASK)
    return (pc + 4) & MASK


def _h_mulh_common(d, regs, *, sign_a: bool, sign_b: bool) -> int:
    a = _signed64(regs[d.rs1]) if sign_a else regs[d.rs1] & MASK
    b = _signed64(regs[d.rs2]) if sign_b else regs[d.rs2] & MASK
    prod = a * b
    return (prod >> 64) & MASK


def _h_mulh(d, pc, regs, mem):
    _write(regs, d.rd, _h_mulh_common(d, regs, sign_a=True, sign_b=True))
    return (pc + 4) & MASK


def _h_mulhsu(d, pc, regs, mem):
    _write(regs, d.rd, _h_mulh_common(d, regs, sign_a=True, sign_b=False))
    return (pc + 4) & MASK


def _h_mulhu(d, pc, regs, mem):
    _write(regs, d.rd, _h_mulh_common(d, regs, sign_a=False, sign_b=False))
    return (pc + 4) & MASK


def _h_divu(d, pc, regs, mem):
    a, b = regs[d.rs1], regs[d.rs2]
    _write(regs, d.rd, MASK if b == 0 else (a // b) & MASK)
    return (pc + 4) & MASK


def _h_remu(d, pc, regs, mem):
    a, b = regs[d.rs1], regs[d.rs2]
    _write(regs, d.rd, a if b == 0 else (a % b) & MASK)
    return (pc + 4) & MASK


def _h_div(d, pc, regs, mem):
    a, b = _signed64(regs[d.rs1]), _signed64(regs[d.rs2])
    if b == 0:
        result = MASK                                # -1
    elif a == -(1 << 63) and b == -1:
        result = 1 << 63                              # INT_MIN overflow
    else:
        result = _trunc_div(a, b) & MASK
    _write(regs, d.rd, result)
    return (pc + 4) & MASK


def _h_rem(d, pc, regs, mem):
    a, b = _signed64(regs[d.rs1]), _signed64(regs[d.rs2])
    if b == 0:
        result = regs[d.rs1]                          # dividend
    elif a == -(1 << 63) and b == -1:
        result = 0                                    # overflow
    else:
        result = (a - _trunc_div(a, b) * b) & MASK
    _write(regs, d.rd, result)
    return (pc + 4) & MASK


# OP-32 M (sign-extended 32-bit results)

def _h_mulw(d, pc, regs, mem):
    _write(regs, d.rd, _sext32((regs[d.rs1] * regs[d.rs2]) & MASK32))
    return (pc + 4) & MASK


def _h_divuw(d, pc, regs, mem):
    a, b = regs[d.rs1] & MASK32, regs[d.rs2] & MASK32
    r32 = MASK32 if b == 0 else (a // b) & MASK32
    _write(regs, d.rd, _sext32(r32))
    return (pc + 4) & MASK


def _h_remuw(d, pc, regs, mem):
    a, b = regs[d.rs1] & MASK32, regs[d.rs2] & MASK32
    r32 = a if b == 0 else (a % b) & MASK32
    _write(regs, d.rd, _sext32(r32))
    return (pc + 4) & MASK


def _h_divw(d, pc, regs, mem):
    a, b = _signed32(regs[d.rs1]), _signed32(regs[d.rs2])
    if b == 0:
        r32 = MASK32
    elif a == -(1 << 31) and b == -1:
        r32 = 1 << 31
    else:
        r32 = _trunc_div(a, b) & MASK32
    _write(regs, d.rd, _sext32(r32))
    return (pc + 4) & MASK


def _h_remw(d, pc, regs, mem):
    a, b = _signed32(regs[d.rs1]), _signed32(regs[d.rs2])
    if b == 0:
        r32 = regs[d.rs1] & MASK32
    elif a == -(1 << 31) and b == -1:
        r32 = 0
    else:
        r32 = (a - _trunc_div(a, b) * b) & MASK32
    _write(regs, d.rd, _sext32(r32))
    return (pc + 4) & MASK


# Branches
def _h_beq(d, pc, regs, mem):
    return (pc + d.imm) & MASK if regs[d.rs1] == regs[d.rs2] else (pc + 4) & MASK


def _h_bne(d, pc, regs, mem):
    return (pc + d.imm) & MASK if regs[d.rs1] != regs[d.rs2] else (pc + 4) & MASK


def _h_blt(d, pc, regs, mem):
    return (pc + d.imm) & MASK if _signed64(regs[d.rs1]) < _signed64(regs[d.rs2]) else (pc + 4) & MASK


def _h_bge(d, pc, regs, mem):
    return (pc + d.imm) & MASK if _signed64(regs[d.rs1]) >= _signed64(regs[d.rs2]) else (pc + 4) & MASK


def _h_bltu(d, pc, regs, mem):
    return (pc + d.imm) & MASK if regs[d.rs1] < regs[d.rs2] else (pc + 4) & MASK


def _h_bgeu(d, pc, regs, mem):
    return (pc + d.imm) & MASK if regs[d.rs1] >= regs[d.rs2] else (pc + 4) & MASK


# U / J
def _h_lui(d, pc, regs, mem):
    _write(regs, d.rd, d.imm)
    return (pc + 4) & MASK


def _h_auipc(d, pc, regs, mem):
    _write(regs, d.rd, (pc + d.imm) & MASK)
    return (pc + 4) & MASK


def _h_jal(d, pc, regs, mem):
    _write(regs, d.rd, (pc + 4) & MASK)
    return (pc + d.imm) & MASK


def _h_jalr(d, pc, regs, mem):
    target = ((regs[d.rs1] + d.imm) & MASK) & ~1
    _write(regs, d.rd, (pc + 4) & MASK)
    return target


# Loads / stores. Byte-addressed memory: little-endian multi-byte access.
def _read_bytes(mem: dict[int, int], addr: int, nbytes: int) -> int:
    value = 0
    for i in range(nbytes):
        b = mem.get((addr + i) & MASK, 0) & MASK8
        value |= b << (8 * i)
    return value


def _write_bytes(mem: dict[int, int], addr: int, value: int, nbytes: int) -> None:
    for i in range(nbytes):
        mem[(addr + i) & MASK] = (value >> (8 * i)) & MASK8


def _sext_n(value: int, nbits: int) -> int:
    sign = 1 << (nbits - 1)
    mask = (1 << nbits) - 1
    value &= mask
    return (value - (1 << nbits)) & MASK if value & sign else value


def _h_lb(d, pc, regs, mem):
    v = _read_bytes(mem, (regs[d.rs1] + d.imm) & MASK, 1)
    _write(regs, d.rd, _sext_n(v, 8))
    return (pc + 4) & MASK


def _h_lh(d, pc, regs, mem):
    v = _read_bytes(mem, (regs[d.rs1] + d.imm) & MASK, 2)
    _write(regs, d.rd, _sext_n(v, 16))
    return (pc + 4) & MASK


def _h_lw(d, pc, regs, mem):
    v = _read_bytes(mem, (regs[d.rs1] + d.imm) & MASK, 4)
    _write(regs, d.rd, _sext_n(v, 32))
    return (pc + 4) & MASK


def _h_ld(d, pc, regs, mem):
    v = _read_bytes(mem, (regs[d.rs1] + d.imm) & MASK, 8)
    _write(regs, d.rd, v)
    return (pc + 4) & MASK


def _h_lbu(d, pc, regs, mem):
    _write(regs, d.rd, _read_bytes(mem, (regs[d.rs1] + d.imm) & MASK, 1))
    return (pc + 4) & MASK


def _h_lhu(d, pc, regs, mem):
    _write(regs, d.rd, _read_bytes(mem, (regs[d.rs1] + d.imm) & MASK, 2))
    return (pc + 4) & MASK


def _h_lwu(d, pc, regs, mem):
    _write(regs, d.rd, _read_bytes(mem, (regs[d.rs1] + d.imm) & MASK, 4))
    return (pc + 4) & MASK


def _h_sb(d, pc, regs, mem):
    _write_bytes(mem, (regs[d.rs1] + d.imm) & MASK, regs[d.rs2], 1)
    return (pc + 4) & MASK


def _h_sh(d, pc, regs, mem):
    _write_bytes(mem, (regs[d.rs1] + d.imm) & MASK, regs[d.rs2], 2)
    return (pc + 4) & MASK


def _h_sw(d, pc, regs, mem):
    _write_bytes(mem, (regs[d.rs1] + d.imm) & MASK, regs[d.rs2], 4)
    return (pc + 4) & MASK


def _h_sd(d, pc, regs, mem):
    _write_bytes(mem, (regs[d.rs1] + d.imm) & MASK, regs[d.rs2], 8)
    return (pc + 4) & MASK


# Misc
def _h_fence(d, pc, regs, mem):
    return (pc + 4) & MASK


def _h_halt(d, pc, regs, mem):
    """ECALL / EBREAK: self-loop at the instruction (see isa.py::_halt)."""
    return pc


_STEP: dict[str, Callable[[Decoded, int, list[int], dict[int, int]], int]] = {
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
    "lb":  _h_lb,  "lh":  _h_lh,  "lw":  _h_lw,  "ld":  _h_ld,
    "lbu": _h_lbu, "lhu": _h_lhu, "lwu": _h_lwu,
    "sb":  _h_sb,  "sh":  _h_sh,  "sw":  _h_sw,  "sd":  _h_sd,
    # M extension
    "mul": _h_mul, "mulh": _h_mulh, "mulhsu": _h_mulhsu, "mulhu": _h_mulhu,
    "div": _h_div, "divu": _h_divu, "rem": _h_rem, "remu": _h_remu,
    "mulw": _h_mulw, "divw": _h_divw, "divuw": _h_divuw,
    "remw": _h_remw, "remuw": _h_remuw,
    "fence": _h_fence,
    "ecall": _h_halt, "ebreak": _h_halt,
}
