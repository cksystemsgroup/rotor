"""Concrete RISC-V simulator for witness reconstruction.

Given the initial register values that the solver produced for a
reachable verdict, the simulator steps the function deterministically
(inputs to the M1 machine model are just the free initial registers,
so with those fixed the trajectory is unique) and records a
MachineStep per cycle. The caller lifts those steps into a source
trace.

The semantics here must match rotor/btor2/riscv/isa.py exactly — any
divergence would invalidate the witness with respect to the BTOR2
model. Kept side-by-side deliberately; cross-checked by tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rotor.binary import Function, Instruction, RISCVBinary
from rotor.btor2.riscv.decoder import Decoded, decode

XLEN = 64
MASK = (1 << XLEN) - 1
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
    """Run the machine from fn.start for up to max_steps cycles.

    The register file is seeded from initial_regs; any xN not present
    defaults to 0. pc always starts at function.start (the BTOR2 model's
    init constraint), regardless of any "pc" entry in initial_regs.
    """
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
            break                               # machine left the function
        pc = _step(d, pc, regs)

    return steps


def _step(d: Decoded, pc: int, regs: list[int]) -> int:
    next_pc = (pc + 4) & MASK

    if d.mnem == "addi":
        _write(regs, d.rd, (regs[d.rs1] + (d.imm & MASK)) & MASK)
    elif d.mnem == "addw":
        lo1 = regs[d.rs1] & 0xFFFFFFFF
        lo2 = regs[d.rs2] & 0xFFFFFFFF
        _write(regs, d.rd, _sext32((lo1 + lo2) & 0xFFFFFFFF))
    elif d.mnem == "sub":
        _write(regs, d.rd, (regs[d.rs1] - regs[d.rs2]) & MASK)
    elif d.mnem == "sltu":
        _write(regs, d.rd, 1 if regs[d.rs1] < regs[d.rs2] else 0)
    elif d.mnem == "blt":
        if _signed64(regs[d.rs1]) < _signed64(regs[d.rs2]):
            next_pc = (pc + d.imm) & MASK
    elif d.mnem == "jalr":
        target = ((regs[d.rs1] + (d.imm & MASK)) & MASK) & ~1
        _write(regs, d.rd, (pc + 4) & MASK)
        next_pc = target
    else:                                       # pragma: no cover — decoder filters
        raise AssertionError(f"simulate: unsupported mnem {d.mnem!r}")

    return next_pc


def _write(regs: list[int], rd: int, value: int) -> None:
    if rd != 0:
        regs[rd] = value & MASK


def _sext32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v | 0xFFFFFFFF_00000000 if v & 0x80000000 else v


def _signed64(v: int) -> int:
    v &= MASK
    return v - (1 << XLEN) if v & SIGN_BIT else v
