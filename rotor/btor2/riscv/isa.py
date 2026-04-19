"""Per-instruction semantic lowering to BTOR2 nodes.

Each lowering returns (writes, next_pc) where:
    writes   — dict mapping destination register index (1..31) to its
               next-state expression.  Writes to x0 are dropped.
    next_pc  — next-PC expression for this instruction.

The caller (builder) combines these across all instructions in a
function via an ITE dispatching on the current PC.
"""

from __future__ import annotations

from rotor.btor2.nodes import Model, Node, Sort
from rotor.btor2.riscv.decoder import Decoded

BV1 = Sort(1)
BV32 = Sort(32)
BV64 = Sort(64)
MASK64 = (1 << 64) - 1


def lower(
    d: Decoded,
    pc_value: int,
    m: Model,
    regs: list[Node],
    pc: Node,
) -> tuple[dict[int, Node], Node]:
    four = m.const(BV64, 4)
    pc_plus_4 = m.op("add", BV64, pc, four)

    writes: dict[int, Node] = {}
    next_pc: Node = pc_plus_4

    if d.mnem == "addi":
        imm = m.const(BV64, d.imm & MASK64)
        result = m.op("add", BV64, regs[d.rs1], imm)
        _write(writes, d.rd, result)

    elif d.mnem == "addw":
        lo1 = m.slice(regs[d.rs1], 31, 0)
        lo2 = m.slice(regs[d.rs2], 31, 0)
        sum32 = m.op("add", BV32, lo1, lo2)
        result = m.sext(sum32, 32)
        _write(writes, d.rd, result)

    elif d.mnem == "sub":
        result = m.op("sub", BV64, regs[d.rs1], regs[d.rs2])
        _write(writes, d.rd, result)

    elif d.mnem == "sltu":
        lt = m.op("ult", BV1, regs[d.rs1], regs[d.rs2])
        one = m.const(BV64, 1)
        zero = m.const(BV64, 0)
        result = m.ite(lt, one, zero)
        _write(writes, d.rd, result)

    elif d.mnem == "blt":
        lt = m.op("slt", BV1, regs[d.rs1], regs[d.rs2])
        target = m.const(BV64, (pc_value + d.imm) & MASK64)
        next_pc = m.ite(lt, target, pc_plus_4)

    elif d.mnem == "jalr":
        imm = m.const(BV64, d.imm & MASK64)
        raw = m.op("add", BV64, regs[d.rs1], imm)
        mask = m.const(BV64, MASK64 ^ 1)
        next_pc = m.op("and", BV64, raw, mask)
        _write(writes, d.rd, pc_plus_4)   # link register

    else:                                  # pragma: no cover — decoder filters
        raise AssertionError(f"lower: unsupported mnem {d.mnem!r}")

    return writes, next_pc


def _write(writes: dict[int, Node], rd: int, expr: Node) -> None:
    if rd == 0:
        return                              # x0 is hard-wired to zero
    writes[rd] = expr
