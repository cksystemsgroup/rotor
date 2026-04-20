"""Per-instruction semantic lowering to BTOR2 nodes.

Each lowering takes a Decoded instruction, the PC of the instruction
itself (known statically — we are inside a pc-dispatch arm of the
top-level transition ITE), the Model under construction, and the
current register state nodes. It returns:

    writes   — dict mapping destination register index (1..31) to its
               next-state expression. Writes to x0 are dropped.
    next_pc  — next-PC expression for this instruction.

Because pc_value is known at lowering time, "pc + 4" and "pc + imm"
are folded to constants in the emitted BTOR2 — no synthesized add
nodes for fall-through PCs or branch targets.

The dispatch table at the bottom of this file keys on the mnemonic
string in Decoded; new instructions are added by writing a small
lowering function and registering it.
"""

from __future__ import annotations

from typing import Callable

from rotor.btor2.nodes import Model, Node, Sort
from rotor.btor2.riscv.decoder import Decoded

BV1 = Sort(1)
BV32 = Sort(32)
BV64 = Sort(64)
MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1

LowerFn = Callable[[Decoded, int, Model, list[Node]], tuple[dict[int, Node], Node]]


def lower(d: Decoded, pc_value: int, m: Model, regs: list[Node]) -> tuple[dict[int, Node], Node]:
    fn = DISPATCH.get(d.mnem)
    if fn is None:                                 # pragma: no cover — decoder filters
        raise AssertionError(f"lower: unsupported mnem {d.mnem!r}")
    return fn(d, pc_value, m, regs)


def _write(writes: dict[int, Node], rd: int, expr: Node) -> None:
    if rd != 0:
        writes[rd] = expr


def _fall(pc_value: int, m: Model) -> Node:
    return m.const(BV64, (pc_value + 4) & MASK64)


# ---------------------------------------------------------------------------
# I-type arithmetic / logic.
# ---------------------------------------------------------------------------

def _addi(d, pc_value, m, regs):
    imm = m.const(BV64, d.imm & MASK64)
    result = m.op("add", BV64, regs[d.rs1], imm)
    writes: dict[int, Node] = {}
    _write(writes, d.rd, result)
    return writes, _fall(pc_value, m)


def _xori(d, pc_value, m, regs):
    imm = m.const(BV64, d.imm & MASK64)
    return _i_writes(m, d, m.op("xor", BV64, regs[d.rs1], imm)), _fall(pc_value, m)


def _ori(d, pc_value, m, regs):
    imm = m.const(BV64, d.imm & MASK64)
    return _i_writes(m, d, m.op("or", BV64, regs[d.rs1], imm)), _fall(pc_value, m)


def _andi(d, pc_value, m, regs):
    imm = m.const(BV64, d.imm & MASK64)
    return _i_writes(m, d, m.op("and", BV64, regs[d.rs1], imm)), _fall(pc_value, m)


def _slti(d, pc_value, m, regs):
    imm = m.const(BV64, d.imm & MASK64)
    cond = m.op("slt", BV1, regs[d.rs1], imm)
    return _i_writes(m, d, _bool_to_bv64(m, cond)), _fall(pc_value, m)


def _sltiu(d, pc_value, m, regs):
    imm = m.const(BV64, d.imm & MASK64)
    cond = m.op("ult", BV1, regs[d.rs1], imm)
    return _i_writes(m, d, _bool_to_bv64(m, cond)), _fall(pc_value, m)


def _slli(d, pc_value, m, regs):
    shamt = m.const(BV64, d.imm & 63)
    return _i_writes(m, d, m.op("sll", BV64, regs[d.rs1], shamt)), _fall(pc_value, m)


def _srli(d, pc_value, m, regs):
    shamt = m.const(BV64, d.imm & 63)
    return _i_writes(m, d, m.op("srl", BV64, regs[d.rs1], shamt)), _fall(pc_value, m)


def _srai(d, pc_value, m, regs):
    shamt = m.const(BV64, d.imm & 63)
    return _i_writes(m, d, m.op("sra", BV64, regs[d.rs1], shamt)), _fall(pc_value, m)


# ---------------------------------------------------------------------------
# OP-IMM-32 (32-bit immediate ops, sign-extend result to 64).
# ---------------------------------------------------------------------------

def _addiw(d, pc_value, m, regs):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    imm32 = m.const(BV32, d.imm & MASK32)
    sum32 = m.op("add", BV32, lo1, imm32)
    return _i_writes(m, d, m.sext(sum32, 32)), _fall(pc_value, m)


def _slliw(d, pc_value, m, regs):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    shamt = m.const(BV32, d.imm & 31)
    sh32 = m.op("sll", BV32, lo1, shamt)
    return _i_writes(m, d, m.sext(sh32, 32)), _fall(pc_value, m)


def _srliw(d, pc_value, m, regs):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    shamt = m.const(BV32, d.imm & 31)
    sh32 = m.op("srl", BV32, lo1, shamt)
    return _i_writes(m, d, m.sext(sh32, 32)), _fall(pc_value, m)


def _sraiw(d, pc_value, m, regs):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    shamt = m.const(BV32, d.imm & 31)
    sh32 = m.op("sra", BV32, lo1, shamt)
    return _i_writes(m, d, m.sext(sh32, 32)), _fall(pc_value, m)


# ---------------------------------------------------------------------------
# R-type (64-bit).
# ---------------------------------------------------------------------------

def _add(d, pc_value, m, regs):
    return _i_writes(m, d, m.op("add", BV64, regs[d.rs1], regs[d.rs2])), _fall(pc_value, m)


def _sub(d, pc_value, m, regs):
    return _i_writes(m, d, m.op("sub", BV64, regs[d.rs1], regs[d.rs2])), _fall(pc_value, m)


def _and(d, pc_value, m, regs):
    return _i_writes(m, d, m.op("and", BV64, regs[d.rs1], regs[d.rs2])), _fall(pc_value, m)


def _or(d, pc_value, m, regs):
    return _i_writes(m, d, m.op("or", BV64, regs[d.rs1], regs[d.rs2])), _fall(pc_value, m)


def _xor(d, pc_value, m, regs):
    return _i_writes(m, d, m.op("xor", BV64, regs[d.rs1], regs[d.rs2])), _fall(pc_value, m)


def _slt(d, pc_value, m, regs):
    cond = m.op("slt", BV1, regs[d.rs1], regs[d.rs2])
    return _i_writes(m, d, _bool_to_bv64(m, cond)), _fall(pc_value, m)


def _sltu(d, pc_value, m, regs):
    cond = m.op("ult", BV1, regs[d.rs1], regs[d.rs2])
    return _i_writes(m, d, _bool_to_bv64(m, cond)), _fall(pc_value, m)


def _sll(d, pc_value, m, regs):
    shamt = m.op("and", BV64, regs[d.rs2], m.const(BV64, 63))
    return _i_writes(m, d, m.op("sll", BV64, regs[d.rs1], shamt)), _fall(pc_value, m)


def _srl(d, pc_value, m, regs):
    shamt = m.op("and", BV64, regs[d.rs2], m.const(BV64, 63))
    return _i_writes(m, d, m.op("srl", BV64, regs[d.rs1], shamt)), _fall(pc_value, m)


def _sra(d, pc_value, m, regs):
    shamt = m.op("and", BV64, regs[d.rs2], m.const(BV64, 63))
    return _i_writes(m, d, m.op("sra", BV64, regs[d.rs1], shamt)), _fall(pc_value, m)


# ---------------------------------------------------------------------------
# OP-32 (R-type, 32-bit).
# ---------------------------------------------------------------------------

def _addw(d, pc_value, m, regs):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    sum32 = m.op("add", BV32, lo1, lo2)
    return _i_writes(m, d, m.sext(sum32, 32)), _fall(pc_value, m)


def _subw(d, pc_value, m, regs):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    diff32 = m.op("sub", BV32, lo1, lo2)
    return _i_writes(m, d, m.sext(diff32, 32)), _fall(pc_value, m)


def _sllw(d, pc_value, m, regs):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    shamt = m.op("and", BV32, lo2, m.const(BV32, 31))
    sh32 = m.op("sll", BV32, lo1, shamt)
    return _i_writes(m, d, m.sext(sh32, 32)), _fall(pc_value, m)


def _srlw(d, pc_value, m, regs):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    shamt = m.op("and", BV32, lo2, m.const(BV32, 31))
    sh32 = m.op("srl", BV32, lo1, shamt)
    return _i_writes(m, d, m.sext(sh32, 32)), _fall(pc_value, m)


def _sraw(d, pc_value, m, regs):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    shamt = m.op("and", BV32, lo2, m.const(BV32, 31))
    sh32 = m.op("sra", BV32, lo1, shamt)
    return _i_writes(m, d, m.sext(sh32, 32)), _fall(pc_value, m)


# ---------------------------------------------------------------------------
# Branches.
# ---------------------------------------------------------------------------

def _branch(opname: str, signed: bool):
    def lower_fn(d, pc_value, m, regs):
        cond = m.op(opname, BV1, regs[d.rs1], regs[d.rs2])
        target = m.const(BV64, (pc_value + d.imm) & MASK64)
        fall = _fall(pc_value, m)
        return {}, m.ite(cond, target, fall)
    return lower_fn


def _beq(d, pc_value, m, regs):
    cond = m.op("eq", BV1, regs[d.rs1], regs[d.rs2])
    return {}, m.ite(cond, m.const(BV64, (pc_value + d.imm) & MASK64), _fall(pc_value, m))


def _bne(d, pc_value, m, regs):
    cond = m.op("neq", BV1, regs[d.rs1], regs[d.rs2])
    return {}, m.ite(cond, m.const(BV64, (pc_value + d.imm) & MASK64), _fall(pc_value, m))


_blt  = _branch("slt", signed=True)
_bge  = _branch("sgte", signed=True)
_bltu = _branch("ult", signed=False)
_bgeu = _branch("ugte", signed=False)


# ---------------------------------------------------------------------------
# U/J/JALR.
# ---------------------------------------------------------------------------

def _lui(d, pc_value, m, regs):
    # Decoder already sign-extended imm to 64 bits.
    return _i_writes(m, d, m.const(BV64, d.imm & MASK64)), _fall(pc_value, m)


def _auipc(d, pc_value, m, regs):
    return _i_writes(m, d, m.const(BV64, (pc_value + d.imm) & MASK64)), _fall(pc_value, m)


def _jal(d, pc_value, m, regs):
    link = m.const(BV64, (pc_value + 4) & MASK64)
    target = m.const(BV64, (pc_value + d.imm) & MASK64)
    return _i_writes(m, d, link), target


def _jalr(d, pc_value, m, regs):
    link = m.const(BV64, (pc_value + 4) & MASK64)
    imm = m.const(BV64, d.imm & MASK64)
    raw = m.op("add", BV64, regs[d.rs1], imm)
    target = m.op("and", BV64, raw, m.const(BV64, MASK64 ^ 1))
    return _i_writes(m, d, link), target


# ---------------------------------------------------------------------------
# Misc.
# ---------------------------------------------------------------------------

def _fence(d, pc_value, m, regs):
    return {}, _fall(pc_value, m)                  # safe approximation: no-op


# ---------------------------------------------------------------------------

def _i_writes(m: Model, d: Decoded, result: Node) -> dict[int, Node]:
    writes: dict[int, Node] = {}
    _write(writes, d.rd, result)
    return writes


def _bool_to_bv64(m: Model, cond: Node) -> Node:
    return m.ite(cond, m.const(BV64, 1), m.const(BV64, 0))


# ---------------------------------------------------------------------------
# Dispatch table.
# ---------------------------------------------------------------------------

DISPATCH: dict[str, LowerFn] = {
    # OP-IMM
    "addi":  _addi,
    "slti":  _slti,
    "sltiu": _sltiu,
    "xori":  _xori,
    "ori":   _ori,
    "andi":  _andi,
    "slli":  _slli,
    "srli":  _srli,
    "srai":  _srai,
    # OP-IMM-32
    "addiw": _addiw,
    "slliw": _slliw,
    "srliw": _srliw,
    "sraiw": _sraiw,
    # OP
    "add":   _add,
    "sub":   _sub,
    "and":   _and,
    "or":    _or,
    "xor":   _xor,
    "slt":   _slt,
    "sltu":  _sltu,
    "sll":   _sll,
    "srl":   _srl,
    "sra":   _sra,
    # OP-32
    "addw":  _addw,
    "subw":  _subw,
    "sllw":  _sllw,
    "srlw":  _srlw,
    "sraw":  _sraw,
    # BRANCH
    "beq":   _beq,
    "bne":   _bne,
    "blt":   _blt,
    "bge":   _bge,
    "bltu":  _bltu,
    "bgeu":  _bgeu,
    # U / J
    "lui":   _lui,
    "auipc": _auipc,
    "jal":   _jal,
    "jalr":  _jalr,
    # Misc
    "fence": _fence,
}
