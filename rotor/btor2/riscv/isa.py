"""Per-instruction semantic lowering to BTOR2 nodes.

Each lowering takes a Decoded instruction, the PC of the instruction
itself (known statically — we are inside a pc-dispatch arm of the
top-level transition ITE), the Model under construction, and the
current register state nodes plus the current memory node. It returns:

    writes    — dict mapping destination register index (1..31) to its
                next-state expression. Writes to x0 are dropped.
    next_pc   — next-PC expression for this instruction.
    next_mem  — memory array after this instruction, or None when the
                instruction does not touch memory.

Because pc_value is known at lowering time, "pc + 4" and "pc + imm"
are folded to constants in the emitted BTOR2 — no synthesized add
nodes for fall-through PCs or branch targets.

The dispatch table at the bottom of this file keys on the mnemonic
string in Decoded; new instructions are added by writing a small
lowering function and registering it.
"""

from __future__ import annotations

from typing import Callable, Optional

from rotor.btor2.nodes import Model, Node, Sort
from rotor.btor2.riscv.decoder import Decoded

BV1 = Sort(1)
BV8 = Sort(8)
BV16 = Sort(16)
BV32 = Sort(32)
BV64 = Sort(64)
MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1

LowerResult = tuple[dict[int, Node], Node, Optional[Node]]
LowerFn = Callable[[Decoded, int, Model, list[Node], Optional[Node]], LowerResult]


def lower(
    d: Decoded,
    pc_value: int,
    m: Model,
    regs: list[Node],
    mem: Optional[Node],
) -> LowerResult:
    """Lower one decoded instruction.

    `mem` is the current memory-array node; the caller may pass None
    when building a model for a function that has no load/store
    instructions. Non-memory lowerings ignore the parameter; memory
    lowerings assume a valid Node.
    """
    fn = DISPATCH.get(d.mnem)
    if fn is None:                                 # pragma: no cover — decoder filters
        raise AssertionError(f"lower: unsupported mnem {d.mnem!r}")
    return fn(d, pc_value, m, regs, mem)


def _write(writes: dict[int, Node], rd: int, expr: Node) -> None:
    if rd != 0:
        writes[rd] = expr


def _fall(pc_value: int, m: Model) -> Node:
    return m.const(BV64, (pc_value + 4) & MASK64)


# ---------------------------------------------------------------------------
# I-type arithmetic / logic.
# ---------------------------------------------------------------------------

def _addi(d, pc_value, m, regs, mem):
    imm = m.const(BV64, d.imm & MASK64)
    result = m.op("add", BV64, regs[d.rs1], imm)
    writes: dict[int, Node] = {}
    _write(writes, d.rd, result)
    return writes, _fall(pc_value, m), None


def _xori(d, pc_value, m, regs, mem):
    imm = m.const(BV64, d.imm & MASK64)
    return _i_writes(m, d, m.op("xor", BV64, regs[d.rs1], imm)), _fall(pc_value, m), None


def _ori(d, pc_value, m, regs, mem):
    imm = m.const(BV64, d.imm & MASK64)
    return _i_writes(m, d, m.op("or", BV64, regs[d.rs1], imm)), _fall(pc_value, m), None


def _andi(d, pc_value, m, regs, mem):
    imm = m.const(BV64, d.imm & MASK64)
    return _i_writes(m, d, m.op("and", BV64, regs[d.rs1], imm)), _fall(pc_value, m), None


def _slti(d, pc_value, m, regs, mem):
    imm = m.const(BV64, d.imm & MASK64)
    cond = m.op("slt", BV1, regs[d.rs1], imm)
    return _i_writes(m, d, _bool_to_bv64(m, cond)), _fall(pc_value, m), None


def _sltiu(d, pc_value, m, regs, mem):
    imm = m.const(BV64, d.imm & MASK64)
    cond = m.op("ult", BV1, regs[d.rs1], imm)
    return _i_writes(m, d, _bool_to_bv64(m, cond)), _fall(pc_value, m), None


def _slli(d, pc_value, m, regs, mem):
    shamt = m.const(BV64, d.imm & 63)
    return _i_writes(m, d, m.op("sll", BV64, regs[d.rs1], shamt)), _fall(pc_value, m), None


def _srli(d, pc_value, m, regs, mem):
    shamt = m.const(BV64, d.imm & 63)
    return _i_writes(m, d, m.op("srl", BV64, regs[d.rs1], shamt)), _fall(pc_value, m), None


def _srai(d, pc_value, m, regs, mem):
    shamt = m.const(BV64, d.imm & 63)
    return _i_writes(m, d, m.op("sra", BV64, regs[d.rs1], shamt)), _fall(pc_value, m), None


# ---------------------------------------------------------------------------
# OP-IMM-32 (32-bit immediate ops, sign-extend result to 64).
# ---------------------------------------------------------------------------

def _addiw(d, pc_value, m, regs, mem):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    imm32 = m.const(BV32, d.imm & MASK32)
    sum32 = m.op("add", BV32, lo1, imm32)
    return _i_writes(m, d, m.sext(sum32, 32)), _fall(pc_value, m), None


def _slliw(d, pc_value, m, regs, mem):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    shamt = m.const(BV32, d.imm & 31)
    sh32 = m.op("sll", BV32, lo1, shamt)
    return _i_writes(m, d, m.sext(sh32, 32)), _fall(pc_value, m), None


def _srliw(d, pc_value, m, regs, mem):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    shamt = m.const(BV32, d.imm & 31)
    sh32 = m.op("srl", BV32, lo1, shamt)
    return _i_writes(m, d, m.sext(sh32, 32)), _fall(pc_value, m), None


def _sraiw(d, pc_value, m, regs, mem):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    shamt = m.const(BV32, d.imm & 31)
    sh32 = m.op("sra", BV32, lo1, shamt)
    return _i_writes(m, d, m.sext(sh32, 32)), _fall(pc_value, m), None


# ---------------------------------------------------------------------------
# R-type (64-bit).
# ---------------------------------------------------------------------------

def _add(d, pc_value, m, regs, mem):
    return _i_writes(m, d, m.op("add", BV64, regs[d.rs1], regs[d.rs2])), _fall(pc_value, m), None


def _sub(d, pc_value, m, regs, mem):
    return _i_writes(m, d, m.op("sub", BV64, regs[d.rs1], regs[d.rs2])), _fall(pc_value, m), None


def _and(d, pc_value, m, regs, mem):
    return _i_writes(m, d, m.op("and", BV64, regs[d.rs1], regs[d.rs2])), _fall(pc_value, m), None


def _or(d, pc_value, m, regs, mem):
    return _i_writes(m, d, m.op("or", BV64, regs[d.rs1], regs[d.rs2])), _fall(pc_value, m), None


def _xor(d, pc_value, m, regs, mem):
    return _i_writes(m, d, m.op("xor", BV64, regs[d.rs1], regs[d.rs2])), _fall(pc_value, m), None


def _slt(d, pc_value, m, regs, mem):
    cond = m.op("slt", BV1, regs[d.rs1], regs[d.rs2])
    return _i_writes(m, d, _bool_to_bv64(m, cond)), _fall(pc_value, m), None


def _sltu(d, pc_value, m, regs, mem):
    cond = m.op("ult", BV1, regs[d.rs1], regs[d.rs2])
    return _i_writes(m, d, _bool_to_bv64(m, cond)), _fall(pc_value, m), None


def _sll(d, pc_value, m, regs, mem):
    shamt = m.op("and", BV64, regs[d.rs2], m.const(BV64, 63))
    return _i_writes(m, d, m.op("sll", BV64, regs[d.rs1], shamt)), _fall(pc_value, m), None


def _srl(d, pc_value, m, regs, mem):
    shamt = m.op("and", BV64, regs[d.rs2], m.const(BV64, 63))
    return _i_writes(m, d, m.op("srl", BV64, regs[d.rs1], shamt)), _fall(pc_value, m), None


def _sra(d, pc_value, m, regs, mem):
    shamt = m.op("and", BV64, regs[d.rs2], m.const(BV64, 63))
    return _i_writes(m, d, m.op("sra", BV64, regs[d.rs1], shamt)), _fall(pc_value, m), None


# ---------------------------------------------------------------------------
# M extension (RV64M). RISC-V spec edge cases:
#   DIVU:   divisor == 0          → all ones (2^XLEN - 1)
#   REMU:   divisor == 0          → dividend
#   DIV:    divisor == 0          → -1
#           INT_MIN / -1          → INT_MIN (overflow)
#   REM:    divisor == 0          → dividend
#           INT_MIN % -1          → 0 (overflow)
# The -w variants operate on the low 32 bits and sign-extend the
# result; their div/rem special cases use INT32_MIN instead.
# ---------------------------------------------------------------------------

BV128 = Sort(128)
ALL_ONES_64 = MASK64
INT_MIN_64 = 1 << 63
ALL_ONES_32 = MASK32
INT_MIN_32 = 1 << 31


def _mul(d, pc_value, m, regs, mem):
    return _i_writes(m, d, m.op("mul", BV64, regs[d.rs1], regs[d.rs2])), _fall(pc_value, m), None


def _mulh_generic(d, pc_value, m, regs, mem, *, sign_a: bool, sign_b: bool):
    """Upper 64 bits of the 128-bit product of rs1 and rs2.

    `sign_a` / `sign_b` select sign-extension (for signed operands)
    vs zero-extension (for unsigned) into 128 bits before the
    multiplication. Covers all three RV64M high-half variants.
    """
    a128 = m.sext(regs[d.rs1], 64) if sign_a else m.uext(regs[d.rs1], 64)
    b128 = m.sext(regs[d.rs2], 64) if sign_b else m.uext(regs[d.rs2], 64)
    prod = m.op("mul", BV128, a128, b128)
    hi = m.slice(prod, 127, 64)
    return _i_writes(m, d, hi), _fall(pc_value, m), None


def _mulh(d, pc_value, m, regs, mem):
    return _mulh_generic(d, pc_value, m, regs, mem, sign_a=True, sign_b=True)


def _mulhsu(d, pc_value, m, regs, mem):
    return _mulh_generic(d, pc_value, m, regs, mem, sign_a=True, sign_b=False)


def _mulhu(d, pc_value, m, regs, mem):
    return _mulh_generic(d, pc_value, m, regs, mem, sign_a=False, sign_b=False)


def _divu(d, pc_value, m, regs, mem):
    a, b = regs[d.rs1], regs[d.rs2]
    is_zero = m.op("eq", BV1, b, m.const(BV64, 0))
    raw = m.op("udiv", BV64, a, b)
    result = m.ite(is_zero, m.const(BV64, ALL_ONES_64), raw)
    return _i_writes(m, d, result), _fall(pc_value, m), None


def _remu(d, pc_value, m, regs, mem):
    a, b = regs[d.rs1], regs[d.rs2]
    is_zero = m.op("eq", BV1, b, m.const(BV64, 0))
    raw = m.op("urem", BV64, a, b)
    result = m.ite(is_zero, a, raw)
    return _i_writes(m, d, result), _fall(pc_value, m), None


def _div(d, pc_value, m, regs, mem):
    a, b = regs[d.rs1], regs[d.rs2]
    is_zero = m.op("eq", BV1, b, m.const(BV64, 0))
    is_int_min = m.op("eq", BV1, a, m.const(BV64, INT_MIN_64))
    is_minus_one = m.op("eq", BV1, b, m.const(BV64, ALL_ONES_64))
    is_overflow = m.op("and", BV1, is_int_min, is_minus_one)
    raw = m.op("sdiv", BV64, a, b)
    overflow_result = m.const(BV64, INT_MIN_64)
    zero_result = m.const(BV64, ALL_ONES_64)
    result = m.ite(is_zero, zero_result, m.ite(is_overflow, overflow_result, raw))
    return _i_writes(m, d, result), _fall(pc_value, m), None


def _rem(d, pc_value, m, regs, mem):
    a, b = regs[d.rs1], regs[d.rs2]
    is_zero = m.op("eq", BV1, b, m.const(BV64, 0))
    is_int_min = m.op("eq", BV1, a, m.const(BV64, INT_MIN_64))
    is_minus_one = m.op("eq", BV1, b, m.const(BV64, ALL_ONES_64))
    is_overflow = m.op("and", BV1, is_int_min, is_minus_one)
    raw = m.op("srem", BV64, a, b)
    result = m.ite(is_zero, a, m.ite(is_overflow, m.const(BV64, 0), raw))
    return _i_writes(m, d, result), _fall(pc_value, m), None


# ---------------------------------------------------------------------------
# OP-32 (R-type, 32-bit).
# ---------------------------------------------------------------------------

def _addw(d, pc_value, m, regs, mem):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    sum32 = m.op("add", BV32, lo1, lo2)
    return _i_writes(m, d, m.sext(sum32, 32)), _fall(pc_value, m), None


def _subw(d, pc_value, m, regs, mem):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    diff32 = m.op("sub", BV32, lo1, lo2)
    return _i_writes(m, d, m.sext(diff32, 32)), _fall(pc_value, m), None


def _sllw(d, pc_value, m, regs, mem):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    shamt = m.op("and", BV32, lo2, m.const(BV32, 31))
    sh32 = m.op("sll", BV32, lo1, shamt)
    return _i_writes(m, d, m.sext(sh32, 32)), _fall(pc_value, m), None


def _srlw(d, pc_value, m, regs, mem):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    shamt = m.op("and", BV32, lo2, m.const(BV32, 31))
    sh32 = m.op("srl", BV32, lo1, shamt)
    return _i_writes(m, d, m.sext(sh32, 32)), _fall(pc_value, m), None


def _sraw(d, pc_value, m, regs, mem):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    shamt = m.op("and", BV32, lo2, m.const(BV32, 31))
    sh32 = m.op("sra", BV32, lo1, shamt)
    return _i_writes(m, d, m.sext(sh32, 32)), _fall(pc_value, m), None


# ---------------------------------------------------------------------------
# OP-32 M extension (RV64M -w variants). Low 32 bits only; sign-extend.
# Div/rem special cases match the -w versions of the ISA spec using
# INT32_MIN as the overflow dividend.
# ---------------------------------------------------------------------------

def _mulw(d, pc_value, m, regs, mem):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    prod32 = m.op("mul", BV32, lo1, lo2)
    return _i_writes(m, d, m.sext(prod32, 32)), _fall(pc_value, m), None


def _divuw(d, pc_value, m, regs, mem):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    is_zero = m.op("eq", BV1, lo2, m.const(BV32, 0))
    raw = m.op("udiv", BV32, lo1, lo2)
    result32 = m.ite(is_zero, m.const(BV32, ALL_ONES_32), raw)
    return _i_writes(m, d, m.sext(result32, 32)), _fall(pc_value, m), None


def _remuw(d, pc_value, m, regs, mem):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    is_zero = m.op("eq", BV1, lo2, m.const(BV32, 0))
    raw = m.op("urem", BV32, lo1, lo2)
    result32 = m.ite(is_zero, lo1, raw)
    return _i_writes(m, d, m.sext(result32, 32)), _fall(pc_value, m), None


def _divw(d, pc_value, m, regs, mem):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    is_zero = m.op("eq", BV1, lo2, m.const(BV32, 0))
    is_int_min = m.op("eq", BV1, lo1, m.const(BV32, INT_MIN_32))
    is_minus_one = m.op("eq", BV1, lo2, m.const(BV32, ALL_ONES_32))
    is_overflow = m.op("and", BV1, is_int_min, is_minus_one)
    raw = m.op("sdiv", BV32, lo1, lo2)
    result32 = m.ite(
        is_zero, m.const(BV32, ALL_ONES_32),
        m.ite(is_overflow, m.const(BV32, INT_MIN_32), raw),
    )
    return _i_writes(m, d, m.sext(result32, 32)), _fall(pc_value, m), None


def _remw(d, pc_value, m, regs, mem):
    lo1 = m.slice(regs[d.rs1], 31, 0)
    lo2 = m.slice(regs[d.rs2], 31, 0)
    is_zero = m.op("eq", BV1, lo2, m.const(BV32, 0))
    is_int_min = m.op("eq", BV1, lo1, m.const(BV32, INT_MIN_32))
    is_minus_one = m.op("eq", BV1, lo2, m.const(BV32, ALL_ONES_32))
    is_overflow = m.op("and", BV1, is_int_min, is_minus_one)
    raw = m.op("srem", BV32, lo1, lo2)
    result32 = m.ite(
        is_zero, lo1,
        m.ite(is_overflow, m.const(BV32, 0), raw),
    )
    return _i_writes(m, d, m.sext(result32, 32)), _fall(pc_value, m), None


# ---------------------------------------------------------------------------
# Branches.
# ---------------------------------------------------------------------------

def _branch(opname: str):
    def lower_fn(d, pc_value, m, regs, mem):
        cond = m.op(opname, BV1, regs[d.rs1], regs[d.rs2])
        target = m.const(BV64, (pc_value + d.imm) & MASK64)
        fall = _fall(pc_value, m)
        return {}, m.ite(cond, target, fall), None
    return lower_fn


def _beq(d, pc_value, m, regs, mem):
    cond = m.op("eq", BV1, regs[d.rs1], regs[d.rs2])
    return {}, m.ite(cond, m.const(BV64, (pc_value + d.imm) & MASK64), _fall(pc_value, m)), None


def _bne(d, pc_value, m, regs, mem):
    cond = m.op("neq", BV1, regs[d.rs1], regs[d.rs2])
    return {}, m.ite(cond, m.const(BV64, (pc_value + d.imm) & MASK64), _fall(pc_value, m)), None


_blt  = _branch("slt")
_bge  = _branch("sgte")
_bltu = _branch("ult")
_bgeu = _branch("ugte")


# ---------------------------------------------------------------------------
# U/J/JALR.
# ---------------------------------------------------------------------------

def _lui(d, pc_value, m, regs, mem):
    # Decoder already sign-extended imm to 64 bits.
    return _i_writes(m, d, m.const(BV64, d.imm & MASK64)), _fall(pc_value, m), None


def _auipc(d, pc_value, m, regs, mem):
    return _i_writes(m, d, m.const(BV64, (pc_value + d.imm) & MASK64)), _fall(pc_value, m), None


def _jal(d, pc_value, m, regs, mem):
    link = m.const(BV64, (pc_value + 4) & MASK64)
    target = m.const(BV64, (pc_value + d.imm) & MASK64)
    return _i_writes(m, d, link), target, None


def _jalr(d, pc_value, m, regs, mem):
    link = m.const(BV64, (pc_value + 4) & MASK64)
    imm = m.const(BV64, d.imm & MASK64)
    raw = m.op("add", BV64, regs[d.rs1], imm)
    target = m.op("and", BV64, raw, m.const(BV64, MASK64 ^ 1))
    return _i_writes(m, d, link), target, None


# ---------------------------------------------------------------------------
# LOAD / STORE.
#
# Byte-addressed memory state. An N-byte load reads N bytes at
# consecutive addresses and concatenates them little-endian; sign or
# zero extension then widens to 64 bits. Stores decompose the value
# into bytes and chain N writes into the array.
# ---------------------------------------------------------------------------

def _addr(m: Model, regs: list[Node], rs1: int, imm: int) -> Node:
    return m.op("add", BV64, regs[rs1], m.const(BV64, imm & MASK64))


def _read_bytes_le(m: Model, mem: Node, base: Node, nbytes: int) -> Node:
    """Read `nbytes` consecutive bytes starting at base, little-endian.

    Returns a bitvector of width nbytes*8. For nbytes == 1, returns the
    single byte directly (no concat).
    """
    byte = _read_byte(m, mem, base, 0)
    result = byte
    for i in range(1, nbytes):
        hi = _read_byte(m, mem, base, i)
        # result currently holds bytes [0..i-1]; prepend hi to get [0..i].
        result = m.op("concat", Sort(8 * (i + 1)), hi, result)
    return result


def _read_byte(m: Model, mem: Node, base: Node, offset: int) -> Node:
    if offset == 0:
        addr = base
    else:
        addr = m.op("add", BV64, base, m.const(BV64, offset & MASK64))
    return m.read(mem, addr)


def _load(width: int, signed: bool):
    def lower_fn(d, pc_value, m, regs, mem):
        addr = _addr(m, regs, d.rs1, d.imm)
        value = _read_bytes_le(m, mem, addr, width)
        if width == 8:                                # ld — already 64-bit
            result = value
        elif signed:
            result = m.sext(value, 64 - 8 * width)
        else:
            result = m.uext(value, 64 - 8 * width)
        return _i_writes(m, d, result), _fall(pc_value, m), None
    return lower_fn


_lb  = _load(1, signed=True)
_lh  = _load(2, signed=True)
_lw  = _load(4, signed=True)
_ld  = _load(8, signed=True)   # signed flag ignored for 64-bit
_lbu = _load(1, signed=False)
_lhu = _load(2, signed=False)
_lwu = _load(4, signed=False)


def _store(nbytes: int):
    def lower_fn(d, pc_value, m, regs, mem):
        addr = _addr(m, regs, d.rs1, d.imm)
        value = regs[d.rs2]
        new_mem = mem
        for i in range(nbytes):
            byte_i = m.slice(value, 8 * i + 7, 8 * i)
            if i == 0:
                a = addr
            else:
                a = m.op("add", BV64, addr, m.const(BV64, i & MASK64))
            new_mem = m.write(new_mem, a, byte_i)
        return {}, _fall(pc_value, m), new_mem
    return lower_fn


_sb = _store(1)
_sh = _store(2)
_sw = _store(4)
_sd = _store(8)


# ---------------------------------------------------------------------------
# Misc.
# ---------------------------------------------------------------------------

def _fence(d, pc_value, m, regs, mem):
    return {}, _fall(pc_value, m), None            # safe approximation: no-op


def _halt(d, pc_value, m, regs, mem):
    """ECALL / EBREAK: halt the machine cleanly by self-looping PC.

    Rotor does not model syscalls or the debug trap, so the most
    honest semantics is "execution stops here." PC becomes a self-
    loop at the instruction's own address; any PC after this
    instruction is unreachable from this path, which is what a real
    unreturned syscall or unhandled breakpoint would produce. Full
    syscall modeling (matching selfie's selfie convention) is in
    scope for a later phase.
    """
    return {}, m.const(BV64, pc_value), None


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

MEMORY_MNEMONICS: frozenset[str] = frozenset({
    "lb", "lh", "lw", "ld", "lbu", "lhu", "lwu",
    "sb", "sh", "sw", "sd",
})


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
    # LOAD / STORE
    "lb":    _lb,
    "lh":    _lh,
    "lw":    _lw,
    "ld":    _ld,
    "lbu":   _lbu,
    "lhu":   _lhu,
    "lwu":   _lwu,
    "sb":    _sb,
    "sh":    _sh,
    "sw":    _sw,
    "sd":    _sd,
    # M extension
    "mul":    _mul,
    "mulh":   _mulh,
    "mulhsu": _mulhsu,
    "mulhu":  _mulhu,
    "div":    _div,
    "divu":   _divu,
    "rem":    _rem,
    "remu":   _remu,
    "mulw":   _mulw,
    "divw":   _divw,
    "divuw":  _divuw,
    "remw":   _remw,
    "remuw":  _remuw,
    # Misc
    "fence": _fence,
    # System — halt-like semantics (pc self-loops; no register/memory effects).
    "ecall":  _halt,
    "ebreak": _halt,
}
