"""RISC-V disassembler for the RV64I + RV64M instruction subset.

Produces canonical assembly text for a 32-bit instruction word, matching
the format used by `riscv64-unknown-elf-objdump -d`: one ``mnemonic
operands`` line per instruction, with ABI register names (``a0`` rather
than ``x10``) and sign-extended decimal/hex immediates. Branches and jumps
render their targets as absolute addresses when ``pc`` is provided.

Unsupported instruction words (compressed 16-bit, RV*F, CSR, FENCE) fall
through to ``".word 0x<hex>"``.
"""

from __future__ import annotations


# ──────────────────────────────────────────────────────────────────────────
# Register ABI names
# ──────────────────────────────────────────────────────────────────────────

ABI_REGISTERS: tuple[str, ...] = (
    "zero", "ra", "sp", "gp", "tp",
    "t0", "t1", "t2",
    "s0", "s1",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    "t3", "t4", "t5", "t6",
)


def reg_name(idx: int) -> str:
    """Return the ABI name for register index ``idx``."""
    if 0 <= idx < len(ABI_REGISTERS):
        return ABI_REGISTERS[idx]
    return f"x{idx}"


# ──────────────────────────────────────────────────────────────────────────
# Field extractors
# ──────────────────────────────────────────────────────────────────────────


def _bits(word: int, hi: int, lo: int) -> int:
    width = hi - lo + 1
    return (word >> lo) & ((1 << width) - 1)


def _sext(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (value & (sign - 1)) - (value & sign)


def _imm_i(word: int) -> int:
    return _sext(_bits(word, 31, 20), 12)


def _imm_s(word: int) -> int:
    return _sext((_bits(word, 31, 25) << 5) | _bits(word, 11, 7), 12)


def _imm_b(word: int) -> int:
    raw = (
        (_bits(word, 31, 31) << 12)
        | (_bits(word, 7, 7) << 11)
        | (_bits(word, 30, 25) << 5)
        | (_bits(word, 11, 8) << 1)
    )
    return _sext(raw, 13)


def _imm_u(word: int) -> int:
    return _sext(_bits(word, 31, 12) << 12, 32)


def _imm_j(word: int) -> int:
    raw = (
        (_bits(word, 31, 31) << 20)
        | (_bits(word, 19, 12) << 12)
        | (_bits(word, 20, 20) << 11)
        | (_bits(word, 30, 21) << 1)
    )
    return _sext(raw, 21)


def _fmt_imm(value: int) -> str:
    """Format a signed integer the way objdump does."""
    if -256 <= value <= 256:
        return str(value)
    return f"{value:#x}" if value >= 0 else f"-{-value:#x}"


def _fmt_addr(addr: int) -> str:
    return f"0x{addr & ((1 << 64) - 1):x}"


# ──────────────────────────────────────────────────────────────────────────
# Opcode dispatch
# ──────────────────────────────────────────────────────────────────────────


def disassemble(word: int, pc: int = 0, is_64bit: bool = True) -> str:
    """Return the canonical assembly text for the instruction ``word``.

    ``pc`` is used to render branch/jump targets as absolute addresses.
    ``is_64bit`` enables RV64-only instructions (LWU, LD, SD, *W family).
    """
    opcode = word & 0x7F
    handler = _DISPATCH.get(opcode)
    if handler is None:
        return f".word 0x{word & 0xFFFFFFFF:08x}"
    result = handler(word, pc, is_64bit)
    return result or f".word 0x{word & 0xFFFFFFFF:08x}"


def _unknown(word: int, pc: int, is_64bit: bool) -> str:
    return ""


def _lui(word: int, pc: int, is_64bit: bool) -> str:
    return f"lui {reg_name(_bits(word, 11, 7))}, {_fmt_imm(_imm_u(word) >> 12)}"


def _auipc(word: int, pc: int, is_64bit: bool) -> str:
    return f"auipc {reg_name(_bits(word, 11, 7))}, {_fmt_imm(_imm_u(word) >> 12)}"


def _jal(word: int, pc: int, is_64bit: bool) -> str:
    rd = _bits(word, 11, 7)
    target = (pc + _imm_j(word)) & ((1 << 64) - 1)
    mnemonic = "j" if rd == 0 else "jal"
    if rd == 0:
        return f"j {_fmt_addr(target)}"
    return f"{mnemonic} {reg_name(rd)}, {_fmt_addr(target)}"


def _jalr(word: int, pc: int, is_64bit: bool) -> str:
    if _bits(word, 14, 12) != 0:
        return ""
    rd = _bits(word, 11, 7)
    rs1 = _bits(word, 19, 15)
    imm = _imm_i(word)
    # Canonical "ret" pseudo-instruction for `jalr x0, ra, 0`.
    if rd == 0 and rs1 == 1 and imm == 0:
        return "ret"
    if rd == 0:
        return f"jr {_fmt_imm(imm)}({reg_name(rs1)})"
    return f"jalr {reg_name(rd)}, {_fmt_imm(imm)}({reg_name(rs1)})"


def _branch(word: int, pc: int, is_64bit: bool) -> str:
    f3 = _bits(word, 14, 12)
    mnemonic = {0: "beq", 1: "bne", 4: "blt", 5: "bge", 6: "bltu", 7: "bgeu"}.get(f3)
    if mnemonic is None:
        return ""
    rs1 = reg_name(_bits(word, 19, 15))
    rs2 = reg_name(_bits(word, 24, 20))
    target = (pc + _imm_b(word)) & ((1 << 64) - 1)
    return f"{mnemonic} {rs1}, {rs2}, {_fmt_addr(target)}"


def _load(word: int, pc: int, is_64bit: bool) -> str:
    f3 = _bits(word, 14, 12)
    names = {0: "lb", 1: "lh", 2: "lw", 4: "lbu", 5: "lhu"}
    if is_64bit:
        names.update({3: "ld", 6: "lwu"})
    mnemonic = names.get(f3)
    if mnemonic is None:
        return ""
    rd = reg_name(_bits(word, 11, 7))
    rs1 = reg_name(_bits(word, 19, 15))
    return f"{mnemonic} {rd}, {_fmt_imm(_imm_i(word))}({rs1})"


def _store(word: int, pc: int, is_64bit: bool) -> str:
    f3 = _bits(word, 14, 12)
    names = {0: "sb", 1: "sh", 2: "sw"}
    if is_64bit:
        names[3] = "sd"
    mnemonic = names.get(f3)
    if mnemonic is None:
        return ""
    rs2 = reg_name(_bits(word, 24, 20))
    rs1 = reg_name(_bits(word, 19, 15))
    return f"{mnemonic} {rs2}, {_fmt_imm(_imm_s(word))}({rs1})"


_OP_IMM_F3 = {
    0: "addi", 1: "slli", 2: "slti", 3: "sltiu",
    4: "xori", 5: "srli",  # 5 also srai, disambiguate by f7
    6: "ori", 7: "andi",
}


def _op_imm(word: int, pc: int, is_64bit: bool) -> str:
    rd = reg_name(_bits(word, 11, 7))
    rs1 = reg_name(_bits(word, 19, 15))
    f3 = _bits(word, 14, 12)
    imm = _imm_i(word)
    mnemonic = _OP_IMM_F3.get(f3)
    if mnemonic is None:
        return ""
    # Shift immediates: shamt = low 6 bits; SRAI if f7 top bit set.
    if mnemonic in ("slli", "srli"):
        shamt = imm & (0x3F if is_64bit else 0x1F)
        if mnemonic == "srli" and (_bits(word, 31, 25) == 0x20 or _bits(word, 31, 26) == 0x10):
            mnemonic = "srai"
        return f"{mnemonic} {rd}, {rs1}, {shamt}"
    # "mv" pseudo-instruction for addi rd, rs1, 0.
    if mnemonic == "addi" and imm == 0:
        if rs1 == "zero":
            return f"li {rd}, 0"
        return f"mv {rd}, {rs1}"
    return f"{mnemonic} {rd}, {rs1}, {_fmt_imm(imm)}"


def _op_imm_32(word: int, pc: int, is_64bit: bool) -> str:
    if not is_64bit:
        return ""
    rd = reg_name(_bits(word, 11, 7))
    rs1 = reg_name(_bits(word, 19, 15))
    f3 = _bits(word, 14, 12)
    imm = _imm_i(word)
    names = {0: "addiw", 1: "slliw", 5: "srliw"}
    mnemonic = names.get(f3)
    if mnemonic is None:
        return ""
    if mnemonic in ("slliw", "srliw"):
        if mnemonic == "srliw" and _bits(word, 31, 25) == 0x20:
            mnemonic = "sraiw"
        return f"{mnemonic} {rd}, {rs1}, {imm & 0x1F}"
    if mnemonic == "addiw" and imm == 0:
        return f"sext.w {rd}, {rs1}"
    return f"{mnemonic} {rd}, {rs1}, {_fmt_imm(imm)}"


_OP_F3_F7_ZERO = {
    0: "add", 1: "sll", 2: "slt", 3: "sltu",
    4: "xor", 5: "srl", 6: "or", 7: "and",
}
_OP_F3_F7_SUB = {0: "sub", 5: "sra"}
_OP_M_F3 = {0: "mul", 1: "mulh", 2: "mulhsu", 3: "mulhu",
            4: "div", 5: "divu", 6: "rem", 7: "remu"}


def _op(word: int, pc: int, is_64bit: bool) -> str:
    rd = reg_name(_bits(word, 11, 7))
    rs1 = reg_name(_bits(word, 19, 15))
    rs2 = reg_name(_bits(word, 24, 20))
    f3 = _bits(word, 14, 12)
    f7 = _bits(word, 31, 25)
    if f7 == 0x00:
        mnemonic = _OP_F3_F7_ZERO.get(f3)
    elif f7 == 0x20:
        mnemonic = _OP_F3_F7_SUB.get(f3)
    elif f7 == 0x01:
        mnemonic = _OP_M_F3.get(f3)
    else:
        return ""
    if mnemonic is None:
        return ""
    return f"{mnemonic} {rd}, {rs1}, {rs2}"


_OP_32_F3_F7_ZERO = {0: "addw", 1: "sllw", 5: "srlw"}
_OP_32_F3_F7_SUB = {0: "subw", 5: "sraw"}
_OP_32_M_F3 = {0: "mulw", 4: "divw", 5: "divuw", 6: "remw", 7: "remuw"}


def _op_32(word: int, pc: int, is_64bit: bool) -> str:
    if not is_64bit:
        return ""
    rd = reg_name(_bits(word, 11, 7))
    rs1 = reg_name(_bits(word, 19, 15))
    rs2 = reg_name(_bits(word, 24, 20))
    f3 = _bits(word, 14, 12)
    f7 = _bits(word, 31, 25)
    if f7 == 0x00:
        mnemonic = _OP_32_F3_F7_ZERO.get(f3)
    elif f7 == 0x20:
        mnemonic = _OP_32_F3_F7_SUB.get(f3)
    elif f7 == 0x01:
        mnemonic = _OP_32_M_F3.get(f3)
    else:
        return ""
    if mnemonic is None:
        return ""
    return f"{mnemonic} {rd}, {rs1}, {rs2}"


def _system(word: int, pc: int, is_64bit: bool) -> str:
    # ECALL: funct3=0 and imm=0; EBREAK: imm=1.
    if _bits(word, 14, 12) == 0 and _bits(word, 11, 7) == 0 and _bits(word, 19, 15) == 0:
        imm = _bits(word, 31, 20)
        if imm == 0:
            return "ecall"
        if imm == 1:
            return "ebreak"
    return ""


_DISPATCH = {
    0x03: _load,
    0x13: _op_imm,
    0x17: _auipc,
    0x1B: _op_imm_32,
    0x23: _store,
    0x33: _op,
    0x37: _lui,
    0x3B: _op_32,
    0x63: _branch,
    0x67: _jalr,
    0x6F: _jal,
    0x73: _system,
}
