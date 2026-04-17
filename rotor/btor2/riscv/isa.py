"""RV64I instruction semantics as BTOR2 transition logic.

Each instruction contributes two combinational values and one predicate:

    enabled(instr)  — true when the 32-bit instruction matches this opcode/fN
    rd_value(instr) — the value written to rd under this instruction
    next_pc(instr)  — the PC to latch after this instruction

The dispatch framework collects all (enabled, rd_value, next_pc) tuples,
builds ITE chains so that exactly one is selected per instruction word, and
finally attaches ``next`` relations to the register file and program counter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from rotor.btor2.riscv import decoder as D

if TYPE_CHECKING:  # pragma: no cover
    from rotor.btor2 import Node
    from rotor.btor2.builder import RISCVMachineBuilder


# ──────────────────────────────────────────────────────────────────────────
# Opcode constants
# ──────────────────────────────────────────────────────────────────────────

OP_LUI = 0x37
OP_AUIPC = 0x17
OP_JAL = 0x6F
OP_JALR = 0x67
OP_BRANCH = 0x63
OP_LOAD = 0x03
OP_STORE = 0x23
OP_OP_IMM = 0x13
OP_OP_IMM_32 = 0x1B
OP_OP = 0x33
OP_OP_32 = 0x3B
OP_SYSTEM = 0x73


# ──────────────────────────────────────────────────────────────────────────
# Framework
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class _InstrSemantics:
    name: str
    enabled: "Node"           # 1-bit predicate
    rd_value: "Node | None"   # machine-word value written to rd, if any
    next_pc_delta: "Node"     # machine-word value to set PC to (absolute)
    writes_rd: bool = True


def build_fetch_decode_execute(b: "RISCVMachineBuilder") -> None:
    """Build fetch/decode/execute transitions for all cores."""
    for core in range(b.config.cores):
        _build_core(b, core)


def _build_core(b: "RISCVMachineBuilder", core: int) -> None:
    assert b.SID_MACHINE_WORD is not None and b.SID_SINGLE_WORD is not None
    assert b.SID_REGISTER_ADDRESS is not None

    pc = b._pc_nodes[core]
    regs = b._register_file[core]
    code = b._memory_segments[(core, "code")]

    # ---- Fetch: read 4 bytes from code memory at PC, little-endian.
    instr = _fetch_instruction(b, code, pc)

    # ---- Decode / execute each instruction form.
    semantics: list[_InstrSemantics] = []
    semantics += _lui_auipc(b, instr, pc)
    semantics += _jal_jalr(b, instr, pc, regs)
    semantics += _branches(b, instr, pc, regs)
    semantics += _op_imm(b, instr, regs)
    semantics += _op(b, instr, regs)

    # ---- Select default next PC = PC + 4.
    four = b.consth(b.SID_MACHINE_WORD, 4, "const_4")
    pc_plus_4 = b.add(pc, four, "pc+4")

    # Build the ITE cascade: start with "unknown instruction" traps.
    # next_pc defaults to PC+4; rd_value defaults to reading current rd (no-op).
    rd_idx = D.rd(b, instr)
    current_rd_value = b.read(regs, rd_idx, "current-rd-value")

    next_pc: "Node" = pc_plus_4
    next_rd_value: "Node" = current_rd_value
    writes_any: "Node" = b.NID_FALSE  # type: ignore[assignment]

    # Walk semantics bottom-up so priority matches declaration order.
    for sem in reversed(semantics):
        if sem.rd_value is not None and sem.writes_rd:
            next_rd_value = b.ite(sem.enabled, sem.rd_value, next_rd_value, f"sel-rd:{sem.name}")
            writes_any = b.ite(sem.enabled, b.NID_TRUE, writes_any, f"sel-we:{sem.name}")  # type: ignore[arg-type]
        next_pc = b.ite(sem.enabled, sem.next_pc_delta, next_pc, f"sel-pc:{sem.name}")

    # rd == x0 must never change: gate the register-file update.
    rd_is_zero = b.eq(rd_idx, b.zero(b.SID_REGISTER_ADDRESS, "x0"), "rd==x0")
    effective_write = b.and_(writes_any, b.not_(rd_is_zero, "rd!=x0"), "effective-write")

    new_rf = b.write(regs, rd_idx, next_rd_value, "new-rf")
    new_rf_selected = b.ite(effective_write, new_rf, regs, "rf-select")

    assert b.SID_REGISTER_STATE is not None
    b._state.next_nodes.append(
        b.next(b.SID_REGISTER_STATE, regs, new_rf_selected, "rf-next")
    )
    b._state.next_nodes.append(
        b.next(b.SID_MACHINE_WORD, pc, next_pc, "pc-next")
    )

    # ---- Illegal-instruction property: at least one of our semantics must
    # have fired. Otherwise raise a bad.
    any_enabled = semantics[0].enabled
    for sem in semantics[1:]:
        any_enabled = b.or_(any_enabled, sem.enabled, "any-enabled")
    illegal = b.not_(any_enabled, "illegal-instruction")
    b._state.property_nodes.append(
        b.bad(illegal, f"core{core}-illegal-instruction", "illegal instruction")
    )


# ──────────────────────────────────────────────────────────────────────────
# Fetch
# ──────────────────────────────────────────────────────────────────────────


def _fetch_instruction(
    b: "RISCVMachineBuilder", code: "Node", pc: "Node"
) -> "Node":
    """Assemble a 32-bit instruction word from four consecutive bytes."""
    assert b.SID_VIRTUAL_ADDRESS is not None and b.SID_MACHINE_WORD is not None

    def byte_at(offset: int) -> "Node":
        off = b.consth(b.SID_MACHINE_WORD, offset, f"+{offset}")
        addr_word = b.add(pc, off, f"pc+{offset}")
        addr = _truncate_to_vaddr(b, addr_word, f"vaddr+{offset}")
        return b.read(code, addr, f"code[pc+{offset}]")

    b3 = byte_at(3)
    b2 = byte_at(2)
    b1 = byte_at(1)
    b0 = byte_at(0)

    # Concatenate big-endian in BTOR2 semantics but assemble little-endian
    # word: {b3, b2, b1, b0} places b3 as MSB, which is what we want for a
    # little-endian view of memory.
    hi16 = b.concat(b3, b2, "instr_hi16")
    lo16 = b.concat(b1, b0, "instr_lo16")
    return b.concat(hi16, lo16, "instruction")


def _truncate_to_vaddr(
    b: "RISCVMachineBuilder", addr_word: "Node", name: str
) -> "Node":
    """Slice a machine-word address down to the virtual-address width."""
    assert b.SID_VIRTUAL_ADDRESS is not None
    vw = b.SID_VIRTUAL_ADDRESS.width or 0
    return b.slice(b.SID_VIRTUAL_ADDRESS, addr_word, vw - 1, 0, name)


# ──────────────────────────────────────────────────────────────────────────
# Instruction families
# ──────────────────────────────────────────────────────────────────────────


def _read_reg(b: "RISCVMachineBuilder", regs: "Node", idx: "Node", name: str) -> "Node":
    return b.read(regs, idx, name)


def _lui_auipc(
    b: "RISCVMachineBuilder", instr: "Node", pc: "Node",
) -> list[_InstrSemantics]:
    imm = D.imm_u(b, instr)
    four = b.consth(b.SID_MACHINE_WORD, 4, "+4")
    pc_plus_4 = b.add(pc, four, "pc+4")
    return [
        _InstrSemantics(
            name="lui",
            enabled=D.is_opcode(b, instr, OP_LUI, "op==lui"),
            rd_value=imm,
            next_pc_delta=pc_plus_4,
        ),
        _InstrSemantics(
            name="auipc",
            enabled=D.is_opcode(b, instr, OP_AUIPC, "op==auipc"),
            rd_value=b.add(pc, imm, "auipc_val"),
            next_pc_delta=pc_plus_4,
        ),
    ]


def _jal_jalr(
    b: "RISCVMachineBuilder", instr: "Node", pc: "Node", regs: "Node",
) -> list[_InstrSemantics]:
    four = b.consth(b.SID_MACHINE_WORD, 4, "+4")
    pc_plus_4 = b.add(pc, four, "pc+4")

    imm_j = D.imm_j(b, instr)
    jal_target = b.add(pc, imm_j, "jal-target")

    rs1_val = _read_reg(b, regs, D.rs1(b, instr), "jalr-rs1")
    imm_i = D.imm_i(b, instr)
    jalr_raw = b.add(rs1_val, imm_i, "jalr-raw")
    # Clear low bit (spec requires target & ~1).
    mask_one = b.ones(b.SID_MACHINE_WORD, "~1")  # placeholder — replaced below
    # We need a node for ~1 = all-ones XOR 1. Build it directly:
    ones = b.ones(b.SID_MACHINE_WORD, "all-ones")
    one = b.one(b.SID_MACHINE_WORD, "1")
    not_one = b.xor(ones, one, "~1")
    jalr_target = b.and_(jalr_raw, not_one, "jalr-target")

    return [
        _InstrSemantics(
            name="jal",
            enabled=D.is_opcode(b, instr, OP_JAL, "op==jal"),
            rd_value=pc_plus_4,
            next_pc_delta=jal_target,
        ),
        _InstrSemantics(
            name="jalr",
            enabled=b.and_(
                D.is_opcode(b, instr, OP_JALR, "op==jalr"),
                D.is_funct3(b, instr, 0, "f3==0"),
                "jalr-full",
            ),
            rd_value=pc_plus_4,
            next_pc_delta=jalr_target,
        ),
    ]


def _branches(
    b: "RISCVMachineBuilder", instr: "Node", pc: "Node", regs: "Node",
) -> list[_InstrSemantics]:
    four = b.consth(b.SID_MACHINE_WORD, 4, "+4")
    pc_plus_4 = b.add(pc, four, "pc+4")

    imm_b = D.imm_b(b, instr)
    taken_target = b.add(pc, imm_b, "branch-target")

    x1 = _read_reg(b, regs, D.rs1(b, instr), "branch-rs1")
    x2 = _read_reg(b, regs, D.rs2(b, instr), "branch-rs2")

    op = D.is_opcode(b, instr, OP_BRANCH, "op==branch")

    def mk(name: str, f3: int, cond: "Node") -> _InstrSemantics:
        enabled = b.and_(
            op,
            D.is_funct3(b, instr, f3, f"f3=={f3}"),
            f"{name}-enabled",
        )
        next_pc = b.ite(cond, taken_target, pc_plus_4, f"{name}-sel")
        return _InstrSemantics(
            name=name, enabled=enabled, rd_value=None,
            next_pc_delta=next_pc, writes_rd=False,
        )

    return [
        mk("beq",  0, b.eq(x1, x2, "x1==x2")),
        mk("bne",  1, b.neq(x1, x2, "x1!=x2")),
        mk("blt",  4, b.slt(x1, x2, "x1<s x2")),
        mk("bge",  5, b.sgte(x1, x2, "x1>=s x2")),
        mk("bltu", 6, b.ult(x1, x2, "x1<u x2")),
        mk("bgeu", 7, b.ugte(x1, x2, "x1>=u x2")),
    ]


def _op_imm(
    b: "RISCVMachineBuilder", instr: "Node", regs: "Node",
) -> list[_InstrSemantics]:
    four = b.consth(b.SID_MACHINE_WORD, 4, "+4")
    # next_pc is handled by caller (default PC+4 when no branch fires).
    pc = b._pc_nodes[0]
    pc_plus_4 = b.add(pc, four, "pc+4")

    op = D.is_opcode(b, instr, OP_OP_IMM, "op==opimm")
    x1 = _read_reg(b, regs, D.rs1(b, instr), "opimm-rs1")
    imm = D.imm_i(b, instr)

    # Shift amounts: low 6 bits of imm for RV64, low 5 bits for RV32.
    shamt_width = 6 if (b.SID_MACHINE_WORD.width or 0) == 64 else 5
    shamt_raw = b.slice(b.bitvec(shamt_width), imm, shamt_width - 1, 0, "shamt")
    shamt_ext = b.uext(
        b.SID_MACHINE_WORD, shamt_raw,
        (b.SID_MACHINE_WORD.width or 0) - shamt_width, "shamt-ext",
    )

    def mk(name: str, f3: int, value: "Node", f7_check: "Node | None" = None) -> _InstrSemantics:
        enabled = b.and_(op, D.is_funct3(b, instr, f3, f"f3=={f3}"), f"{name}-enabled")
        if f7_check is not None:
            enabled = b.and_(enabled, f7_check, f"{name}-f7")
        return _InstrSemantics(
            name=name, enabled=enabled, rd_value=value, next_pc_delta=pc_plus_4,
        )

    f7_zero = D.is_funct7(b, instr, 0x00, "f7==00")
    f7_sra = D.is_funct7(b, instr, 0x20, "f7==20")

    return [
        mk("addi",  0, b.add(x1, imm, "addi")),
        mk("slti",  2, b.uext(b.SID_MACHINE_WORD,
                              b.slt(x1, imm, "slti-cmp"),
                              (b.SID_MACHINE_WORD.width or 0) - 1, "slti-ext")),
        mk("sltiu", 3, b.uext(b.SID_MACHINE_WORD,
                              b.ult(x1, imm, "sltiu-cmp"),
                              (b.SID_MACHINE_WORD.width or 0) - 1, "sltiu-ext")),
        mk("xori",  4, b.xor(x1, imm, "xori")),
        mk("ori",   6, b.or_(x1, imm, "ori")),
        mk("andi",  7, b.and_(x1, imm, "andi")),
        mk("slli",  1, b.sll(x1, shamt_ext, "slli"), f7_zero),
        mk("srli",  5, b.srl(x1, shamt_ext, "srli"), f7_zero),
        mk("srai",  5, b.sra(x1, shamt_ext, "srai"), f7_sra),
    ]


def _op(
    b: "RISCVMachineBuilder", instr: "Node", regs: "Node",
) -> list[_InstrSemantics]:
    four = b.consth(b.SID_MACHINE_WORD, 4, "+4")
    pc = b._pc_nodes[0]
    pc_plus_4 = b.add(pc, four, "pc+4")

    op = D.is_opcode(b, instr, OP_OP, "op==op")
    x1 = _read_reg(b, regs, D.rs1(b, instr), "op-rs1")
    x2 = _read_reg(b, regs, D.rs2(b, instr), "op-rs2")

    # Shift amount: low bits of rs2.
    shamt_width = 6 if (b.SID_MACHINE_WORD.width or 0) == 64 else 5
    shamt_raw = b.slice(b.bitvec(shamt_width), x2, shamt_width - 1, 0, "op-shamt")
    shamt_ext = b.uext(
        b.SID_MACHINE_WORD, shamt_raw,
        (b.SID_MACHINE_WORD.width or 0) - shamt_width, "op-shamt-ext",
    )

    f7_zero = D.is_funct7(b, instr, 0x00, "f7==00")
    f7_sub_sra = D.is_funct7(b, instr, 0x20, "f7==20")

    def mk(name: str, f3: int, f7_ok: "Node", value: "Node") -> _InstrSemantics:
        enabled = b.and_(
            b.and_(op, D.is_funct3(b, instr, f3, f"f3=={f3}"), f"{name}-f3"),
            f7_ok,
            f"{name}-enabled",
        )
        return _InstrSemantics(
            name=name, enabled=enabled, rd_value=value, next_pc_delta=pc_plus_4,
        )

    return [
        mk("add",  0, f7_zero,   b.add(x1, x2, "add")),
        mk("sub",  0, f7_sub_sra, b.sub(x1, x2, "sub")),
        mk("sll",  1, f7_zero,   b.sll(x1, shamt_ext, "sll")),
        mk("slt",  2, f7_zero,   b.uext(b.SID_MACHINE_WORD,
                                        b.slt(x1, x2, "slt-cmp"),
                                        (b.SID_MACHINE_WORD.width or 0) - 1, "slt-ext")),
        mk("sltu", 3, f7_zero,   b.uext(b.SID_MACHINE_WORD,
                                        b.ult(x1, x2, "sltu-cmp"),
                                        (b.SID_MACHINE_WORD.width or 0) - 1, "sltu-ext")),
        mk("xor",  4, f7_zero,   b.xor(x1, x2, "xor")),
        mk("srl",  5, f7_zero,   b.srl(x1, shamt_ext, "srl")),
        mk("sra",  5, f7_sub_sra, b.sra(x1, shamt_ext, "sra")),
        mk("or",   6, f7_zero,   b.or_(x1, x2, "or")),
        mk("and",  7, f7_zero,   b.and_(x1, x2, "and")),
    ]
