"""RV64I instruction semantics as BTOR2 transition logic.

Each instruction contributes a predicate ``enabled`` plus optional effects:

    rd_value    : machine-word value written to rd (None ⇒ no rd write)
    next_pc     : absolute PC after this instruction
    next_memory : unified memory array after this instruction
                  (None ⇒ memory unchanged)
    halt        : if True, enabling this instruction sets the halted latch

The dispatch framework collects all semantics, builds ITE cascades so that
exactly one fires per instruction word, and attaches ``next`` relations to
the register file, program counter, memory, and halt latch. A ``halted``
latch gates all updates so that a halted machine stays halted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    enabled: "Node"
    rd_value: "Node | None" = None
    next_pc: "Node | None" = None      # None ⇒ default PC+4
    next_memory: "Node | None" = None  # None ⇒ memory unchanged
    writes_rd: bool = True
    halt: bool = False


def build_fetch_decode_execute(b: "RISCVMachineBuilder") -> None:
    """Build fetch/decode/execute transitions for all cores."""
    for core in range(b.config.cores):
        _build_core(b, core)


def _build_core(b: "RISCVMachineBuilder", core: int) -> None:
    assert b.SID_MACHINE_WORD is not None and b.SID_BOOLEAN is not None
    assert b.SID_REGISTER_ADDRESS is not None and b.SID_REGISTER_STATE is not None
    assert b.NID_TRUE is not None and b.NID_FALSE is not None

    pc = b._pc_nodes[core]
    regs = b._register_file[core]
    code = b._memory_segments[(core, "code")]
    memory = b._memory_segments[(core, "memory")]
    halted = b._halted_nodes[core]

    # ---- Fetch: read 4 bytes from code memory at PC.
    instr = _fetch_instruction(b, code, pc)

    # ---- Collect instruction semantics.
    pc_plus_4 = b.add(pc, b.consth(b.SID_MACHINE_WORD, 4, "+4"), "pc+4")
    sem: list[_InstrSemantics] = []
    sem += _lui_auipc(b, instr, pc, pc_plus_4)
    sem += _jal_jalr(b, instr, pc, regs, pc_plus_4)
    sem += _branches(b, instr, pc, regs, pc_plus_4)
    sem += _op_imm(b, instr, regs)
    sem += _op(b, instr, regs)
    sem += _loads(b, instr, regs, memory)
    sem += _stores(b, instr, regs, memory)
    sem += _ecall(b, instr)

    # ---- Select outputs via ITE cascade over enabled predicates.
    rd_idx = D.rd(b, instr)
    current_rd_value = b.read(regs, rd_idx, "current-rd-value")

    next_pc: "Node" = pc_plus_4
    next_rd_value: "Node" = current_rd_value
    writes_any: "Node" = b.NID_FALSE
    next_memory: "Node" = memory
    ecall_taken: "Node" = b.NID_FALSE

    for s in reversed(sem):
        if s.rd_value is not None and s.writes_rd:
            next_rd_value = b.ite(s.enabled, s.rd_value, next_rd_value, f"sel-rd:{s.name}")
            writes_any = b.ite(s.enabled, b.NID_TRUE, writes_any, f"sel-we:{s.name}")
        if s.next_pc is not None:
            next_pc = b.ite(s.enabled, s.next_pc, next_pc, f"sel-pc:{s.name}")
        if s.next_memory is not None:
            next_memory = b.ite(s.enabled, s.next_memory, next_memory, f"sel-mem:{s.name}")
        if s.halt:
            ecall_taken = b.ite(s.enabled, b.NID_TRUE, ecall_taken, f"sel-halt:{s.name}")

    # rd == x0 must never change: gate the register-file update.
    rd_is_zero = b.eq(rd_idx, b.zero(b.SID_REGISTER_ADDRESS, "x0"), "rd==x0")
    effective_write = b.and_(writes_any, b.not_(rd_is_zero, "rd!=x0"), "effective-write")

    new_rf = b.write(regs, rd_idx, next_rd_value, "new-rf")
    new_rf_selected = b.ite(effective_write, new_rf, regs, "rf-select")

    # When halted, everything stays put.
    not_halted = b.not_(halted, "!halted")
    gated_rf = b.ite(not_halted, new_rf_selected, regs, "rf-gated")
    gated_pc = b.ite(not_halted, next_pc, pc, "pc-gated")
    gated_mem = b.ite(not_halted, next_memory, memory, "mem-gated")
    next_halted = b.ite(not_halted, ecall_taken, halted, "halted-gated")

    b._state.next_nodes.append(
        b.next(b.SID_REGISTER_STATE, regs, gated_rf, "rf-next")
    )
    b._state.next_nodes.append(
        b.next(b.SID_MACHINE_WORD, pc, gated_pc, "pc-next")
    )
    assert memory.sort is not None
    b._state.next_nodes.append(
        b.next(memory.sort, memory, gated_mem, "mem-next")
    )
    b._state.next_nodes.append(
        b.next(b.SID_BOOLEAN, halted, next_halted, "halted-next")
    )

    # ---- Illegal-instruction property: at least one semantic must fire
    # when the core is not halted.
    any_enabled = sem[0].enabled
    for s in sem[1:]:
        any_enabled = b.or_(any_enabled, s.enabled, "any-enabled")
    illegal = b.and_(b.not_(any_enabled, "no-semantic"), not_halted,
                     "illegal-instruction")
    b._state.property_nodes.append(
        b.bad(illegal, f"core{core}-illegal-instruction", "illegal instruction")
    )


# ──────────────────────────────────────────────────────────────────────────
# Fetch
# ──────────────────────────────────────────────────────────────────────────


def _fetch_instruction(
    b: "RISCVMachineBuilder", code: "Node", pc: "Node"
) -> "Node":
    assert b.SID_VIRTUAL_ADDRESS is not None and b.SID_MACHINE_WORD is not None

    def byte_at(offset: int) -> "Node":
        off = b.consth(b.SID_MACHINE_WORD, offset, f"+{offset}")
        addr_word = b.add(pc, off, f"pc+{offset}")
        addr = _trunc_vaddr(b, addr_word, f"vaddr+{offset}")
        return b.read(code, addr, f"code[pc+{offset}]")

    b3, b2, b1, b0 = byte_at(3), byte_at(2), byte_at(1), byte_at(0)
    hi16 = b.concat(b3, b2, "instr_hi16")
    lo16 = b.concat(b1, b0, "instr_lo16")
    return b.concat(hi16, lo16, "instruction")


def _trunc_vaddr(b: "RISCVMachineBuilder", addr_word: "Node", name: str) -> "Node":
    assert b.SID_VIRTUAL_ADDRESS is not None
    vw = b.SID_VIRTUAL_ADDRESS.width or 0
    return b.slice(b.SID_VIRTUAL_ADDRESS, addr_word, vw - 1, 0, name)


# ──────────────────────────────────────────────────────────────────────────
# Instruction families
# ──────────────────────────────────────────────────────────────────────────


def _read_reg(b: "RISCVMachineBuilder", regs: "Node", idx: "Node", name: str) -> "Node":
    return b.read(regs, idx, name)


def _lui_auipc(
    b: "RISCVMachineBuilder", instr: "Node", pc: "Node", pc_plus_4: "Node",
) -> list[_InstrSemantics]:
    imm = D.imm_u(b, instr)
    return [
        _InstrSemantics(
            name="lui",
            enabled=D.is_opcode(b, instr, OP_LUI, "op==lui"),
            rd_value=imm,
            next_pc=pc_plus_4,
        ),
        _InstrSemantics(
            name="auipc",
            enabled=D.is_opcode(b, instr, OP_AUIPC, "op==auipc"),
            rd_value=b.add(pc, imm, "auipc_val"),
            next_pc=pc_plus_4,
        ),
    ]


def _jal_jalr(
    b: "RISCVMachineBuilder", instr: "Node", pc: "Node", regs: "Node",
    pc_plus_4: "Node",
) -> list[_InstrSemantics]:
    imm_j = D.imm_j(b, instr)
    jal_target = b.add(pc, imm_j, "jal-target")

    rs1_val = _read_reg(b, regs, D.rs1(b, instr), "jalr-rs1")
    imm_i = D.imm_i(b, instr)
    jalr_raw = b.add(rs1_val, imm_i, "jalr-raw")
    ones = b.ones(b.SID_MACHINE_WORD, "all-ones")
    one = b.one(b.SID_MACHINE_WORD, "1")
    not_one = b.xor(ones, one, "~1")
    jalr_target = b.and_(jalr_raw, not_one, "jalr-target")

    return [
        _InstrSemantics(
            name="jal",
            enabled=D.is_opcode(b, instr, OP_JAL, "op==jal"),
            rd_value=pc_plus_4,
            next_pc=jal_target,
        ),
        _InstrSemantics(
            name="jalr",
            enabled=b.and_(
                D.is_opcode(b, instr, OP_JALR, "op==jalr"),
                D.is_funct3(b, instr, 0, "f3==0"),
                "jalr-full",
            ),
            rd_value=pc_plus_4,
            next_pc=jalr_target,
        ),
    ]


def _branches(
    b: "RISCVMachineBuilder", instr: "Node", pc: "Node", regs: "Node",
    pc_plus_4: "Node",
) -> list[_InstrSemantics]:
    imm_b = D.imm_b(b, instr)
    taken_target = b.add(pc, imm_b, "branch-target")
    x1 = _read_reg(b, regs, D.rs1(b, instr), "branch-rs1")
    x2 = _read_reg(b, regs, D.rs2(b, instr), "branch-rs2")
    op = D.is_opcode(b, instr, OP_BRANCH, "op==branch")

    def mk(name: str, f3: int, cond: "Node") -> _InstrSemantics:
        enabled = b.and_(op, D.is_funct3(b, instr, f3, f"f3=={f3}"), f"{name}-enabled")
        next_pc = b.ite(cond, taken_target, pc_plus_4, f"{name}-sel")
        return _InstrSemantics(name=name, enabled=enabled, next_pc=next_pc, writes_rd=False)

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
    op = D.is_opcode(b, instr, OP_OP_IMM, "op==opimm")
    x1 = _read_reg(b, regs, D.rs1(b, instr), "opimm-rs1")
    imm = D.imm_i(b, instr)

    shamt_width = 6 if (b.SID_MACHINE_WORD.width or 0) == 64 else 5
    shamt_raw = b.slice(b.bitvec(shamt_width), imm, shamt_width - 1, 0, "shamt")
    shamt_ext = b.uext(
        b.SID_MACHINE_WORD, shamt_raw,
        (b.SID_MACHINE_WORD.width or 0) - shamt_width, "shamt-ext",
    )

    f7_zero = D.is_funct7(b, instr, 0x00, "f7==00")
    f7_sra = D.is_funct7(b, instr, 0x20, "f7==20")

    def mk(name: str, f3: int, value: "Node", f7_check: "Node | None" = None) -> _InstrSemantics:
        enabled = b.and_(op, D.is_funct3(b, instr, f3, f"f3=={f3}"), f"{name}-enabled")
        if f7_check is not None:
            enabled = b.and_(enabled, f7_check, f"{name}-f7")
        return _InstrSemantics(name=name, enabled=enabled, rd_value=value)

    mw = b.SID_MACHINE_WORD.width or 0
    return [
        mk("addi",  0, b.add(x1, imm, "addi")),
        mk("slti",  2, b.uext(b.SID_MACHINE_WORD, b.slt(x1, imm, "slti-cmp"), mw - 1, "slti-ext")),
        mk("sltiu", 3, b.uext(b.SID_MACHINE_WORD, b.ult(x1, imm, "sltiu-cmp"), mw - 1, "sltiu-ext")),
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
    op = D.is_opcode(b, instr, OP_OP, "op==op")
    x1 = _read_reg(b, regs, D.rs1(b, instr), "op-rs1")
    x2 = _read_reg(b, regs, D.rs2(b, instr), "op-rs2")

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
        return _InstrSemantics(name=name, enabled=enabled, rd_value=value)

    mw = b.SID_MACHINE_WORD.width or 0
    return [
        mk("add",  0, f7_zero,   b.add(x1, x2, "add")),
        mk("sub",  0, f7_sub_sra, b.sub(x1, x2, "sub")),
        mk("sll",  1, f7_zero,   b.sll(x1, shamt_ext, "sll")),
        mk("slt",  2, f7_zero,   b.uext(b.SID_MACHINE_WORD, b.slt(x1, x2, "slt-cmp"), mw - 1, "slt-ext")),
        mk("sltu", 3, f7_zero,   b.uext(b.SID_MACHINE_WORD, b.ult(x1, x2, "sltu-cmp"), mw - 1, "sltu-ext")),
        mk("xor",  4, f7_zero,   b.xor(x1, x2, "xor")),
        mk("srl",  5, f7_zero,   b.srl(x1, shamt_ext, "srl")),
        mk("sra",  5, f7_sub_sra, b.sra(x1, shamt_ext, "sra")),
        mk("or",   6, f7_zero,   b.or_(x1, x2, "or")),
        mk("and",  7, f7_zero,   b.and_(x1, x2, "and")),
    ]


# ──────────────────────────────────────────────────────────────────────────
# Loads
# ──────────────────────────────────────────────────────────────────────────


def _loads(
    b: "RISCVMachineBuilder", instr: "Node", regs: "Node", memory: "Node",
) -> list[_InstrSemantics]:
    """Model signed/unsigned byte/half/word/double loads from unified memory."""
    assert b.SID_VIRTUAL_ADDRESS is not None and b.SID_MACHINE_WORD is not None

    op = D.is_opcode(b, instr, OP_LOAD, "op==load")
    base = _read_reg(b, regs, D.rs1(b, instr), "load-base")
    imm = D.imm_i(b, instr)
    addr_word = b.add(base, imm, "load-addr-word")

    def byte_at(k: int) -> "Node":
        off = b.consth(b.SID_MACHINE_WORD, k, f"+{k}")
        word = b.add(addr_word, off, f"load-addr+{k}")
        addr = _trunc_vaddr(b, word, f"load-vaddr+{k}")
        return b.read(memory, addr, f"mem[addr+{k}]")

    mw = b.SID_MACHINE_WORD.width or 0

    def mk(name: str, f3: int, width: int, signed: bool) -> _InstrSemantics:
        enabled = b.and_(op, D.is_funct3(b, instr, f3, f"f3=={f3}"), f"{name}-enabled")
        bytes_ = [byte_at(k) for k in range(width)]
        # Little-endian assembly: most-significant byte first in BTOR2 concat.
        value = bytes_[0]
        for k in range(1, width):
            value = b.concat(bytes_[k], value, f"{name}-concat{k}")
        ext_bits = mw - (width * 8)
        if ext_bits > 0:
            if signed:
                value = b.sext(b.SID_MACHINE_WORD, value, ext_bits, f"{name}-sext")
            else:
                value = b.uext(b.SID_MACHINE_WORD, value, ext_bits, f"{name}-uext")
        return _InstrSemantics(name=name, enabled=enabled, rd_value=value)

    entries = [
        mk("lb",  0, 1, signed=True),
        mk("lh",  1, 2, signed=True),
        mk("lw",  2, 4, signed=True),
        mk("lbu", 4, 1, signed=False),
        mk("lhu", 5, 2, signed=False),
    ]
    if mw == 64:
        entries.append(mk("ld",  3, 8, signed=True))
        entries.append(mk("lwu", 6, 4, signed=False))
    return entries


# ──────────────────────────────────────────────────────────────────────────
# Stores
# ──────────────────────────────────────────────────────────────────────────


def _stores(
    b: "RISCVMachineBuilder", instr: "Node", regs: "Node", memory: "Node",
) -> list[_InstrSemantics]:
    """Model byte/half/word/double stores into unified memory."""
    assert b.SID_VIRTUAL_ADDRESS is not None and b.SID_BYTE is not None

    op = D.is_opcode(b, instr, OP_STORE, "op==store")
    base = _read_reg(b, regs, D.rs1(b, instr), "store-base")
    imm = D.imm_s(b, instr)
    addr_word = b.add(base, imm, "store-addr-word")
    value_reg = _read_reg(b, regs, D.rs2(b, instr), "store-rs2")

    def byte_slice(lo: int) -> "Node":
        """Bits [lo+7:lo] of rs2 — one byte to store."""
        return b.slice(b.SID_BYTE, value_reg, lo + 7, lo, f"store-byte@{lo}")

    def addr_plus(k: int) -> "Node":
        off = b.consth(b.SID_MACHINE_WORD, k, f"+{k}")
        word = b.add(addr_word, off, f"store-addr+{k}")
        return _trunc_vaddr(b, word, f"store-vaddr+{k}")

    def mk(name: str, f3: int, width: int) -> _InstrSemantics:
        enabled = b.and_(op, D.is_funct3(b, instr, f3, f"f3=={f3}"), f"{name}-enabled")
        new_mem: "Node" = memory
        for k in range(width):
            new_mem = b.write(new_mem, addr_plus(k), byte_slice(k * 8), f"{name}-w{k}")
        return _InstrSemantics(
            name=name, enabled=enabled, rd_value=None, writes_rd=False,
            next_memory=new_mem,
        )

    mw = b.SID_MACHINE_WORD.width or 0
    entries = [
        mk("sb", 0, 1),
        mk("sh", 1, 2),
        mk("sw", 2, 4),
    ]
    if mw == 64:
        entries.append(mk("sd", 3, 8))
    return entries


# ──────────────────────────────────────────────────────────────────────────
# ECALL → halt
# ──────────────────────────────────────────────────────────────────────────


def _ecall(
    b: "RISCVMachineBuilder", instr: "Node",
) -> list[_InstrSemantics]:
    """Recognize the ECALL instruction and latch the halted state.

    Full syscall dispatch (exit/read/write/brk) is deferred; for verification
    purposes we treat any ECALL as a program-terminating event, which is
    enough to prevent ECALL from tripping the illegal-instruction bad state
    and to stop the machine transitioning further.
    """
    op = D.is_opcode(b, instr, OP_SYSTEM, "op==system")
    f3_zero = D.is_funct3(b, instr, 0, "f3==0")
    imm = D.imm_i(b, instr)
    # ECALL is the SYSTEM instruction with imm[11:0] == 0.
    imm_zero = b.eq(imm, b.zero(b.SID_MACHINE_WORD, "0"), "imm==0")
    enabled = b.and_(b.and_(op, f3_zero, "system-f3-0"), imm_zero, "ecall")
    return [
        _InstrSemantics(
            name="ecall",
            enabled=enabled,
            rd_value=None,
            writes_rd=False,
            halt=True,
        ),
    ]
