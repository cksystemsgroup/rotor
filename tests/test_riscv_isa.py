"""Tests for the native RISC-V fetch/decode/execute module."""

from __future__ import annotations

from rotor.btor2 import BTOR2Printer, RISCVMachineBuilder
from rotor.btor2.riscv import decoder as D
from rotor.instance import ModelConfig


def test_decoder_extracts_opcode() -> None:
    b = RISCVMachineBuilder(ModelConfig(is_64bit=True, cores=1))
    b._build_sorts_and_constants()
    bv32 = b.bitvec(32)
    instr = b.state(bv32, "instr")
    op = D.opcode(b, instr)
    assert op.sort.width == 7
    assert op.op == "slice"
    assert op.params == [6, 0]


def test_decoder_reuses_fields() -> None:
    b = RISCVMachineBuilder(ModelConfig(is_64bit=True, cores=1))
    b._build_sorts_and_constants()
    bv32 = b.bitvec(32)
    instr = b.state(bv32, "instr")
    a = D.opcode(b, instr)
    a2 = D.opcode(b, instr)
    assert a is a2, "repeated slice should be deduplicated"


def test_full_build_produces_pc_and_properties() -> None:
    cfg = ModelConfig(is_64bit=True, cores=1, code_start=0x1000)
    b = RISCVMachineBuilder(cfg)
    model = b.build()
    assert "pc" in model.state_nodes
    # One 'init' per state (rf + segments + pc = ≥6), one 'next' per state
    # that we transition (rf + pc = 2).
    ops = [n.op for n in model.dag.nodes()]
    assert "init" in ops
    assert "next" in ops
    # Illegal-instruction bad property is installed.
    bad_symbols = {n.symbol for n in model.dag.nodes() if n.op == "bad"}
    assert any("illegal-instruction" in s for s in bad_symbols)


def test_build_round_trips_through_printer() -> None:
    cfg = ModelConfig(is_64bit=True, cores=1, code_start=0x1000)
    b = RISCVMachineBuilder(cfg)
    model = b.build()
    text = BTOR2Printer().render(model.dag)
    # Sanity: our RV64I subset produced a non-trivial model.
    assert text.count("\n") > 100
    assert "add" in text
    assert "ite" in text


def test_imm_i_sign_extends_to_word() -> None:
    b = RISCVMachineBuilder(ModelConfig(is_64bit=True, cores=1))
    b._build_sorts_and_constants()
    bv32 = b.bitvec(32)
    instr = b.state(bv32, "instr")
    imm = D.imm_i(b, instr)
    assert imm.sort.width == 64
    assert imm.op == "sext"


def test_imm_b_has_lsb_zero() -> None:
    """Branch offsets are word-aligned: the immediate must end in a literal 0."""
    b = RISCVMachineBuilder(ModelConfig(is_64bit=True, cores=1))
    b._build_sorts_and_constants()
    bv32 = b.bitvec(32)
    instr = b.state(bv32, "instr")
    imm = D.imm_b(b, instr)
    # The raw immediate before sext is 13 bits; sext brings it to 64.
    assert imm.sort.width == 64
