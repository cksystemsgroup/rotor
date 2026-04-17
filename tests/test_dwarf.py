"""Tests for the DWARF expression evaluator."""

from __future__ import annotations

from rotor.dwarf import (
    DW_OP_addr,
    DW_OP_breg0,
    DW_OP_const1u,
    DW_OP_const2u,
    DW_OP_consts,
    DW_OP_deref,
    DW_OP_fbreg,
    DW_OP_lit0,
    DW_OP_plus,
    DW_OP_reg0,
    DW_OP_stack_value,
    DWARFExprEvaluator,
)


def test_literal_push() -> None:
    ev = DWARFExprEvaluator(registers={}, memory={}, word_bytes=8)
    assert ev.evaluate(bytes([DW_OP_lit0 + 7])) == 7


def test_register_value() -> None:
    ev = DWARFExprEvaluator(registers={3: 0xDEADBEEF}, memory={}, word_bytes=8)
    # reg3 → value
    assert ev.evaluate(bytes([DW_OP_reg0 + 3])) == 0xDEADBEEF
    assert ev.is_value


def test_breg_plus_offset() -> None:
    # DW_OP_breg2 +16  →  regs[2] + 16
    regs = {2: 0x1000}
    # signed LEB128 for 16 → 0x10
    expr = bytes([DW_OP_breg0 + 2, 0x10])
    ev = DWARFExprEvaluator(registers=regs, memory={}, word_bytes=8)
    assert ev.evaluate(expr) == 0x1010
    assert not ev.is_value  # address, not value


def test_fbreg_negative_offset() -> None:
    expr = bytes([DW_OP_fbreg, 0x70])  # -16 in signed LEB128 (0x70 = 0b1110000 sign-extended)
    ev = DWARFExprEvaluator(registers={}, memory={}, frame_base=0x1000, word_bytes=8)
    assert ev.evaluate(expr) == 0x1000 - 16


def test_addr_and_deref() -> None:
    # memory at 0x1000 holds little-endian 64-bit value 42
    memory = {0x1000 + i: (42 >> (8 * i)) & 0xFF for i in range(8)}
    expr = bytes(
        [DW_OP_addr]
        + list((0x1000).to_bytes(8, "little"))
        + [DW_OP_deref]
    )
    ev = DWARFExprEvaluator(registers={}, memory=memory, word_bytes=8)
    assert ev.evaluate(expr) == 42


def test_plus() -> None:
    expr = bytes(
        [
            DW_OP_const1u, 10,
            DW_OP_const2u, 0x01, 0x00,  # little-endian 1
            DW_OP_plus,
            DW_OP_stack_value,
        ]
    )
    ev = DWARFExprEvaluator(registers={}, memory={}, word_bytes=8)
    assert ev.evaluate(expr) == 11
    assert ev.is_value


def test_consts_negative() -> None:
    # DW_OP_consts -1 (LEB128 0x7f)
    expr = bytes([DW_OP_consts, 0x7F, DW_OP_stack_value])
    ev = DWARFExprEvaluator(registers={}, memory={}, word_bytes=4)
    # -1 masked to 32 bits = 0xFFFFFFFF
    assert ev.evaluate(expr) == 0xFFFFFFFF
