"""Phase 5: DWARF expression evaluator.

A minimal implementation of the DWARF 5 expression stack machine sufficient
to resolve variable locations emitted by common RISC-V compilers (GCC and
clang with ``-O0``). The evaluator consumes the raw bytes of a
``DW_AT_location`` expression and a concrete machine state, and returns
either an address (if the expression describes a memory location) or a value
(if the expression ends with ``DW_OP_stack_value``).
"""

from __future__ import annotations


# DW_OP opcode constants — subset we understand.
DW_OP_addr = 0x03
DW_OP_deref = 0x06
DW_OP_const1u = 0x08
DW_OP_const1s = 0x09
DW_OP_const2u = 0x0a
DW_OP_const2s = 0x0b
DW_OP_const4u = 0x0c
DW_OP_const4s = 0x0d
DW_OP_const8u = 0x0e
DW_OP_const8s = 0x0f
DW_OP_constu = 0x10
DW_OP_consts = 0x11
DW_OP_dup = 0x12
DW_OP_drop = 0x13
DW_OP_over = 0x14
DW_OP_pick = 0x15
DW_OP_swap = 0x16
DW_OP_rot = 0x17
DW_OP_xderef = 0x18
DW_OP_abs = 0x19
DW_OP_and = 0x1a
DW_OP_div = 0x1b
DW_OP_minus = 0x1c
DW_OP_mod = 0x1d
DW_OP_mul = 0x1e
DW_OP_neg = 0x1f
DW_OP_not = 0x20
DW_OP_or = 0x21
DW_OP_plus = 0x22
DW_OP_plus_uconst = 0x23
DW_OP_shl = 0x24
DW_OP_shr = 0x25
DW_OP_shra = 0x26
DW_OP_xor = 0x27
DW_OP_skip = 0x2f
DW_OP_bra = 0x28
DW_OP_eq = 0x29
DW_OP_ge = 0x2a
DW_OP_gt = 0x2b
DW_OP_le = 0x2c
DW_OP_lt = 0x2d
DW_OP_ne = 0x2e
DW_OP_lit0 = 0x30
DW_OP_reg0 = 0x50
DW_OP_breg0 = 0x70
DW_OP_regx = 0x90
DW_OP_fbreg = 0x91
DW_OP_bregx = 0x92
DW_OP_piece = 0x93
DW_OP_deref_size = 0x94
DW_OP_stack_value = 0x9f


class DWARFExprEvaluator:
    """Evaluate a DWARF location expression against a concrete machine state.

    Usage::

        ev = DWARFExprEvaluator(registers, memory, frame_base, word_bytes)
        result = ev.evaluate(expr_bytes)
        if ev.is_value:
            # result is the variable's value
        else:
            # result is the address at which the variable lives
    """

    def __init__(
        self,
        registers: dict[int, int],
        memory: dict[int, int],
        frame_base: int = 0,
        word_bytes: int = 8,
    ) -> None:
        self.regs = registers
        self.mem = memory
        self.frame_base = frame_base
        self.word_bytes = word_bytes
        self.stack: list[int] = []
        self.is_value: bool = False

    # ----------------------------------------------------------------- top

    def evaluate(self, expr: bytes) -> int | None:
        """Run the expression; return the top of stack, or ``None`` if empty."""
        self.stack = []
        self.is_value = False
        i = 0
        while i < len(expr):
            op = expr[i]
            i += 1
            i = self._step(op, expr, i)
            if i < 0:
                return None
        return self.stack[-1] if self.stack else None

    # ---------------------------------------------------------- dispatcher

    def _step(self, op: int, expr: bytes, i: int) -> int:
        # Literal opcodes lit0..lit31.
        if DW_OP_lit0 <= op <= DW_OP_lit0 + 31:
            self.stack.append(op - DW_OP_lit0)
            return i
        # reg0..reg31: push register contents as the *location* (value is
        # in the register). For rotor's purposes we treat reg as value.
        if DW_OP_reg0 <= op <= DW_OP_reg0 + 31:
            self.stack.append(self.regs.get(op - DW_OP_reg0, 0))
            self.is_value = True
            return i
        # breg0..breg31: push register + signed LEB128 offset (address).
        if DW_OP_breg0 <= op <= DW_OP_breg0 + 31:
            offset, i = _read_sleb128(expr, i)
            self.stack.append(self._mask(self.regs.get(op - DW_OP_breg0, 0) + offset))
            return i

        handler = _DISPATCH.get(op)
        if handler is None:
            return -1
        return handler(self, expr, i)

    # --------------------------------------------------------- arith helpers

    def _mask(self, value: int) -> int:
        mask = (1 << (self.word_bytes * 8)) - 1
        return value & mask

    def _read_mem_word(self, addr: int, size: int | None = None) -> int | None:
        size = size or self.word_bytes
        value = 0
        for j in range(size):
            byte = self.mem.get(addr + j)
            if byte is None:
                return None
            value |= (byte & 0xFF) << (8 * j)
        return value


# ──────────────────────────────────────────────────────────────────────────
# Opcode handlers
# ──────────────────────────────────────────────────────────────────────────


def _h_addr(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
    addr = int.from_bytes(expr[i:i + ev.word_bytes], "little", signed=False)
    ev.stack.append(addr)
    return i + ev.word_bytes


def _h_deref(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
    if not ev.stack:
        return -1
    addr = ev.stack.pop()
    val = ev._read_mem_word(addr)
    if val is None:
        return -1
    ev.stack.append(val)
    return i


def _h_deref_size(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
    size = expr[i]
    if not ev.stack:
        return -1
    addr = ev.stack.pop()
    val = ev._read_mem_word(addr, size)
    if val is None:
        return -1
    ev.stack.append(val)
    return i + 1


def _const_bytes(size: int, signed: bool):
    def handler(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
        value = int.from_bytes(expr[i:i + size], "little", signed=signed)
        ev.stack.append(value)
        return i + size
    return handler


def _h_constu(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
    value, i = _read_uleb128(expr, i)
    ev.stack.append(value)
    return i


def _h_consts(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
    value, i = _read_sleb128(expr, i)
    ev.stack.append(ev._mask(value))
    return i


def _binary(fn):
    def handler(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
        if len(ev.stack) < 2:
            return -1
        b = ev.stack.pop()
        a = ev.stack.pop()
        ev.stack.append(ev._mask(fn(a, b)))
        return i
    return handler


def _h_plus_uconst(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
    if not ev.stack:
        return -1
    value, i = _read_uleb128(expr, i)
    ev.stack.append(ev._mask(ev.stack.pop() + value))
    return i


def _h_dup(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
    if not ev.stack:
        return -1
    ev.stack.append(ev.stack[-1])
    return i


def _h_drop(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
    if not ev.stack:
        return -1
    ev.stack.pop()
    return i


def _h_swap(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
    if len(ev.stack) < 2:
        return -1
    ev.stack[-1], ev.stack[-2] = ev.stack[-2], ev.stack[-1]
    return i


def _h_fbreg(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
    offset, i = _read_sleb128(expr, i)
    ev.stack.append(ev._mask(ev.frame_base + offset))
    return i


def _h_regx(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
    reg, i = _read_uleb128(expr, i)
    ev.stack.append(ev.regs.get(reg, 0))
    ev.is_value = True
    return i


def _h_bregx(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
    reg, i = _read_uleb128(expr, i)
    offset, i = _read_sleb128(expr, i)
    ev.stack.append(ev._mask(ev.regs.get(reg, 0) + offset))
    return i


def _h_stack_value(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
    ev.is_value = True
    return i


def _h_piece(ev: DWARFExprEvaluator, expr: bytes, i: int) -> int:
    # Piece boundaries are ignored; we surface the current top of stack.
    _, i = _read_uleb128(expr, i)
    return i


_DISPATCH: dict[int, object] = {
    DW_OP_addr: _h_addr,
    DW_OP_deref: _h_deref,
    DW_OP_deref_size: _h_deref_size,
    DW_OP_const1u: _const_bytes(1, False),
    DW_OP_const1s: _const_bytes(1, True),
    DW_OP_const2u: _const_bytes(2, False),
    DW_OP_const2s: _const_bytes(2, True),
    DW_OP_const4u: _const_bytes(4, False),
    DW_OP_const4s: _const_bytes(4, True),
    DW_OP_const8u: _const_bytes(8, False),
    DW_OP_const8s: _const_bytes(8, True),
    DW_OP_constu: _h_constu,
    DW_OP_consts: _h_consts,
    DW_OP_dup: _h_dup,
    DW_OP_drop: _h_drop,
    DW_OP_swap: _h_swap,
    DW_OP_plus: _binary(lambda a, b: a + b),
    DW_OP_minus: _binary(lambda a, b: a - b),
    DW_OP_mul: _binary(lambda a, b: a * b),
    DW_OP_and: _binary(lambda a, b: a & b),
    DW_OP_or: _binary(lambda a, b: a | b),
    DW_OP_xor: _binary(lambda a, b: a ^ b),
    DW_OP_plus_uconst: _h_plus_uconst,
    DW_OP_fbreg: _h_fbreg,
    DW_OP_regx: _h_regx,
    DW_OP_bregx: _h_bregx,
    DW_OP_stack_value: _h_stack_value,
    DW_OP_piece: _h_piece,
}


# ──────────────────────────────────────────────────────────────────────────
# LEB128 helpers
# ──────────────────────────────────────────────────────────────────────────


def _read_uleb128(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result, pos
        shift += 7


def _read_sleb128(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if (byte & 0x80) == 0:
            if byte & 0x40:  # sign bit of final byte
                result |= -(1 << shift)
            return result, pos
