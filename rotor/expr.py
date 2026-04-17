"""Condition expression language for :class:`RotorAPI`.

A compact expression grammar that maps to BTOR2 nodes, so users of
:class:`RotorAPI` can write conditions like

    pc == 0x1040
    a0 < 0
    x3 == x4
    x5 & 0xff != 0

without hand-building Node graphs. Variables:

    pc                            — program counter of core 0
    x0..x31                       — integer register by number
    a0..a7, t0..t6, s0..s11,      — ABI register names
    zero, ra, sp, gp, tp, fp
    core<N>.pc, core<N>.x<K>      — multicore qualification
    integer literals              — 0x1040, 0b0001, 42, -1

Operators (standard precedence):

    unary     :  !  ~  -
    mul/div   :  *  /  %
    add/sub   :  +  -
    shift     :  <<  >>  >>>
    bitwise   :  &  ^  |
    compare   :  ==  !=  <  <=  >  >=  <u  <=u  >u  >=u
    logical   :  &&  ||
    ternary   :  cond ? a : b

All operations are bitvector-typed at the machine word width of the instance.
Signed vs unsigned comparisons: the default (``<`` etc.) is signed; append
``u`` (``<u``, ``>=u``) for unsigned. ``==``/``!=`` reduce to a 1-bit node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover
    from rotor.btor2 import Node, Sort
    from rotor.instance import RotorInstance


# ──────────────────────────────────────────────────────────────────────────
# ABI register name → number
# ──────────────────────────────────────────────────────────────────────────

_ABI = {
    "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
    "t0": 5, "t1": 6, "t2": 7,
    "s0": 8, "fp": 8, "s1": 9,
    "a0": 10, "a1": 11, "a2": 12, "a3": 13, "a4": 14, "a5": 15, "a6": 16, "a7": 17,
    "s2": 18, "s3": 19, "s4": 20, "s5": 21, "s6": 22, "s7": 23, "s8": 24, "s9": 25,
    "s10": 26, "s11": 27,
    "t3": 28, "t4": 29, "t5": 30, "t6": 31,
}


# ──────────────────────────────────────────────────────────────────────────
# Tokens
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class _Tok:
    kind: str        # 'ident', 'int', 'op', 'lparen', 'rparen', 'end'
    text: str
    value: int | None = None


def _tokenize(src: str) -> list[_Tok]:
    tokens: list[_Tok] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_" or src[j] == "."):
                j += 1
            tokens.append(_Tok("ident", src[i:j]))
            i = j
        elif c.isdigit() or (c == "-" and i + 1 < n and src[i + 1].isdigit()
                              and (not tokens or tokens[-1].kind in ("op", "lparen"))):
            j = i + (1 if c == "-" else 0)
            if j + 1 < n and src[j] == "0" and src[j + 1] in "xXbB":
                j += 2
                while j < n and (src[j].isalnum() or src[j] == "_"):
                    j += 1
            else:
                while j < n and (src[j].isdigit() or src[j] == "_"):
                    j += 1
            raw = src[i:j].replace("_", "")
            value = int(raw, 0)
            tokens.append(_Tok("int", src[i:j], value=value))
            i = j
        elif c == "(":
            tokens.append(_Tok("lparen", c))
            i += 1
        elif c == ")":
            tokens.append(_Tok("rparen", c))
            i += 1
        else:
            # Multi-char operators first, longest match.
            for op in ("<<", ">>>", ">>", "<=u", ">=u", "<u", ">u",
                       "==", "!=", "<=", ">=", "&&", "||", "?", ":"):
                if src.startswith(op, i):
                    tokens.append(_Tok("op", op))
                    i += len(op)
                    break
            else:
                if c in "+-*/%&|^~!<>":
                    tokens.append(_Tok("op", c))
                    i += 1
                else:
                    raise ValueError(f"expression: unexpected character {c!r} at {i}")
    tokens.append(_Tok("end", ""))
    return tokens


# ──────────────────────────────────────────────────────────────────────────
# Compiler
# ──────────────────────────────────────────────────────────────────────────


class ExpressionCompiler:
    """Compile a condition expression against a :class:`RotorInstance`.

    Construct per-instance (the compiler captures the instance's builder,
    machine-word sort, etc.) or invoke statically via :meth:`compile`.
    """

    def __init__(self, instance: "RotorInstance") -> None:
        self.instance = instance
        model = instance.model
        if model.builder is None:
            raise RuntimeError(
                "ExpressionCompiler: requires an instance built with "
                "model_backend='python' so a BTOR2 builder is available"
            )
        self.builder = model.builder  # type: ignore[attr-defined]
        self.word_sort: "Sort" = self.builder.SID_MACHINE_WORD
        self.bool_sort: "Sort" = self.builder.SID_BOOLEAN

    # ----------------------------------------------------- public

    @classmethod
    def compile(cls, instance: "RotorInstance", expression: str) -> "Node":
        return cls(instance)._compile(expression)

    def _compile(self, expression: str) -> "Node":
        self._tokens = _tokenize(expression)
        self._pos = 0
        node = self._parse_ternary()
        self._expect("end")
        # Top-level expressions must reduce to a 1-bit predicate.
        return self._to_bool(node)

    # ----------------------------------------------------- token helpers

    def _peek(self) -> _Tok:
        return self._tokens[self._pos]

    def _advance(self) -> _Tok:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _accept(self, kind: str, text: str | None = None) -> _Tok | None:
        tok = self._peek()
        if tok.kind == kind and (text is None or tok.text == text):
            return self._advance()
        return None

    def _expect(self, kind: str, text: str | None = None) -> _Tok:
        tok = self._accept(kind, text)
        if tok is None:
            cur = self._peek()
            raise ValueError(f"expression: expected {kind}{'' if text is None else f' {text!r}'} at {cur.text!r}")
        return tok

    # ----------------------------------------------------- precedence levels

    def _parse_ternary(self) -> "Node":
        cond = self._parse_or()
        if self._accept("op", "?"):
            lhs = self._parse_ternary()
            self._expect("op", ":")
            rhs = self._parse_ternary()
            lhs_w = self._to_word(lhs)
            rhs_w = self._to_word(rhs)
            return self.builder.ite(self._to_bool(cond), lhs_w, rhs_w, "?:")
        return cond

    def _parse_or(self) -> "Node":
        left = self._parse_and()
        while self._accept("op", "||"):
            right = self._parse_and()
            left = self.builder.or_(self._to_bool(left), self._to_bool(right), "||")
        return left

    def _parse_and(self) -> "Node":
        left = self._parse_bitor()
        while self._accept("op", "&&"):
            right = self._parse_bitor()
            left = self.builder.and_(self._to_bool(left), self._to_bool(right), "&&")
        return left

    def _parse_bitor(self) -> "Node":
        left = self._parse_bitxor()
        while self._accept("op", "|"):
            right = self._parse_bitxor()
            left = self.builder.or_(self._to_word(left), self._to_word(right), "|")
        return left

    def _parse_bitxor(self) -> "Node":
        left = self._parse_bitand()
        while self._accept("op", "^"):
            right = self._parse_bitand()
            left = self.builder.xor(self._to_word(left), self._to_word(right), "^")
        return left

    def _parse_bitand(self) -> "Node":
        left = self._parse_compare()
        while self._accept("op", "&"):
            right = self._parse_compare()
            left = self.builder.and_(self._to_word(left), self._to_word(right), "&")
        return left

    _CMP_OPS: dict[str, str] = {
        "==": "eq", "!=": "neq",
        "<": "slt", "<=": "slte", ">": "sgt", ">=": "sgte",
        "<u": "ult", "<=u": "ulte", ">u": "ugt", ">=u": "ugte",
    }

    def _parse_compare(self) -> "Node":
        left = self._parse_shift()
        tok = self._peek()
        if tok.kind == "op" and tok.text in self._CMP_OPS:
            self._advance()
            right = self._parse_shift()
            method = getattr(self.builder, self._CMP_OPS[tok.text])
            return method(self._to_word(left), self._to_word(right), tok.text)
        return left

    def _parse_shift(self) -> "Node":
        left = self._parse_addsub()
        while True:
            tok = self._peek()
            if tok.kind == "op" and tok.text == "<<":
                self._advance()
                right = self._parse_addsub()
                left = self.builder.sll(self._to_word(left), self._to_word(right), "<<")
            elif tok.kind == "op" and tok.text == ">>":
                self._advance()
                right = self._parse_addsub()
                left = self.builder.sra(self._to_word(left), self._to_word(right), ">>")
            elif tok.kind == "op" and tok.text == ">>>":
                self._advance()
                right = self._parse_addsub()
                left = self.builder.srl(self._to_word(left), self._to_word(right), ">>>")
            else:
                break
        return left

    def _parse_addsub(self) -> "Node":
        left = self._parse_muldiv()
        while True:
            tok = self._peek()
            if tok.kind == "op" and tok.text == "+":
                self._advance()
                right = self._parse_muldiv()
                left = self.builder.add(self._to_word(left), self._to_word(right), "+")
            elif tok.kind == "op" and tok.text == "-":
                self._advance()
                right = self._parse_muldiv()
                left = self.builder.sub(self._to_word(left), self._to_word(right), "-")
            else:
                break
        return left

    def _parse_muldiv(self) -> "Node":
        left = self._parse_unary()
        while True:
            tok = self._peek()
            if tok.kind == "op" and tok.text == "*":
                self._advance()
                right = self._parse_unary()
                left = self.builder.mul(self._to_word(left), self._to_word(right), "*")
            elif tok.kind == "op" and tok.text == "/":
                self._advance()
                right = self._parse_unary()
                left = self.builder.sdiv(self._to_word(left), self._to_word(right), "/")
            elif tok.kind == "op" and tok.text == "%":
                self._advance()
                right = self._parse_unary()
                left = self.builder.srem(self._to_word(left), self._to_word(right), "%")
            else:
                break
        return left

    def _parse_unary(self) -> "Node":
        tok = self._peek()
        if tok.kind == "op" and tok.text == "!":
            self._advance()
            node = self._parse_unary()
            return self.builder.not_(self._to_bool(node), "!")
        if tok.kind == "op" and tok.text == "~":
            self._advance()
            node = self._parse_unary()
            return self.builder.not_(self._to_word(node), "~")
        if tok.kind == "op" and tok.text == "-":
            self._advance()
            node = self._parse_unary()
            return self.builder.neg(self._to_word(node), "-")
        return self._parse_primary()

    def _parse_primary(self) -> "Node":
        tok = self._advance()
        if tok.kind == "lparen":
            node = self._parse_ternary()
            self._expect("rparen")
            return node
        if tok.kind == "int":
            assert tok.value is not None
            width = self.word_sort.width or 0
            value = tok.value & ((1 << width) - 1)
            return self.builder.constd(self.word_sort, value, str(tok.value))
        if tok.kind == "ident":
            return self._resolve_identifier(tok.text)
        raise ValueError(f"expression: unexpected token {tok.text!r}")

    # ----------------------------------------------------- identifiers

    def _resolve_identifier(self, name: str) -> "Node":
        core, base = self._split_core(name)
        if base == "pc":
            return self._core_pc(core)
        if base in _ABI:
            return self._core_reg(core, _ABI[base])
        if base.startswith("x") and base[1:].isdigit():
            idx = int(base[1:])
            if 0 <= idx < 32:
                return self._core_reg(core, idx)
        raise ValueError(f"expression: unknown identifier {name!r}")

    @staticmethod
    def _split_core(name: str) -> tuple[int, str]:
        if name.startswith("core") and "." in name:
            head, base = name.split(".", 1)
            return int(head[4:]), base
        return 0, name

    def _core_pc(self, core: int) -> "Node":
        state_nodes = self.instance.model.state_nodes
        # The Python builder may not have created a pc node yet when only
        # the register file and memory scaffolding are present; in that case
        # the caller should use the C Rotor backend.
        key_candidates = [f"core{core}-pc" if self.instance.config.cores > 1 else "pc",
                          f"core{core}.pc", "pc"]
        for key in key_candidates:
            if key in state_nodes:
                return state_nodes[key]
        raise KeyError(f"pc state node for core {core} not found in model")

    def _core_reg(self, core: int, reg: int) -> "Node":
        return self.instance.model.register_value(core=core, reg=reg)

    # ----------------------------------------------------- coercions

    def _to_bool(self, node: "Node") -> "Node":
        if node.sort is self.bool_sort:
            return node
        zero = self.builder.zero(node.sort, "zero")
        return self.builder.neq(node, zero, "to-bool")

    def _to_word(self, node: "Node") -> "Node":
        if node.sort is self.word_sort:
            return node
        # 1-bit predicate → zero-extend to word width.
        if node.sort.kind == "bitvec" and node.sort.width == 1:
            width = (self.word_sort.width or 0) - 1
            return self.builder.uext(self.word_sort, node, width, "uext")
        if node.sort.kind == "bitvec" and (node.sort.width or 0) < (self.word_sort.width or 0):
            width = (self.word_sort.width or 0) - (node.sort.width or 0)
            return self.builder.uext(self.word_sort, node, width, "uext")
        return node


# ──────────────────────────────────────────────────────────────────────────
# Public factory
# ──────────────────────────────────────────────────────────────────────────


def default_condition_compiler() -> Callable[["RotorInstance", str], "Node"]:
    """Return a compiler suitable for :class:`~rotor.api.RotorAPI`."""
    return ExpressionCompiler.compile
