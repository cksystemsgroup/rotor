"""L1: bitvector expression DAG.

DagBuilder is a drop-in replacement for `rotor.btor2.nodes.Model` that:

    1. Hash-conses every emitted node — identical sub-expressions are
       shared, so `build_reach` no longer emits duplicate constants,
       duplicate address-compute `add`s, or redundant sort declarations.
    2. Applies a small auditable set of local rewrites before emitting:
       constant folding, identity laws (x + 0, x & -1, ...), ITE
       reductions, extract-of-constant, and equality reflexivity.

Both behaviors are pure Python and stay inside the BTOR2 seam: the
output is still a Model, printable by `printer.py`, consumable by the
Z3 backend, and required to pass the L0-equivalence harness on the
full corpus.

The rewrites are intentionally limited. If a rule is not obviously
semantics-preserving over two's-complement bitvectors, it is not
here. BVDD (M9) is where set-theoretic reasoning enters; DAG stays
local.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rotor.btor2.nodes import ArraySort, Model, Node, Sort, AnySort

# Commutative operations — keyed by op name in BTOR2.
_COMMUTATIVE = frozenset({"add", "and", "or", "xor", "eq", "neq"})
# Boolean-result comparisons (width 1).
_CMP_OPS = frozenset({
    "eq", "neq", "ult", "ulte", "ugt", "ugte",
    "slt", "slte", "sgt", "sgte",
})


def _is_const(n: Node) -> bool:
    return n.kind == "const"


def _const_val(n: Node) -> int:
    assert _is_const(n)
    (value,) = n.operands
    return value


def _mask(width: int) -> int:
    return (1 << width) - 1


def _as_signed(value: int, width: int) -> int:
    sign = 1 << (width - 1)
    value &= _mask(width)
    return value - (1 << width) if value & sign else value


class DagBuilder(Model):
    """A Model that hash-conses and lightly simplifies as nodes are added."""

    def __init__(self) -> None:
        super().__init__()
        # cache keys vary by kind; one dict is enough.
        self._cons: dict[tuple, Node] = {}

    # ---- constants --------------------------------------------------

    def const(self, sort: Sort, value: int) -> Node:
        value = value & _mask(sort.width)
        key = ("const", sort.width, value)
        hit = self._cons.get(key)
        if hit is not None:
            return hit
        node = super().const(sort, value)
        self._cons[key] = node
        return node

    # ---- operations -------------------------------------------------

    def op(self, opname: str, sort: Sort, *args: Node) -> Node:
        simp = self._simplify_op(opname, sort, args)
        if simp is not None:
            return simp
        # Canonicalize argument order for commutative ops.
        key_args: tuple
        if opname in _COMMUTATIVE and len(args) == 2:
            key_args = tuple(sorted((args[0].id, args[1].id)))
        else:
            key_args = tuple(a.id for a in args)
        key = ("op", opname, sort.width, key_args)
        hit = self._cons.get(key)
        if hit is not None:
            return hit
        node = super().op(opname, sort, *args)
        self._cons[key] = node
        return node

    def ite(self, cond: Node, t: Node, e: Node) -> Node:
        # Constant guard -> collapse.
        if _is_const(cond):
            return t if _const_val(cond) & 1 else e
        # Both arms identical -> redundant.
        if t.id == e.id:
            return t
        assert t.sort == e.sort
        sort = t.sort
        assert sort is not None
        key = ("ite", self.sort_id_of(sort), cond.id, t.id, e.id)
        hit = self._cons.get(key)
        if hit is not None:
            return hit
        node = super().ite(cond, t, e)
        self._cons[key] = node
        return node

    def slice(self, a: Node, upper: int, lower: int) -> Node:
        # slice of constant -> new constant.
        if _is_const(a):
            value = _const_val(a)
            width = upper - lower + 1
            return self.const(Sort(width), (value >> lower) & _mask(width))
        # slice covering the full width -> identity.
        if isinstance(a.sort, Sort) and lower == 0 and upper == a.sort.width - 1:
            return a
        key = ("slice", a.id, upper, lower)
        hit = self._cons.get(key)
        if hit is not None:
            return hit
        node = super().slice(a, upper, lower)
        self._cons[key] = node
        return node

    def uext(self, a: Node, extra: int) -> Node:
        if extra == 0:
            return a
        if _is_const(a):
            assert isinstance(a.sort, Sort)
            return self.const(Sort(a.sort.width + extra), _const_val(a))
        key = ("uext", a.id, extra)
        hit = self._cons.get(key)
        if hit is not None:
            return hit
        node = super().uext(a, extra)
        self._cons[key] = node
        return node

    def sext(self, a: Node, extra: int) -> Node:
        if extra == 0:
            return a
        if _is_const(a):
            assert isinstance(a.sort, Sort)
            w = a.sort.width
            v = _const_val(a)
            signed = _as_signed(v, w)
            return self.const(Sort(w + extra), signed & _mask(w + extra))
        key = ("sext", a.id, extra)
        hit = self._cons.get(key)
        if hit is not None:
            return hit
        node = super().sext(a, extra)
        self._cons[key] = node
        return node

    def read(self, array: Node, addr: Node) -> Node:
        # Read-after-write forwarding: read(write(arr, a2, v), a1) — only
        # safe when a1 and a2 are both constants we can compare.
        if array.kind == "write":
            warr, waddr, wval = array.operands
            if _is_const(addr) and _is_const(waddr):
                if _const_val(addr) == _const_val(waddr):
                    return wval
                # Distinct constant addresses: skip past the write.
                return self.read(warr, addr)
        key = ("read", array.id, addr.id)
        hit = self._cons.get(key)
        if hit is not None:
            return hit
        node = super().read(array, addr)
        self._cons[key] = node
        return node

    def write(self, array: Node, addr: Node, value: Node) -> Node:
        # Writing back the same byte the array already holds is a no-op
        # — only detectable in the all-constant case.
        if _is_const(addr) and _is_const(value) and array.kind == "write":
            warr, waddr, wval = array.operands
            if _is_const(waddr) and _const_val(waddr) == _const_val(addr):
                # Later write wins; drop the inner write for the same addr.
                return self.write(warr, addr, value)
        key = ("write", array.id, addr.id, value.id)
        hit = self._cons.get(key)
        if hit is not None:
            return hit
        node = super().write(array, addr, value)
        self._cons[key] = node
        return node

    # ---- simplification table --------------------------------------

    def _simplify_op(self, opname: str, sort: Sort, args: tuple) -> Optional[Node]:
        if len(args) == 1:
            return self._simplify_unary(opname, sort, args[0])
        if len(args) == 2:
            return self._simplify_binary(opname, sort, args[0], args[1])
        return None

    def _simplify_unary(self, opname: str, sort: Sort, a: Node) -> Optional[Node]:
        if _is_const(a):
            v = _const_val(a)
            w = sort.width
            if opname == "not":
                return self.const(sort, (~v) & _mask(w))
            if opname == "neg":
                return self.const(sort, (-v) & _mask(w))
        return None

    def _simplify_binary(
        self, opname: str, sort: Sort, a: Node, b: Node
    ) -> Optional[Node]:
        w = sort.width
        if _is_const(a) and _is_const(b):
            folded = _fold_binary(opname, _const_val(a), _const_val(b), sort.width, a.sort, b.sort)
            if folded is not None:
                return self.const(sort, folded & _mask(w))

        # Identity / annihilator rules. Commute so constant (if any) sits on b.
        if _is_const(a) and not _is_const(b) and opname in _COMMUTATIVE:
            a, b = b, a
        if opname == "add":
            if _is_const(b) and _const_val(b) == 0:
                return a
        elif opname == "sub":
            if _is_const(b) and _const_val(b) == 0:
                return a
            if a.id == b.id:
                return self.const(sort, 0)
        elif opname == "and":
            if _is_const(b):
                v = _const_val(b)
                if v == 0:
                    return b
                if v == _mask(w):
                    return a
            if a.id == b.id:
                return a
        elif opname == "or":
            if _is_const(b):
                v = _const_val(b)
                if v == 0:
                    return a
                if v == _mask(w):
                    return b
            if a.id == b.id:
                return a
        elif opname == "xor":
            if _is_const(b) and _const_val(b) == 0:
                return a
            if a.id == b.id:
                return self.const(sort, 0)
        elif opname in ("sll", "srl", "sra"):
            if _is_const(b) and _const_val(b) == 0:
                return a
        elif opname == "eq":
            if a.id == b.id:
                return self.const(Sort(1), 1)
        elif opname == "neq":
            if a.id == b.id:
                return self.const(Sort(1), 0)
        return None


def _fold_binary(
    opname: str,
    av: int,
    bv: int,
    w: int,
    a_sort: AnySort | None,
    b_sort: AnySort | None,
) -> Optional[int]:
    """Fold a two-argument op applied to two constants."""
    # a_sort and b_sort are the operand sorts (not the result sort).
    if isinstance(a_sort, Sort):
        a_w = a_sort.width
    else:
        a_w = w
    if isinstance(b_sort, Sort):
        b_w = b_sort.width
    else:
        b_w = w

    if opname == "add":
        return (av + bv) & _mask(w)
    if opname == "sub":
        return (av - bv) & _mask(w)
    if opname == "and":
        return (av & bv) & _mask(w)
    if opname == "or":
        return (av | bv) & _mask(w)
    if opname == "xor":
        return (av ^ bv) & _mask(w)
    if opname == "sll":
        return (av << (bv & (w - 1))) & _mask(w)
    if opname == "srl":
        return (av & _mask(a_w)) >> (bv & (w - 1))
    if opname == "sra":
        signed = _as_signed(av, a_w)
        return (signed >> (bv & (w - 1))) & _mask(w)
    if opname == "concat":
        # concat a b -> (a << width(b)) | b in BTOR2 (high, low).
        return ((av << b_w) | bv) & _mask(a_w + b_w)
    if opname == "eq":
        return 1 if av == bv else 0
    if opname == "neq":
        return 0 if av == bv else 1
    if opname == "ult":
        return 1 if av < bv else 0
    if opname == "ulte":
        return 1 if av <= bv else 0
    if opname == "ugt":
        return 1 if av > bv else 0
    if opname == "ugte":
        return 1 if av >= bv else 0
    if opname == "slt":
        return 1 if _as_signed(av, a_w) < _as_signed(bv, b_w) else 0
    if opname == "slte":
        return 1 if _as_signed(av, a_w) <= _as_signed(bv, b_w) else 0
    if opname == "sgt":
        return 1 if _as_signed(av, a_w) > _as_signed(bv, b_w) else 0
    if opname == "sgte":
        return 1 if _as_signed(av, a_w) >= _as_signed(bv, b_w) else 0
    return None
