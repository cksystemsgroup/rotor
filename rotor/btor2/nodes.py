"""BTOR2 node DAG.

A Model owns an ordered sequence of typed Nodes. Nodes come in a small
number of kinds, each matching a BTOR2 construct:

    sort <width>                  — bitvector sort
    constd <sort> <value>         — decimal constant
    input <sort> <name>           — symbolic input
    state <sort> <name>           — state variable
    op <sort> <args...>           — pure bitvector operation
    ite <sort> <c> <t> <e>        — ternary
    slice <sort> <a> <hi> <lo>    — bit range
    uext/sext <sort> <a> <n>      — extension by n bits
    init <state> <expr>           — initial constraint
    next <state> <expr>           — transition
    bad <expr>                    — property whose satisfiability is a bug

The in-memory design is deliberately uniform: everything is a Node with
an integer id, a kind string, a Sort (or None), and a tuple of operand
Nodes plus small scalar payload. This keeps the printer and the Z3
translator short, and makes hash-consing straightforward to add for L1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Sort:
    """Bitvector sort."""
    width: int

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"bitvec width must be positive: {self.width}")


@dataclass(frozen=True)
class Node:
    id: int
    kind: str                       # "const", "input", "state", "op", "ite",
                                    # "slice", "ext", "init", "next", "bad"
    sort: Optional[Sort]
    operands: tuple                 # tuple of Node / int / str (kind-dependent)
    name: Optional[str] = None
    opname: Optional[str] = None    # for kind == "op": "add", "eq", "ult", ...
                                    # for kind == "ext": "uext" | "sext"


class Model:
    """Ordered collection of BTOR2 nodes built incrementally."""

    def __init__(self) -> None:
        self._nodes: list[Node] = []
        self._sort_ids: dict[int, int] = {}   # width -> node id of sort

    # -- accessors -----------------------------------------------------

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes)

    def sort_id(self, width: int) -> int:
        """Return the BTOR2 line id of the sort node for the given width.

        Emits one if not already present. Sorts are 1-bit-identified by
        width in BTOR2; we hash-cons them.
        """
        nid = self._sort_ids.get(width)
        if nid is not None:
            return nid
        nid = len(self._nodes) + 1
        self._nodes.append(Node(id=nid, kind="sort", sort=None, operands=(width,)))
        self._sort_ids[width] = nid
        return nid

    # -- constructors --------------------------------------------------

    def const(self, sort: Sort, value: int) -> Node:
        self.sort_id(sort.width)
        value = value & ((1 << sort.width) - 1)
        return self._emit("const", sort, (value,))

    def const_bool(self, sort: Sort, *, true: bool) -> Node:
        assert sort.width == 1
        return self.const(sort, 1 if true else 0)

    def input(self, sort: Sort, name: str) -> Node:
        self.sort_id(sort.width)
        return self._emit("input", sort, (), name=name)

    def state(self, sort: Sort, name: str) -> Node:
        self.sort_id(sort.width)
        return self._emit("state", sort, (), name=name)

    def op(self, opname: str, sort: Sort, *args: Node) -> Node:
        self.sort_id(sort.width)
        return self._emit("op", sort, args, opname=opname)

    def ite(self, cond: Node, t: Node, e: Node) -> Node:
        assert cond.sort == Sort(1)
        assert t.sort == e.sort
        assert t.sort is not None
        self.sort_id(t.sort.width)
        return self._emit("ite", t.sort, (cond, t, e))

    def slice(self, a: Node, upper: int, lower: int) -> Node:
        assert a.sort is not None
        assert 0 <= lower <= upper < a.sort.width
        sort = Sort(upper - lower + 1)
        self.sort_id(sort.width)
        return self._emit("slice", sort, (a, upper, lower))

    def uext(self, a: Node, extra: int) -> Node:
        return self._ext("uext", a, extra)

    def sext(self, a: Node, extra: int) -> Node:
        return self._ext("sext", a, extra)

    def init(self, state: Node, expr: Node) -> Node:
        assert state.kind == "state"
        assert state.sort == expr.sort
        return self._emit("init", None, (state, expr))

    def next(self, state: Node, expr: Node) -> Node:
        assert state.kind == "state"
        assert state.sort == expr.sort
        return self._emit("next", None, (state, expr))

    def bad(self, expr: Node) -> Node:
        assert expr.sort == Sort(1)
        return self._emit("bad", None, (expr,))

    # -- internals -----------------------------------------------------

    def _ext(self, kind: str, a: Node, extra: int) -> Node:
        assert a.sort is not None
        assert extra >= 0
        sort = Sort(a.sort.width + extra)
        self.sort_id(sort.width)
        return self._emit("ext", sort, (a, extra), opname=kind)

    def _emit(
        self,
        kind: str,
        sort: Optional[Sort],
        operands: tuple,
        *,
        name: Optional[str] = None,
        opname: Optional[str] = None,
    ) -> Node:
        nid = len(self._nodes) + 1
        node = Node(id=nid, kind=kind, sort=sort, operands=operands, name=name, opname=opname)
        self._nodes.append(node)
        return node
