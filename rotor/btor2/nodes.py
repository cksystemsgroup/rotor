"""BTOR2 node DAG.

A Model owns an ordered sequence of typed Nodes. Nodes come in a small
number of kinds, each matching a BTOR2 construct:

    sort <width>                    — bitvector sort
    sort array <index> <element>    — array sort (M6)
    constd <sort> <value>           — decimal constant
    input <sort> <name>             — symbolic input
    state <sort> <name>             — state variable (bv or array)
    op <sort> <args...>             — pure bitvector operation
    ite <sort> <c> <t> <e>          — ternary
    slice <sort> <a> <hi> <lo>      — bit range
    uext/sext <sort> <a> <n>        — extension by n bits
    read  <element> <array> <addr>  — M6: array read
    write <array> <array> <addr> <val> — M6: array update
    init <state> <expr>             — initial constraint
    next <state> <expr>             — transition
    bad <expr>                      — property whose satisfiability is a bug

The in-memory design is deliberately uniform: everything is a Node with
an integer id, a kind string, a Sort (or None), and a tuple of operand
Nodes plus small scalar payload. This keeps the printer and the Z3
translator short, and makes hash-consing straightforward to add for L1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class Sort:
    """Bitvector sort."""
    width: int

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"bitvec width must be positive: {self.width}")


@dataclass(frozen=True)
class ArraySort:
    """Array sort: index bitvector -> element bitvector (M6)."""
    index: Sort
    element: Sort


AnySort = Union[Sort, ArraySort]


@dataclass(frozen=True)
class Node:
    id: int
    kind: str                       # "const", "input", "state", "op", "ite",
                                    # "slice", "ext", "read", "write",
                                    # "init", "next", "bad"
    sort: Optional[AnySort]
    operands: tuple                 # tuple of Node / int / str (kind-dependent)
    name: Optional[str] = None
    opname: Optional[str] = None    # for kind == "op": "add", "eq", "ult", ...
                                    # for kind == "ext": "uext" | "sext"


class Model:
    """Ordered collection of BTOR2 nodes built incrementally."""

    def __init__(self) -> None:
        self._nodes: list[Node] = []
        self._bv_sort_ids: dict[int, int] = {}             # width -> node id
        self._array_sort_ids: dict[tuple[int, int], int] = {}  # (index_w, elem_w) -> id

    # -- accessors -----------------------------------------------------

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes)

    def sort_id(self, width: int) -> int:
        """Return the BTOR2 line id of the bitvector sort of the given width."""
        nid = self._bv_sort_ids.get(width)
        if nid is not None:
            return nid
        nid = len(self._nodes) + 1
        self._nodes.append(Node(id=nid, kind="sort", sort=None, operands=(width,)))
        self._bv_sort_ids[width] = nid
        return nid

    def array_sort_id(self, index: Sort, element: Sort) -> int:
        """Return the BTOR2 line id of the array sort (index -> element)."""
        key = (index.width, element.width)
        nid = self._array_sort_ids.get(key)
        if nid is not None:
            return nid
        # Ensure the bitvec sorts are declared first.
        idx_sid = self.sort_id(index.width)
        elt_sid = self.sort_id(element.width)
        nid = len(self._nodes) + 1
        self._nodes.append(
            Node(id=nid, kind="array_sort", sort=None, operands=(idx_sid, elt_sid))
        )
        self._array_sort_ids[key] = nid
        return nid

    def sort_id_of(self, sort: AnySort) -> int:
        """Return the BTOR2 line id for any sort (bitvec or array)."""
        if isinstance(sort, ArraySort):
            return self.array_sort_id(sort.index, sort.element)
        return self.sort_id(sort.width)

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

    def state(self, sort: AnySort, name: str) -> Node:
        """Declare a state variable (bitvec or array)."""
        self.sort_id_of(sort)
        return self._emit("state", sort, (), name=name)

    def op(self, opname: str, sort: Sort, *args: Node) -> Node:
        self.sort_id(sort.width)
        return self._emit("op", sort, args, opname=opname)

    def ite(self, cond: Node, t: Node, e: Node) -> Node:
        assert cond.sort == Sort(1)
        assert t.sort == e.sort
        assert t.sort is not None
        self.sort_id_of(t.sort)
        return self._emit("ite", t.sort, (cond, t, e))

    def slice(self, a: Node, upper: int, lower: int) -> Node:
        assert isinstance(a.sort, Sort)
        assert 0 <= lower <= upper < a.sort.width
        sort = Sort(upper - lower + 1)
        self.sort_id(sort.width)
        return self._emit("slice", sort, (a, upper, lower))

    def uext(self, a: Node, extra: int) -> Node:
        return self._ext("uext", a, extra)

    def sext(self, a: Node, extra: int) -> Node:
        return self._ext("sext", a, extra)

    def read(self, array: Node, addr: Node) -> Node:
        """Array read: array[addr] -> element."""
        assert isinstance(array.sort, ArraySort)
        assert isinstance(addr.sort, Sort)
        assert addr.sort == array.sort.index
        sort = array.sort.element
        self.sort_id(sort.width)
        return self._emit("read", sort, (array, addr))

    def write(self, array: Node, addr: Node, value: Node) -> Node:
        """Array update: array with [addr] = value, same array sort."""
        assert isinstance(array.sort, ArraySort)
        assert isinstance(addr.sort, Sort)
        assert isinstance(value.sort, Sort)
        assert addr.sort == array.sort.index
        assert value.sort == array.sort.element
        self.array_sort_id(array.sort.index, array.sort.element)
        return self._emit("write", array.sort, (array, addr, value))

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
        assert isinstance(a.sort, Sort)
        assert extra >= 0
        sort = Sort(a.sort.width + extra)
        self.sort_id(sort.width)
        return self._emit("ext", sort, (a, extra), opname=kind)

    def _emit(
        self,
        kind: str,
        sort: Optional[AnySort],
        operands: tuple,
        *,
        name: Optional[str] = None,
        opname: Optional[str] = None,
    ) -> Node:
        nid = len(self._nodes) + 1
        node = Node(id=nid, kind=kind, sort=sort, operands=operands, name=name, opname=opname)
        self._nodes.append(node)
        return node
