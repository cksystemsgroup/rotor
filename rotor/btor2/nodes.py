"""BTOR2 node DAG with structural sharing.

The DAG mirrors C Rotor's invariant that pointer equivalence implies
structural equivalence: two nodes constructed with the same operator, sort,
arguments and parameters are represented by the same Python object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# ──────────────────────────────────────────────────────────────────────────
# Sorts
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Sort:
    """A BTOR2 sort: ``bitvec w`` or ``array index_sort elem_sort``."""

    nid: int
    kind: str  # 'bitvec' or 'array'
    width: int | None = None
    index_sort: "Sort | None" = None
    elem_sort: "Sort | None" = None
    comment: str = ""

    def __hash__(self) -> int:
        return hash((self.kind, self.width,
                     self.index_sort.nid if self.index_sort else None,
                     self.elem_sort.nid if self.elem_sort else None))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sort):
            return NotImplemented
        return (self.kind == other.kind and self.width == other.width
                and self.index_sort is other.index_sort
                and self.elem_sort is other.elem_sort)

    def describe(self) -> str:
        if self.kind == "bitvec":
            return f"bitvec {self.width}"
        return (f"array {self.index_sort.describe() if self.index_sort else '?'} "
                f"-> {self.elem_sort.describe() if self.elem_sort else '?'}")


# ──────────────────────────────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Node:
    """A single BTOR2 node.

    ``op`` is the BTOR2 operator name (``'add'``, ``'ite'``, ``'state'``,
    ``'next'``, ``'bad'``, ...). ``args`` are the referenced nodes;
    ``params`` are integer parameters that some ops carry (literal values for
    ``constd``/``consth``, width/cutoff for ``sext``/``slice``). ``symbol``
    is used by ``state`` and ``input`` nodes.
    """

    nid: int
    op: str
    sort: Sort
    args: list["Node"] = field(default_factory=list)
    params: list[int] = field(default_factory=list)
    symbol: str = ""
    comment: str = ""

    # Evaluation cache (Phase 5): useful for concrete witness replay.
    _concrete_value: int | None = None
    _is_symbolic: bool = False

    def __repr__(self) -> str:
        return f"Node(nid={self.nid}, op={self.op!r}, sort={self.sort.describe()})"


# ──────────────────────────────────────────────────────────────────────────
# DAG
# ──────────────────────────────────────────────────────────────────────────


class NodeDAG:
    """Structurally-shared BTOR2 node DAG.

    Nodes are keyed by (op, sort.nid, arg nids, params, symbol) so that
    identical requests return the same Python object. Sorts are tracked in
    insertion order so a printer can topologically emit them.
    """

    def __init__(self) -> None:
        self._next_nid = 1
        self._nodes: dict[int, Node] = {}
        self._dedup: dict[tuple, Node] = {}
        self._sorts: list[Sort] = []
        self._sort_dedup: dict[tuple, Sort] = {}

    # ------------------------------------------------------------- sort pool

    def intern_sort(
        self,
        kind: str,
        width: int | None = None,
        index_sort: Sort | None = None,
        elem_sort: Sort | None = None,
        comment: str = "",
    ) -> Sort:
        key = (kind, width,
               index_sort.nid if index_sort else None,
               elem_sort.nid if elem_sort else None)
        if key in self._sort_dedup:
            return self._sort_dedup[key]
        s = Sort(
            nid=self._next_nid,
            kind=kind,
            width=width,
            index_sort=index_sort,
            elem_sort=elem_sort,
            comment=comment,
        )
        self._next_nid += 1
        self._sort_dedup[key] = s
        self._sorts.append(s)
        return s

    # ------------------------------------------------------------- node pool

    def _key(
        self,
        op: str,
        sort: Sort,
        args: Iterable[Node],
        params: Iterable[int],
        symbol: str,
    ) -> tuple:
        return (
            op,
            sort.nid,
            tuple(a.nid for a in args),
            tuple(params),
            symbol,
        )

    def get_or_create(
        self,
        op: str,
        sort: Sort,
        args: Iterable[Node] = (),
        params: Iterable[int] = (),
        symbol: str = "",
        comment: str = "",
    ) -> Node:
        arg_list = list(args)
        param_list = list(params)
        key = self._key(op, sort, arg_list, param_list, symbol)
        # Shared nodes only make sense for stateless ops; state/input/next/
        # init/bad/constraint must remain unique even when arg-structurally
        # identical, because they represent distinct latches or properties.
        sticky = op in ("state", "input", "next", "init", "bad", "constraint")
        if not sticky and key in self._dedup:
            return self._dedup[key]
        node = Node(
            nid=self._next_nid,
            op=op,
            sort=sort,
            args=arg_list,
            params=param_list,
            symbol=symbol,
            comment=comment,
        )
        self._next_nid += 1
        self._nodes[node.nid] = node
        if not sticky:
            self._dedup[key] = node
        return node

    # ---------------------------------------------------------- introspection

    def __len__(self) -> int:
        return len(self._nodes) + len(self._sorts)

    def nodes(self) -> list[Node]:
        return [self._nodes[nid] for nid in sorted(self._nodes)]

    def sorts(self) -> list[Sort]:
        return list(self._sorts)

    def by_nid(self, nid: int) -> Node | None:
        return self._nodes.get(nid)


# ──────────────────────────────────────────────────────────────────────────
# MachineModel aggregate
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class MachineModel:
    """Outcome of a machine builder: DAG, state and property indexes."""

    dag: NodeDAG
    state_nodes: dict[str, Node] = field(default_factory=dict)
    input_nodes: dict[str, Node] = field(default_factory=dict)
    init_nodes: list[Node] = field(default_factory=list)
    next_nodes: list[Node] = field(default_factory=list)
    property_nodes: list[Node] = field(default_factory=list)  # 'bad' nodes
    constraint_nodes: list[Node] = field(default_factory=list)
    builder: "object | None" = None  # back-pointer to RISCVMachineBuilder

    def register_value(self, core: int, reg: int) -> Node:
        """Return the Node for register ``reg`` of ``core`` (see builder)."""
        if self.builder is None:
            raise RuntimeError("MachineModel has no builder reference")
        return self.builder.register_value(core, reg)  # type: ignore[attr-defined]
