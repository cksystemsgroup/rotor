"""BTOR2 text emitter.

Walks a :class:`NodeDAG` in topological order, reassigns nids sequentially,
and writes valid BTOR2 text.
"""

from __future__ import annotations

from typing import IO

from rotor.btor2.nodes import Node, NodeDAG, Sort


class BTOR2Printer:
    """Render a :class:`NodeDAG` as BTOR2 text."""

    def __init__(self) -> None:
        self._nid_of_sort: dict[int, int] = {}
        self._nid_of_node: dict[int, int] = {}
        self._next_nid = 1

    # --------------------------------------------------------------- public

    def print(self, dag: NodeDAG, out: IO[str]) -> None:
        """Emit all sorts and nodes of ``dag`` to ``out``.

        All nodes in ``dag`` are emitted. Callers that want only a subset
        should construct a new DAG containing just those nodes.
        """
        self._nid_of_sort.clear()
        self._nid_of_node.clear()
        self._next_nid = 1

        for sort in dag.sorts():
            self._emit_sort(sort, out)

        # Topological order: emit children before parents, except for
        # 'sticky' nodes (state/input/init/next/bad/constraint) which we
        # emit in the original order to preserve semantic grouping.
        ordered = self._topological_order(dag)
        for node in ordered:
            self._emit_node(node, out)

    def render(self, dag: NodeDAG) -> str:
        import io

        buf = io.StringIO()
        self.print(dag, buf)
        return buf.getvalue()

    # ----------------------------------------------------------- internals

    def _topological_order(self, dag: NodeDAG) -> list[Node]:
        visited: set[int] = set()
        order: list[Node] = []

        def visit(node: Node) -> None:
            if node.nid in visited:
                return
            visited.add(node.nid)
            for arg in node.args:
                visit(arg)
            order.append(node)

        for node in dag.nodes():
            visit(node)
        return order

    def _sort_nid(self, sort: Sort) -> int:
        return self._nid_of_sort[sort.nid]

    def _node_nid(self, node: Node) -> int:
        return self._nid_of_node[node.nid]

    def _emit_sort(self, sort: Sort, out: IO[str]) -> None:
        nid = self._next_nid
        self._next_nid += 1
        self._nid_of_sort[sort.nid] = nid
        if sort.kind == "bitvec":
            line = f"{nid} sort bitvec {sort.width}"
        else:
            assert sort.index_sort is not None and sort.elem_sort is not None
            idx = self._sort_nid(sort.index_sort)
            elem = self._sort_nid(sort.elem_sort)
            line = f"{nid} sort array {idx} {elem}"
        if sort.comment:
            line += f" ; {sort.comment}"
        out.write(line + "\n")

    def _emit_node(self, node: Node, out: IO[str]) -> None:
        nid = self._next_nid
        self._next_nid += 1
        self._nid_of_node[node.nid] = nid

        parts: list[str] = [str(nid), node.op, str(self._sort_nid(node.sort))]

        if node.op in ("zero", "one", "ones"):
            pass
        elif node.op == "constd":
            parts.append(str(node.params[0]))
        elif node.op == "consth":
            parts.append(f"{node.params[0]:x}")
        elif node.op == "const":
            value, width = node.params[0], node.params[1]
            parts.append(format(value, f"0{width}b"))
        elif node.op in ("state", "input"):
            # 'state s [symbol]' — symbol handled below.
            pass
        elif node.op in ("sext", "uext"):
            parts.append(str(self._node_nid(node.args[0])))
            parts.append(str(node.params[0]))
        elif node.op == "slice":
            parts.append(str(self._node_nid(node.args[0])))
            parts.extend(str(p) for p in node.params[:2])
        elif node.op == "justice":
            parts.append(str(node.params[0]))
            parts.extend(str(self._node_nid(a)) for a in node.args)
        else:
            parts.extend(str(self._node_nid(a)) for a in node.args)

        if node.symbol:
            parts.append(node.symbol)
        line = " ".join(parts)
        if node.comment:
            line += f" ; {node.comment}"
        out.write(line + "\n")
