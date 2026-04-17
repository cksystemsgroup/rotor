"""Minimal BTOR2 parser.

Reads BTOR2 text (as produced by C Rotor or any conformant emitter) into a
:class:`NodeDAG`. This parser accepts the subset of BTOR2 used by Rotor:
``sort``, constants, unary/binary/ternary logic, ``state``, ``input``,
``init``, ``next``, ``bad``, ``constraint``, ``justice``, and slicing.
"""

from __future__ import annotations

from rotor.btor2.nodes import Node, NodeDAG, Sort


# Operators that take no arguments besides their sort.
_NULLARY = {"zero", "one", "ones"}
# Operators with a single node argument.
_UNARY = {"not", "inc", "dec", "neg", "redor", "redand", "redxor"}
# Operators with two node arguments.
_BINARY = {
    "add", "sub", "mul", "udiv", "urem", "sdiv", "srem", "smod",
    "and", "or", "xor", "nand", "nor", "xnor",
    "sll", "srl", "sra", "rol", "ror",
    "concat",
    "eq", "neq", "ult", "ulte", "ugt", "ugte",
    "slt", "slte", "sgt", "sgte",
    "read",
    "uaddo", "saddo", "usubo", "ssubo", "umulo", "smulo", "sdivo",
}
# Operators with three node arguments.
_TERNARY = {"ite", "write"}
# Ops we emit as nested init/next relationships.
_SEQ_BINARY = {"init", "next"}


def parse_btor2(text: str) -> NodeDAG:
    """Parse BTOR2 text into a :class:`NodeDAG`.

    The parser is permissive: unknown operators are preserved as-is with
    their raw arguments, so round-tripping through :class:`BTOR2Printer`
    remains possible.
    """
    dag = NodeDAG()

    # We need to track the mapping from the file's nids to the DAG's nids
    # (the DAG allocates its own, interleaved across sorts and nodes). We
    # also track file-nid → Sort for sort lookups.
    sort_by_file_nid: dict[int, Sort] = {}
    node_by_file_nid: dict[int, Node] = {}

    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        try:
            file_nid = int(parts[0])
        except ValueError:
            continue
        op = parts[1]
        rest = parts[2:]
        comment_idx = raw.find(";")
        comment = raw[comment_idx + 1:].strip() if comment_idx >= 0 else ""

        if op == "sort":
            kind = rest[0]
            if kind == "bitvec":
                width = int(rest[1])
                sort = dag.intern_sort("bitvec", width=width, comment=comment)
            elif kind == "array":
                idx = sort_by_file_nid[int(rest[1])]
                elem = sort_by_file_nid[int(rest[2])]
                sort = dag.intern_sort(
                    "array", index_sort=idx, elem_sort=elem, comment=comment
                )
            else:
                raise ValueError(
                    f"BTOR2 parser: unsupported sort kind {kind!r} (line {line_no})"
                )
            sort_by_file_nid[file_nid] = sort
            continue

        # Everything else: the first ``rest`` token is the result sort.
        sort = sort_by_file_nid[int(rest[0])]
        tail = rest[1:]

        symbol = ""
        args: list[Node] = []
        params: list[int] = []

        if op in _NULLARY:
            pass
        elif op == "constd":
            params = [int(tail[0])]
        elif op == "consth":
            params = [int(tail[0], 16)]
        elif op == "const":
            binary = tail[0]
            params = [int(binary, 2), len(binary)]
        elif op in ("state", "input"):
            if tail:
                symbol = tail[0]
        elif op in ("sext", "uext"):
            args = [node_by_file_nid[int(tail[0])]]
            params = [int(tail[1])]
        elif op == "slice":
            args = [node_by_file_nid[int(tail[0])]]
            params = [int(tail[1]), int(tail[2])]
        elif op == "justice":
            n = int(tail[0])
            args = [node_by_file_nid[int(t)] for t in tail[1:1 + n]]
            params = [n]
            if len(tail) > 1 + n:
                symbol = tail[1 + n]
        elif op in _UNARY:
            args = [node_by_file_nid[int(tail[0])]]
        elif op in _BINARY or op in _SEQ_BINARY:
            args = [node_by_file_nid[int(tail[0])], node_by_file_nid[int(tail[1])]]
        elif op in _TERNARY:
            args = [
                node_by_file_nid[int(tail[0])],
                node_by_file_nid[int(tail[1])],
                node_by_file_nid[int(tail[2])],
            ]
        elif op in ("bad", "constraint", "fair", "output"):
            args = [node_by_file_nid[int(tail[0])]]
            if len(tail) > 1:
                symbol = tail[1]
        else:
            # Unknown op: take all remaining tokens as node references if
            # they parse as ints, else as the symbol.
            for tok in tail:
                try:
                    args.append(node_by_file_nid[int(tok)])
                except (ValueError, KeyError):
                    symbol = tok
                    break

        node = dag.get_or_create(
            op=op,
            sort=sort,
            args=args,
            params=params,
            symbol=symbol,
            comment=comment,
        )
        node_by_file_nid[file_nid] = node

    return dag
