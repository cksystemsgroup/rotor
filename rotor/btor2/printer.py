"""Emit text BTOR2 from a Model.

This is the external seam rotor presents to solvers. Any BTOR2-accepting
tool (btormc, bitwuzla, rIC3, avr, abc) can consume the output. The Z3
backend in rotor consumes the in-memory Model directly; the printer is
used for debugging, archival, and cross-checking against external tools.
"""

from __future__ import annotations

from rotor.btor2.nodes import Model, Node


def to_text(model: Model) -> str:
    """Render the model as BTOR2 text."""
    lines: list[str] = []
    for node in model.nodes:
        lines.append(_line(node, model))
    return "\n".join(lines) + "\n"


def _line(node: Node, model: Model) -> str:
    if node.kind == "sort":
        (width,) = node.operands
        return f"{node.id} sort bitvec {width}"

    # init/next/bad are the only kinds that do not carry their own sort.
    # init/next borrow the state's sort; bad takes no sort.
    if node.kind == "init":
        state, expr = node.operands
        return f"{node.id} init {model.sort_id(state.sort.width)} {state.id} {expr.id}"
    if node.kind == "next":
        state, expr = node.operands
        return f"{node.id} next {model.sort_id(state.sort.width)} {state.id} {expr.id}"
    if node.kind == "bad":
        (expr,) = node.operands
        return f"{node.id} bad {expr.id}"

    sort_id = _sort_id(node, model)

    if node.kind == "const":
        (value,) = node.operands
        return f"{node.id} constd {sort_id} {value}"

    if node.kind == "input":
        return f"{node.id} input {sort_id} {node.name}"

    if node.kind == "state":
        return f"{node.id} state {sort_id} {node.name}"

    if node.kind == "op":
        refs = " ".join(str(o.id) for o in node.operands)
        return f"{node.id} {node.opname} {sort_id} {refs}"

    if node.kind == "ite":
        c, t, e = node.operands
        return f"{node.id} ite {sort_id} {c.id} {t.id} {e.id}"

    if node.kind == "slice":
        a, upper, lower = node.operands
        return f"{node.id} slice {sort_id} {a.id} {upper} {lower}"

    if node.kind == "ext":
        a, extra = node.operands
        return f"{node.id} {node.opname} {sort_id} {a.id} {extra}"

    raise AssertionError(f"unknown node kind: {node.kind}")


def _sort_id(node: Node, model: Model) -> int:
    assert node.sort is not None
    return model.sort_id(node.sort.width)
