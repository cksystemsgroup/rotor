"""BTOR2 text printer.

Canonical form: one line per :class:`Line`, fields space-separated, optional
symbol appended last. No comments, no blank lines, trailing newline.
"""

from __future__ import annotations

from .nodes import ArraySort, BitvecSort, Line, Model, Node


def _sort_tokens(line: BitvecSort | ArraySort) -> list[str]:
    if isinstance(line, BitvecSort):
        return [str(line.nid), "sort", "bitvec", str(line.width)]
    return [str(line.nid), "sort", "array", str(line.index_sort), str(line.element_sort)]


def _node_tokens(node: Node) -> list[str]:
    tokens: list[str] = [str(node.nid), node.op]
    if node.sort is not None:
        tokens.append(str(node.sort))
    tokens.extend(str(arg) for arg in node.args)
    return tokens


def line_to_text(line: Line) -> str:
    if isinstance(line, (BitvecSort, ArraySort)):
        tokens = _sort_tokens(line)
    else:
        tokens = _node_tokens(line)
    if line.symbol is not None:
        tokens.append(line.symbol)
    return " ".join(tokens)


def to_text(model: Model) -> str:
    """Render ``model`` to canonical BTOR2 text (with trailing newline)."""

    if not model.lines:
        return ""
    return "\n".join(line_to_text(line) for line in model.lines) + "\n"


__all__ = ["line_to_text", "to_text"]
