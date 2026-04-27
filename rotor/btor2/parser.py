"""BTOR2 text parser.

Reads canonical or HWMCC-superset BTOR2 text and returns a :class:`Model`
plus a list of :class:`Diagnostic` describing any non-fatal issues. The
parser does not raise on bad input; it skips offending lines, records
diagnostics, and continues. This lets the caller decide whether to treat
diagnostics as errors.

Accepted superset features:

* Comments: any line whose first non-whitespace character is ``;``.
* Blank lines.
* Trailing optional symbol (whitespace-free identifier) on every line.
* The ``zero`` / ``one`` / ``ones`` constant shorthands.
* ``const`` (binary), ``constd`` (decimal), ``consth`` (hexadecimal).

The parser does not validate that nid references are defined or that
sort/operand widths agree. Such checks are out of scope for the pure-BTOR2
layer; they are the schema layer's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .nodes import ArraySort, BitvecSort, Line, Model, Node, OP_SPECS


@dataclass(frozen=True)
class Diagnostic:
    line_no: int
    message: str


@dataclass
class ParseResult:
    model: Model
    diagnostics: list[Diagnostic] = field(default_factory=list)


def from_text(text: str) -> ParseResult:
    """Parse BTOR2 text into a :class:`ParseResult`."""

    model = Model()
    diagnostics: list[Diagnostic] = []

    for line_no, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith(";"):
            continue

        tokens = stripped.split()
        if len(tokens) < 2:
            diagnostics.append(Diagnostic(line_no, f"too few tokens: {stripped!r}"))
            continue

        try:
            nid = int(tokens[0])
        except ValueError:
            diagnostics.append(Diagnostic(line_no, f"non-integer nid: {tokens[0]!r}"))
            continue

        op = tokens[1]
        rest = tokens[2:]

        if op == "sort":
            line, diag = _parse_sort(line_no, nid, rest)
        else:
            line, diag = _parse_node(line_no, nid, op, rest)

        if diag is not None:
            diagnostics.append(diag)
        if line is not None:
            model.add(line)

    return ParseResult(model=model, diagnostics=diagnostics)


def from_path(path: str | Path) -> ParseResult:
    return from_text(Path(path).read_text())


def _parse_sort(
    line_no: int, nid: int, rest: list[str]
) -> tuple[Line | None, Diagnostic | None]:
    if not rest:
        return None, Diagnostic(line_no, "sort: missing kind")

    kind = rest[0]
    body = rest[1:]

    if kind == "bitvec":
        if len(body) < 1:
            return None, Diagnostic(line_no, "sort bitvec: missing width")
        try:
            width = int(body[0])
        except ValueError:
            return None, Diagnostic(line_no, f"sort bitvec: non-integer width {body[0]!r}")
        symbol, diag = _trailing_symbol(line_no, body[1:])
        if diag is not None:
            return None, diag
        return BitvecSort(nid=nid, width=width, symbol=symbol), None

    if kind == "array":
        if len(body) < 2:
            return None, Diagnostic(line_no, "sort array: missing index/element sort")
        try:
            index_sort = int(body[0])
            element_sort = int(body[1])
        except ValueError:
            return None, Diagnostic(line_no, f"sort array: non-integer sort refs {body[:2]!r}")
        symbol, diag = _trailing_symbol(line_no, body[2:])
        if diag is not None:
            return None, diag
        return (
            ArraySort(
                nid=nid,
                index_sort=index_sort,
                element_sort=element_sort,
                symbol=symbol,
            ),
            None,
        )

    return None, Diagnostic(line_no, f"sort: unknown kind {kind!r}")


def _parse_node(
    line_no: int, nid: int, op: str, rest: list[str]
) -> tuple[Line | None, Diagnostic | None]:
    spec = OP_SPECS.get(op)
    if spec is None:
        return None, Diagnostic(line_no, f"unknown op {op!r}")

    cursor = 0
    sort: int | None = None

    if spec.has_sort:
        if cursor >= len(rest):
            return None, Diagnostic(line_no, f"{op}: missing sort")
        try:
            sort = int(rest[cursor])
        except ValueError:
            return None, Diagnostic(line_no, f"{op}: non-integer sort {rest[cursor]!r}")
        cursor += 1

    args: list[int | str] = []

    if spec.variable:
        # justice: <num> <node1> ... <nodeN>
        if cursor >= len(rest):
            return None, Diagnostic(line_no, f"{op}: missing arity")
        try:
            count = int(rest[cursor])
        except ValueError:
            return None, Diagnostic(line_no, f"{op}: non-integer arity {rest[cursor]!r}")
        args.append(count)
        cursor += 1
        if cursor + count > len(rest):
            return None, Diagnostic(
                line_no, f"{op}: expected {count} operands, found {len(rest) - cursor}"
            )
        for tok in rest[cursor : cursor + count]:
            try:
                args.append(int(tok))
            except ValueError:
                return None, Diagnostic(line_no, f"{op}: non-integer operand {tok!r}")
        cursor += count
    else:
        needed = spec.extra
        if cursor + needed > len(rest):
            return None, Diagnostic(
                line_no, f"{op}: expected {needed} operand(s), found {len(rest) - cursor}"
            )
        for i in range(needed):
            tok = rest[cursor + i]
            args.append(_parse_arg(op, i, tok))
        cursor += needed

    symbol, diag = _trailing_symbol(line_no, rest[cursor:])
    if diag is not None:
        return None, diag

    return (
        Node(nid=nid, op=op, sort=sort, args=tuple(args), symbol=symbol),
        None,
    )


def _parse_arg(op: str, index: int, tok: str) -> int | str:
    """Decide whether an op's arg is a nid (int) or a literal (str).

    Constant payloads (``const``/``constd``/``consth``'s last arg) stay as
    strings so leading zeros and signs round-trip exactly. Everything else
    is a nid or an immediate integer (slice indices, ext widths, justice
    arity), all of which parse as integers.
    """

    if op in ("const", "constd", "consth") and index == 0:
        return tok
    try:
        return int(tok)
    except ValueError:
        # Fall back to string so we don't lose information; the diagnostic
        # path will surface this if it matters at a higher layer.
        return tok


def _trailing_symbol(
    line_no: int, leftover: list[str]
) -> tuple[str | None, Diagnostic | None]:
    if not leftover:
        return None, None
    if len(leftover) == 1:
        return leftover[0], None
    return None, Diagnostic(
        line_no, f"unexpected trailing tokens: {' '.join(leftover)!r}"
    )


__all__ = ["Diagnostic", "ParseResult", "from_path", "from_text"]
