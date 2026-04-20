"""Parse BTOR2 text into a Model.

The inverse of `rotor.btor2.printer.to_text`, for the subset rotor emits.
This exists to support round-trip debugging (parse -> re-emit) and to let
rotor's portfolio/engine run against external BTOR2 benchmarks. Rotor's
compile pipeline is unaffected: binary + question remain the only input.

Parsing collects diagnostics rather than aborting on the first issue. A
line that cannot be interpreted is skipped and recorded; the caller
inspects `ParseResult.diagnostics` to decide whether the resulting Model
is usable.

Phase 1 supports exactly the subset produced by printer.to_text:

    sort bitvec <w>
    sort array <idx> <elt>
    constd <sort> <value>
    input <sort> <name>
    state <sort> <name>
    <opname> <sort> <args...>          opname in SUPPORTED_OPS
    ite <sort> <c> <t> <e>
    slice <sort> <a> <hi> <lo>
    uext|sext <sort> <a> <n>
    read <sort> <array> <addr>
    write <sort> <array> <addr> <val>
    init <sort> <state> <expr>
    next <sort> <state> <expr>
    bad <expr>

HWMCC constructs (zero/one/ones/const/consth, output/constraint/justice/
fair, trailing symbol tokens on any line) are deferred to a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from rotor.btor2.nodes import AnySort, ArraySort, Model, Node, Sort


SUPPORTED_OPS: frozenset[str] = frozenset({
    "add", "sub", "and", "or", "xor",
    "eq", "neq", "ult", "slt",
    "sll", "srl", "sra",
    "concat",
})


@dataclass(frozen=True)
class Diagnostic:
    line_no: int
    severity: str        # "error" | "warning"
    message: str


@dataclass
class ParseResult:
    model: Model
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)


def from_text(src: str) -> ParseResult:
    p = _Parser(src)
    p.run()
    return ParseResult(model=p.model, diagnostics=p.diagnostics)


def from_path(path: Union[str, Path]) -> ParseResult:
    return from_text(Path(path).read_text())


class _SkipLine(Exception):
    """Abort this line and record the reason as an error diagnostic."""


class _Parser:
    def __init__(self, src: str) -> None:
        self.src = src
        self.model = Model()
        self.diagnostics: list[Diagnostic] = []
        self.sorts: dict[int, AnySort] = {}
        self.nodes: dict[int, Node] = {}

    def run(self) -> None:
        for line_no, raw in enumerate(self.src.splitlines(), start=1):
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            try:
                self._dispatch(line_no, tokens)
            except _SkipLine as e:
                self._err(line_no, str(e))
            except AssertionError as e:
                self._err(line_no, f"model consistency error: {str(e) or 'assertion failed'}")

    def _dispatch(self, line_no: int, tokens: list[str]) -> None:
        if len(tokens) < 2:
            raise _SkipLine(f"expected '<id> <tag> ...', got {' '.join(tokens)!r}")
        ext_id = self._int(tokens[0], "id")
        tag = tokens[1]
        args = tokens[2:]

        if tag == "sort":
            self._parse_sort(ext_id, args)
        elif tag == "constd":
            self._parse_constd(ext_id, args)
        elif tag == "input":
            self._parse_input(ext_id, args)
        elif tag == "state":
            self._parse_state(ext_id, args)
        elif tag == "ite":
            self._parse_ite(ext_id, args)
        elif tag == "slice":
            self._parse_slice(ext_id, args)
        elif tag in ("uext", "sext"):
            self._parse_ext(ext_id, tag, args)
        elif tag == "read":
            self._parse_read(ext_id, args)
        elif tag == "write":
            self._parse_write(ext_id, args)
        elif tag == "init":
            self._parse_init_or_next(ext_id, "init", args)
        elif tag == "next":
            self._parse_init_or_next(ext_id, "next", args)
        elif tag == "bad":
            self._parse_bad(ext_id, args)
        elif tag in SUPPORTED_OPS:
            self._parse_op(ext_id, tag, args)
        else:
            raise _SkipLine(f"unsupported tag {tag!r}")

    # -- tag handlers --------------------------------------------------

    def _parse_sort(self, ext_id: int, args: list[str]) -> None:
        if not args:
            raise _SkipLine("sort requires a kind (bitvec|array)")
        kind = args[0]
        if kind == "bitvec":
            if len(args) != 2:
                raise _SkipLine("sort bitvec requires exactly <width>")
            width = self._int(args[1], "width")
            if width <= 0:
                raise _SkipLine(f"bitvec width must be positive, got {width}")
            self.sorts[ext_id] = Sort(width)
            self.model.sort_id(width)
        elif kind == "array":
            if len(args) != 3:
                raise _SkipLine("sort array requires <index_sort> <element_sort>")
            idx = self._sort_ref(args[1])
            elt = self._sort_ref(args[2])
            if not isinstance(idx, Sort) or not isinstance(elt, Sort):
                raise _SkipLine("array sort components must be bitvec sorts")
            self.sorts[ext_id] = ArraySort(index=idx, element=elt)
            self.model.array_sort_id(idx, elt)
        else:
            raise _SkipLine(f"unknown sort kind {kind!r}")

    def _parse_constd(self, ext_id: int, args: list[str]) -> None:
        if len(args) != 2:
            raise _SkipLine("constd requires <sort> <value>")
        sort = self._bv_sort_ref(args[0])
        value = self._int(args[1], "value")
        self.nodes[ext_id] = self.model.const(sort, value)

    def _parse_input(self, ext_id: int, args: list[str]) -> None:
        if not args:
            raise _SkipLine("input requires <sort> [<name>]")
        sort = self._bv_sort_ref(args[0])
        name = args[1] if len(args) >= 2 else f"input{ext_id}"
        self.nodes[ext_id] = self.model.input(sort, name)

    def _parse_state(self, ext_id: int, args: list[str]) -> None:
        if not args:
            raise _SkipLine("state requires <sort> [<name>]")
        sort = self._sort_ref(args[0])
        name = args[1] if len(args) >= 2 else f"state{ext_id}"
        self.nodes[ext_id] = self.model.state(sort, name)

    def _parse_op(self, ext_id: int, opname: str, args: list[str]) -> None:
        if len(args) < 3:
            raise _SkipLine(f"op {opname} requires <sort> <arg> <arg>")
        sort = self._bv_sort_ref(args[0])
        operands = tuple(self._node_ref(a) for a in args[1:])
        self.nodes[ext_id] = self.model.op(opname, sort, *operands)

    def _parse_ite(self, ext_id: int, args: list[str]) -> None:
        if len(args) != 4:
            raise _SkipLine("ite requires <sort> <cond> <then> <else>")
        sort = self._sort_ref(args[0])
        cond = self._node_ref(args[1])
        t = self._node_ref(args[2])
        e = self._node_ref(args[3])
        if t.sort != sort or e.sort != sort:
            raise _SkipLine("ite branch sorts do not match declared sort")
        self.nodes[ext_id] = self.model.ite(cond, t, e)

    def _parse_slice(self, ext_id: int, args: list[str]) -> None:
        if len(args) != 4:
            raise _SkipLine("slice requires <sort> <arg> <hi> <lo>")
        self._bv_sort_ref(args[0])
        a = self._node_ref(args[1])
        hi = self._int(args[2], "hi")
        lo = self._int(args[3], "lo")
        self.nodes[ext_id] = self.model.slice(a, hi, lo)

    def _parse_ext(self, ext_id: int, which: str, args: list[str]) -> None:
        if len(args) != 3:
            raise _SkipLine(f"{which} requires <sort> <arg> <extra>")
        self._bv_sort_ref(args[0])
        a = self._node_ref(args[1])
        extra = self._int(args[2], "extra")
        if which == "uext":
            self.nodes[ext_id] = self.model.uext(a, extra)
        else:
            self.nodes[ext_id] = self.model.sext(a, extra)

    def _parse_read(self, ext_id: int, args: list[str]) -> None:
        if len(args) != 3:
            raise _SkipLine("read requires <elt_sort> <array> <addr>")
        self._bv_sort_ref(args[0])
        array = self._node_ref(args[1])
        addr = self._node_ref(args[2])
        self.nodes[ext_id] = self.model.read(array, addr)

    def _parse_write(self, ext_id: int, args: list[str]) -> None:
        if len(args) != 4:
            raise _SkipLine("write requires <array_sort> <array> <addr> <val>")
        self._sort_ref(args[0])
        array = self._node_ref(args[1])
        addr = self._node_ref(args[2])
        val = self._node_ref(args[3])
        self.nodes[ext_id] = self.model.write(array, addr, val)

    def _parse_init_or_next(self, ext_id: int, kind: str, args: list[str]) -> None:
        if len(args) != 3:
            raise _SkipLine(f"{kind} requires <sort> <state> <expr>")
        self._sort_ref(args[0])
        state = self._node_ref(args[1])
        expr = self._node_ref(args[2])
        if kind == "init":
            self.nodes[ext_id] = self.model.init(state, expr)
        else:
            self.nodes[ext_id] = self.model.next(state, expr)

    def _parse_bad(self, ext_id: int, args: list[str]) -> None:
        if len(args) != 1:
            raise _SkipLine("bad requires <expr>")
        expr = self._node_ref(args[0])
        self.nodes[ext_id] = self.model.bad(expr)

    # -- helpers -------------------------------------------------------

    def _int(self, tok: str, what: str) -> int:
        try:
            return int(tok)
        except ValueError:
            raise _SkipLine(f"expected integer {what}, got {tok!r}")

    def _sort_ref(self, tok: str) -> AnySort:
        sid = self._int(tok, "sort id")
        sort = self.sorts.get(sid)
        if sort is None:
            raise _SkipLine(f"unknown sort id {sid}")
        return sort

    def _bv_sort_ref(self, tok: str) -> Sort:
        sort = self._sort_ref(tok)
        if not isinstance(sort, Sort):
            raise _SkipLine(f"expected bitvec sort, got array sort id {tok}")
        return sort

    def _node_ref(self, tok: str) -> Node:
        nid = self._int(tok, "node id")
        node = self.nodes.get(nid)
        if node is None:
            raise _SkipLine(f"unknown node id {nid}")
        return node

    def _err(self, line_no: int, message: str) -> None:
        self.diagnostics.append(Diagnostic(line_no=line_no, severity="error", message=message))
