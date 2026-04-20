"""Parse BTOR2 text into a Model.

The inverse of `rotor.btor2.printer.to_text`, for the subset rotor emits.
This exists to support round-trip debugging (parse -> re-emit) and to let
rotor's portfolio/engine run against external BTOR2 benchmarks. Rotor's
compile pipeline is unaffected: binary + question remain the only input.

Parsing collects diagnostics rather than aborting on the first issue. A
line that cannot be interpreted is skipped and recorded; the caller
inspects `ParseResult.diagnostics` to decide whether the resulting Model
is usable.

Phase 1 supports exactly the subset produced by printer.to_text. Phase 3
extends coverage to HWMCC benchmark files by accepting:

    zero|one|ones <sort>                 — normalized to constd
    const  <sort> <binary-string>        — normalized to constd
    consth <sort> <hex-string>           — normalized to constd
    constraint <expr>                    — invariant assumption (kept)
    output <sort> <expr>                 — warning, ignored
    justice <n> <expr>...                — warning, ignored
    fair    <n> <expr>...                — warning, ignored
    trailing symbol tokens on any line   — tolerated (silently dropped for
                                           kinds rotor's Model does not
                                           carry a name for)

The full HWMCC op set (unary, reductions, comparisons, arith, rotates,
overflow predicates) is accepted; arity is checked but operand sort is
not — operand-sort validation is the solver backend's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from rotor.btor2.nodes import AnySort, ArraySort, Model, Node, Sort


UNARY_OPS: frozenset[str] = frozenset({
    "not", "neg", "inc", "dec",
    "redand", "redor", "redxor",
})

BINARY_OPS: frozenset[str] = frozenset({
    # bitwise
    "and", "or", "xor", "nand", "nor", "xnor", "implies", "iff",
    # arithmetic
    "add", "sub", "mul",
    "udiv", "sdiv", "urem", "srem", "smod",
    # equality / comparison
    "eq", "neq",
    "ult", "ulte", "ugt", "ugte",
    "slt", "slte", "sgt", "sgte",
    # shifts / rotates
    "sll", "srl", "sra", "rol", "ror",
    # overflow predicates
    "saddo", "uaddo", "ssubo", "usubo", "smulo", "umulo", "sdivo",
    # concatenation
    "concat",
})

# Exposed for tests / introspection; kept as a union for convenience.
SUPPORTED_OPS: frozenset[str] = UNARY_OPS | BINARY_OPS


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
        elif tag in ("zero", "one", "ones"):
            self._parse_const_shorthand(ext_id, tag, args)
        elif tag == "const":
            self._parse_const_radix(ext_id, args, base=2, label="const")
        elif tag == "consth":
            self._parse_const_radix(ext_id, args, base=16, label="consth")
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
        elif tag == "constraint":
            self._parse_constraint(ext_id, args)
        elif tag == "output":
            self._warn(line_no, "output: accepted but not tracked")
        elif tag in ("justice", "fair"):
            self._warn(line_no, f"{tag}: liveness / fairness ignored")
        elif tag in UNARY_OPS:
            self._parse_op_n(ext_id, tag, args, arity=1)
        elif tag in BINARY_OPS:
            self._parse_op_n(ext_id, tag, args, arity=2)
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
        positional, _ = self._split(args, 2, tag="constd")
        sort = self._bv_sort_ref(positional[0])
        value = self._int(positional[1], "value")
        self.nodes[ext_id] = self.model.const(sort, value)

    def _parse_const_shorthand(self, ext_id: int, tag: str, args: list[str]) -> None:
        positional, _ = self._split(args, 1, tag=tag)
        sort = self._bv_sort_ref(positional[0])
        if tag == "zero":
            value = 0
        elif tag == "one":
            value = 1
        else:  # "ones"
            value = (1 << sort.width) - 1
        self.nodes[ext_id] = self.model.const(sort, value)

    def _parse_const_radix(
        self, ext_id: int, args: list[str], *, base: int, label: str
    ) -> None:
        positional, _ = self._split(args, 2, tag=label)
        sort = self._bv_sort_ref(positional[0])
        literal = positional[1]
        try:
            value = int(literal, base)
        except ValueError:
            raise _SkipLine(f"{label} literal {literal!r} is not base-{base}")
        if value < 0:
            raise _SkipLine(f"{label} literal {literal!r} is negative")
        self.nodes[ext_id] = self.model.const(sort, value)

    def _parse_input(self, ext_id: int, args: list[str]) -> None:
        if not args or len(args) > 2:
            raise _SkipLine("input requires <sort> [<name>]")
        sort = self._bv_sort_ref(args[0])
        name = args[1] if len(args) == 2 else f"input{ext_id}"
        self.nodes[ext_id] = self.model.input(sort, name)

    def _parse_state(self, ext_id: int, args: list[str]) -> None:
        if not args or len(args) > 2:
            raise _SkipLine("state requires <sort> [<name>]")
        sort = self._sort_ref(args[0])
        name = args[1] if len(args) == 2 else f"state{ext_id}"
        self.nodes[ext_id] = self.model.state(sort, name)

    def _parse_op_n(self, ext_id: int, opname: str, args: list[str], *, arity: int) -> None:
        positional, _ = self._split(args, 1 + arity, tag=opname)
        sort = self._bv_sort_ref(positional[0])
        operands = tuple(self._node_ref(a) for a in positional[1:])
        self.nodes[ext_id] = self.model.op(opname, sort, *operands)

    def _parse_ite(self, ext_id: int, args: list[str]) -> None:
        positional, _ = self._split(args, 4, tag="ite")
        sort = self._sort_ref(positional[0])
        cond = self._node_ref(positional[1])
        t = self._node_ref(positional[2])
        e = self._node_ref(positional[3])
        if t.sort != sort or e.sort != sort:
            raise _SkipLine("ite branch sorts do not match declared sort")
        self.nodes[ext_id] = self.model.ite(cond, t, e)

    def _parse_slice(self, ext_id: int, args: list[str]) -> None:
        positional, _ = self._split(args, 4, tag="slice")
        self._bv_sort_ref(positional[0])
        a = self._node_ref(positional[1])
        hi = self._int(positional[2], "hi")
        lo = self._int(positional[3], "lo")
        self.nodes[ext_id] = self.model.slice(a, hi, lo)

    def _parse_ext(self, ext_id: int, which: str, args: list[str]) -> None:
        positional, _ = self._split(args, 3, tag=which)
        self._bv_sort_ref(positional[0])
        a = self._node_ref(positional[1])
        extra = self._int(positional[2], "extra")
        if which == "uext":
            self.nodes[ext_id] = self.model.uext(a, extra)
        else:
            self.nodes[ext_id] = self.model.sext(a, extra)

    def _parse_read(self, ext_id: int, args: list[str]) -> None:
        positional, _ = self._split(args, 3, tag="read")
        self._bv_sort_ref(positional[0])
        array = self._node_ref(positional[1])
        addr = self._node_ref(positional[2])
        self.nodes[ext_id] = self.model.read(array, addr)

    def _parse_write(self, ext_id: int, args: list[str]) -> None:
        positional, _ = self._split(args, 4, tag="write")
        self._sort_ref(positional[0])
        array = self._node_ref(positional[1])
        addr = self._node_ref(positional[2])
        val = self._node_ref(positional[3])
        self.nodes[ext_id] = self.model.write(array, addr, val)

    def _parse_init_or_next(self, ext_id: int, kind: str, args: list[str]) -> None:
        positional, _ = self._split(args, 3, tag=kind)
        self._sort_ref(positional[0])
        state = self._node_ref(positional[1])
        expr = self._node_ref(positional[2])
        if kind == "init":
            self.nodes[ext_id] = self.model.init(state, expr)
        else:
            self.nodes[ext_id] = self.model.next(state, expr)

    def _parse_bad(self, ext_id: int, args: list[str]) -> None:
        positional, _ = self._split(args, 1, tag="bad")
        expr = self._node_ref(positional[0])
        self.nodes[ext_id] = self.model.bad(expr)

    def _parse_constraint(self, ext_id: int, args: list[str]) -> None:
        positional, _ = self._split(args, 1, tag="constraint")
        expr = self._node_ref(positional[0])
        self.nodes[ext_id] = self.model.constraint(expr)

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

    def _warn(self, line_no: int, message: str) -> None:
        self.diagnostics.append(Diagnostic(line_no=line_no, severity="warning", message=message))

    def _split(self, args: list[str], n: int, *, tag: str) -> tuple[list[str], str | None]:
        """Split `args` into `n` required positional tokens plus an optional
        trailing symbol. BTOR2 allows any node line to carry a symbol after
        its operands — rotor's Model has no storage for symbols on most
        kinds, so they are tolerated and silently dropped."""
        if len(args) < n:
            raise _SkipLine(f"{tag} requires {n} arg(s), got {len(args)}")
        if len(args) > n + 1:
            raise _SkipLine(f"{tag} takes {n} arg(s) plus optional symbol, got {len(args)}")
        symbol = args[n] if len(args) == n + 1 else None
        return args[:n], symbol
