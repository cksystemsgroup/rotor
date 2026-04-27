"""In-memory BTOR2 representation.

Pure BTOR2: this module knows nothing about RISC-V, the rotor schema, or any
higher-level concept. It models the BTOR2 text format (as described in
https://fmv.jku.at/papers/NiemetzPreinerWolfBiere-CAV18.pdf) plus the small
HWMCC superset accepted by btor2tools (``zero``/``one``/``ones`` shorthand,
``const``/``constd``/``consth``, optional trailing symbol on every line).

A :class:`Model` is an ordered list of :class:`Sort` and :class:`Node` lines
sharing one nid namespace. The printer in ``rotor.btor2.printer`` emits a
canonical text form; the parser in ``rotor.btor2.parser`` reads either
canonical or HWMCC-superset input and normalizes it back into a model whose
``to_text`` is byte-identical on re-emission.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BitvecSort:
    """``<nid> sort bitvec <width>``."""

    nid: int
    width: int
    symbol: str | None = None


@dataclass(frozen=True)
class ArraySort:
    """``<nid> sort array <index_sort_nid> <element_sort_nid>``."""

    nid: int
    index_sort: int
    element_sort: int
    symbol: str | None = None


Sort = BitvecSort | ArraySort


@dataclass(frozen=True)
class Node:
    """A non-sort BTOR2 line.

    ``op`` is the operator name as it appears in the text (``input``,
    ``state``, ``add``, ``ite``, ``bad``, ``constraint``, ``init``,
    ``next``, ``zero``, ``const``, ``slice``, ``sext``, ``justice``, …).

    ``sort`` is the nid of the sort line for ops that produce a value, or
    ``None`` for the sortless ops: ``init``, ``next``, ``bad``,
    ``constraint``, ``fair``, ``output``, ``justice``.

    ``args`` is the operator-specific tail. Each element is either an int
    (a nid reference, or an immediate width / slice index / justice arity)
    or a str (a constant's bit / decimal / hex representation). The order
    matches the operator's positional order in the BTOR2 text.

    ``symbol`` is the optional trailing identifier permitted on every line.
    """

    nid: int
    op: str
    sort: int | None = None
    args: tuple[int | str, ...] = ()
    symbol: str | None = None


Line = BitvecSort | ArraySort | Node


@dataclass
class Model:
    """An ordered collection of BTOR2 lines.

    Order is significant: BTOR2 requires that every nid reference point at a
    line that appears earlier in the file. The model preserves insertion
    order; the printer emits lines in that order.
    """

    lines: list[Line] = field(default_factory=list)

    def add(self, line: Line) -> Line:
        self.lines.append(line)
        return line

    def by_nid(self) -> dict[int, Line]:
        return {line.nid: line for line in self.lines}


# Operator metadata. Each entry says whether the op carries a sort nid as
# its first positional argument and how many positional args follow the
# sort. A negative ``extra`` means variable-length: the first extra arg is
# an integer count N and the remaining N args are nid references. The
# trailing symbol (always optional) is not counted.

@dataclass(frozen=True)
class OpSpec:
    has_sort: bool
    extra: int  # number of positional args after sort; -1 for variable

    @property
    def variable(self) -> bool:
        return self.extra < 0


_BINARY_OPS = (
    "iff", "implies",
    "eq", "neq",
    "sgt", "ugt", "sgte", "ugte", "slt", "ult", "slte", "ulte",
    "and", "nand", "nor", "or", "xnor", "xor",
    "rol", "ror", "sll", "sra", "srl",
    "add", "mul", "sdiv", "udiv", "smod", "srem", "urem", "sub",
    "saddo", "uaddo", "sdivo", "smulo", "umulo", "ssubo",
    "concat", "read",
)

_UNARY_OPS = ("not", "inc", "dec", "neg", "redand", "redor", "redxor")

OP_SPECS: dict[str, OpSpec] = {
    "input": OpSpec(True, 0),
    "state": OpSpec(True, 0),
    "zero": OpSpec(True, 0),
    "one": OpSpec(True, 0),
    "ones": OpSpec(True, 0),
    "const": OpSpec(True, 1),
    "constd": OpSpec(True, 1),
    "consth": OpSpec(True, 1),
    "slice": OpSpec(True, 3),
    "sext": OpSpec(True, 2),
    "uext": OpSpec(True, 2),
    "ite": OpSpec(True, 3),
    "write": OpSpec(True, 3),
    "init": OpSpec(True, 2),
    "next": OpSpec(True, 2),
    "bad": OpSpec(False, 1),
    "constraint": OpSpec(False, 1),
    "fair": OpSpec(False, 1),
    "output": OpSpec(False, 1),
    "justice": OpSpec(False, -1),
}

for _op in _UNARY_OPS:
    OP_SPECS[_op] = OpSpec(True, 1)
for _op in _BINARY_OPS:
    OP_SPECS[_op] = OpSpec(True, 2)


__all__ = [
    "ArraySort",
    "BitvecSort",
    "Line",
    "Model",
    "Node",
    "OP_SPECS",
    "OpSpec",
    "Sort",
]
