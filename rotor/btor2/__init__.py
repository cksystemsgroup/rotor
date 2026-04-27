"""Pure BTOR2 in-memory model, printer, and parser.

This package is the bottom of rotor's stack: it knows BTOR2 and nothing
else. It is the substrate the schema-deterministic compile pipeline emits
into and the flat text the linker hands to external solvers.
"""

from .nodes import (
    OP_SPECS,
    ArraySort,
    BitvecSort,
    Line,
    Model,
    Node,
    OpSpec,
    Sort,
)
from .parser import Diagnostic, ParseResult, from_path, from_text
from .printer import line_to_text, to_text

__all__ = [
    "ArraySort",
    "BitvecSort",
    "Diagnostic",
    "Line",
    "Model",
    "Node",
    "OP_SPECS",
    "OpSpec",
    "ParseResult",
    "Sort",
    "from_path",
    "from_text",
    "line_to_text",
    "to_text",
]
