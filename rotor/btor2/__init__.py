"""In-memory BTOR2 representation plus textual printer.

The Model class is rotor's internal DAG. Every L0 question compiles to a
Model; higher IR layers lower into the same Model. The printer emits
text BTOR2 which is the external seam rotor presents to solvers.
"""

from rotor.btor2.nodes import Model, Node, Sort
from rotor.btor2.printer import to_text

__all__ = ["Model", "Node", "Sort", "to_text"]
