"""Phase 2: BTOR2 node DAG, builder, printer, and C Rotor backend.

Public surface:

    Sort, Node, NodeDAG        — node representation
    BTOR2Builder               — constructor API for all BTOR2 operations
    RISCVMachineBuilder        — (skeleton) builds a RISC-V machine model
    MachineModel               — aggregate returned by a builder
    BTOR2Printer               — emits a NodeDAG as BTOR2 text
    parse_btor2                — minimal BTOR2 parser (for C Rotor output)
    CRotorBackend              — subprocess wrapper around ``rotor`` binary
"""

from __future__ import annotations

from rotor.btor2.nodes import Sort, Node, NodeDAG, MachineModel
from rotor.btor2.builder import BTOR2Builder, RISCVMachineBuilder, CRotorBackend
from rotor.btor2.printer import BTOR2Printer
from rotor.btor2.parser import parse_btor2

__all__ = [
    "Sort",
    "Node",
    "NodeDAG",
    "MachineModel",
    "BTOR2Builder",
    "RISCVMachineBuilder",
    "CRotorBackend",
    "BTOR2Printer",
    "parse_btor2",
]
