"""Python Rotor — orchestration layer and reasoning engine over C Rotor.

This package exposes two entry points:

    rotor.RotorAPI   — high-level question API (Phase 7)
    rotor.RotorEngine — lower-level engine for multi-instance workflows (Phase 4)

Plus the building blocks:

    rotor.RISCVBinary  — ELF + DWARF loader (Phase 1)
    rotor.RotorInstance — a single BTOR2 model instance (Phase 3)
    rotor.ModelConfig   — configuration for a model instance
    rotor.btor2         — BTOR2 node DAG, builder, printer (Phase 2)
    rotor.solvers       — solver backends (Phase 3, 6)
    rotor.trace         — source-level traces (Phase 5)
"""

from __future__ import annotations

from rotor.binary import (
    RISCVBinary,
    Segment,
    Symbol,
    SourceLocation,
    FunctionInfo,
    VariableInfo,
    DWARFLocation,
)
from rotor.instance import RotorInstance, ModelConfig
from rotor.engine import RotorEngine
from rotor.api import RotorAPI
from rotor.solvers.base import CheckResult, SolverBackend
from rotor.trace import MachineState, SourceStep, SourceTrace

__all__ = [
    "RISCVBinary",
    "Segment",
    "Symbol",
    "SourceLocation",
    "FunctionInfo",
    "VariableInfo",
    "DWARFLocation",
    "RotorInstance",
    "ModelConfig",
    "RotorEngine",
    "RotorAPI",
    "CheckResult",
    "SolverBackend",
    "MachineState",
    "SourceStep",
    "SourceTrace",
]

__version__ = "0.1.0"
