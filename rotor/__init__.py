"""Rotor: a BTOR2 compiler for RISC-V questions.

L0 entry points:

    from rotor import RISCVBinary, RotorAPI
"""

from rotor.api import ReachResult, RotorAPI
from rotor.binary import Function, RISCVBinary
from rotor.engine import EngineConfig, RotorEngine
from rotor.ir import BTOR2Emitter, IdentityEmitter, QuestionSpec, ReachSpec
from rotor.solvers import Portfolio, PortfolioEntry, Z3BMC
from rotor.trace import Trace

__all__ = [
    "RISCVBinary",
    "Function",
    "RotorAPI",
    "ReachResult",
    "RotorEngine",
    "EngineConfig",
    "BTOR2Emitter",
    "IdentityEmitter",
    "QuestionSpec",
    "ReachSpec",
    "Portfolio",
    "PortfolioEntry",
    "Z3BMC",
    "Trace",
]
