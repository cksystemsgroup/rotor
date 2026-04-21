"""Rotor: a BTOR2 compiler for RISC-V questions.

L0 entry points:

    from rotor import RISCVBinary, RotorAPI

Phase 6 adds unbounded reasoning and abstraction refinement:

    from rotor import Z3Spacer, cegar_reach, CegarConfig
"""

from rotor.api import ReachResult, RotorAPI
from rotor.binary import Function, RISCVBinary
from rotor.cegar import CegarConfig, cegar_reach
from rotor.engine import EngineConfig, RotorEngine
from rotor.ir import BTOR2Emitter, DagEmitter, IdentityEmitter, QuestionSpec, ReachSpec, SsaEmitter
from rotor.solvers import BitwuzlaBMC, Portfolio, PortfolioEntry, Z3BMC, Z3Spacer
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
    "DagEmitter",
    "SsaEmitter",
    "QuestionSpec",
    "ReachSpec",
    "Portfolio",
    "PortfolioEntry",
    "Z3BMC",
    "Z3Spacer",
    "BitwuzlaBMC",
    "CegarConfig",
    "cegar_reach",
    "Trace",
]
