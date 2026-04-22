"""Internal representation layer.

Rotor's architecture uses the BTOR2 emitter Protocol as the seam
between question orchestration (engine, api, cli) and the actual
production of a BTOR2 proof obligation. Every IR (identity, DAG,
SSA-BV, BVDD) implements BTOR2Emitter; the layer above the seam never
cares which one is used beyond what capability protocols it supports.

M4 ships only the seam plus IdentityEmitter (L0). Future layers are
added by implementing BTOR2Emitter against the same QuestionSpec.
"""

from rotor.ir.dag import DagBuilder
from rotor.ir.emitter import BTOR2Emitter, DagEmitter, IdentityEmitter, emit_btor2_bytes
from rotor.ir.spec import Comparison, QuestionSpec, ReachSpec, VerifySpec
from rotor.ir.ssa import SsaEmitter

__all__ = [
    "BTOR2Emitter",
    "IdentityEmitter",
    "DagEmitter",
    "DagBuilder",
    "SsaEmitter",
    "emit_btor2_bytes",
    "QuestionSpec",
    "ReachSpec",
    "VerifySpec",
    "Comparison",
]
