"""The BTOR2 emitter seam.

    BTOR2Emitter  — Protocol that every IR implements.
    IdentityEmitter — L0: delegates to the in-process BTOR2 builder.

Rotor's whole layered architecture hinges on this seam: every layer
(identity, DAG, SSA-BV, BVDD) produces the same BTOR2 proof
obligations at the external boundary, so solvers and IR layers are
independently pluggable. Everything above this file assumes only the
BTOR2Emitter protocol; everything below emits BTOR2.

The Protocol's primary method returns an in-memory Model. A free
helper `emit_btor2_bytes` serializes that Model to the text BTOR2
format that external tools (Bitwuzla, BtorMC, rIC3, AVR, ABC) consume.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rotor.binary import RISCVBinary
from rotor.btor2.builder import build_find_input, build_reach, build_verify
from rotor.btor2.nodes import Model
from rotor.btor2.printer import to_text
from rotor.ir.dag import DagBuilder
from rotor.ir.spec import FindInputSpec, QuestionSpec, ReachSpec, VerifySpec


@runtime_checkable
class BTOR2Emitter(Protocol):
    """Compile a QuestionSpec into a BTOR2 Model.

    Every IR layer implements this. Consumers (RotorEngine, solvers,
    text emitters) interact with IRs exclusively through this protocol.
    """
    name: str

    def emit(self, spec: QuestionSpec) -> Model: ...


def emit_btor2_bytes(emitter: BTOR2Emitter, spec: QuestionSpec) -> bytes:
    """Serialize the emitter's Model to text BTOR2 bytes.

    This is the external artifact rotor presents to solvers that expect
    BTOR2 on stdin or in a file. In-process rotor backends (Z3BMC today)
    consume the Model directly, bypassing serialization.
    """
    return to_text(emitter.emit(spec)).encode("utf-8")


class IdentityEmitter:
    """L0 emitter: in-process BTOR2 generation without IR transformation.

    Delegates every supported spec to rotor/btor2/builder.py. Conceptually
    this is the seat of C Rotor — had rotor chosen to bridge to the C
    implementation, the bridge would replace this class without anyone
    above the seam noticing.

    IdentityEmitter is the reference against which every other IR is
    validated by the L0-equivalence harness (see tests/equivalence/).
    """
    name = "identity"

    def __init__(self, binary: RISCVBinary) -> None:
        self._binary = binary

    @property
    def binary(self) -> RISCVBinary:
        return self._binary

    def emit(self, spec: QuestionSpec) -> Model:
        if isinstance(spec, ReachSpec):
            return build_reach(self._binary, spec)
        if isinstance(spec, VerifySpec):
            return build_verify(self._binary, spec)
        if isinstance(spec, FindInputSpec):
            return build_find_input(self._binary, spec)
        raise TypeError(
            f"{type(self).__name__} does not support spec type "
            f"{type(spec).__name__}"
        )


class DagEmitter:
    """L1 emitter: hash-consed BV expression DAG with local simplification.

    Reuses the L0 builder by injecting a DagBuilder in place of the plain
    Model. Every node goes through hash-consing and a small set of
    auditable rewrites (constant folding, identity laws, ITE collapse,
    extract-of-constant). The output remains a Model — printable,
    solver-consumable, and equivalent to L0 on the full corpus.
    """
    name = "dag"

    def __init__(self, binary: RISCVBinary) -> None:
        self._binary = binary

    @property
    def binary(self) -> RISCVBinary:
        return self._binary

    def emit(self, spec: QuestionSpec) -> Model:
        if isinstance(spec, ReachSpec):
            return build_reach(self._binary, spec, builder=DagBuilder())
        if isinstance(spec, VerifySpec):
            return build_verify(self._binary, spec, builder=DagBuilder())
        if isinstance(spec, FindInputSpec):
            return build_find_input(self._binary, spec, builder=DagBuilder())
        raise TypeError(
            f"{type(self).__name__} does not support spec type "
            f"{type(spec).__name__}"
        )
