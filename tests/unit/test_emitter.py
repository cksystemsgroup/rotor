"""BTOR2 emitter seam contract and IdentityEmitter equivalence."""

from pathlib import Path

import pytest

from rotor.binary import RISCVBinary
from rotor.btor2.builder import build_reach
from rotor.btor2.printer import to_text
from rotor.ir import BTOR2Emitter, IdentityEmitter, ReachSpec, emit_btor2_bytes

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "add2.elf"


def test_identity_emitter_satisfies_protocol() -> None:
    with RISCVBinary(FIXTURE) as b:
        emitter = IdentityEmitter(b)
        assert isinstance(emitter, BTOR2Emitter)
        assert emitter.name == "identity"


def test_identity_matches_build_reach_byte_for_byte() -> None:
    """The seam must not alter the BTOR2 produced by L0's core code path."""
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("add2")
        spec = ReachSpec(function="add2", target_pc=fn.start + 4)
        direct_text = to_text(build_reach(b, spec))
        seam_text = to_text(IdentityEmitter(b).emit(spec))
    assert seam_text == direct_text


def test_emit_btor2_bytes_roundtrips_to_same_text() -> None:
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("sign")
        spec = ReachSpec(function="sign", target_pc=fn.start + 0x14)
        emitter = IdentityEmitter(b)
        blob = emit_btor2_bytes(emitter, spec)
    assert isinstance(blob, bytes)
    text = blob.decode("utf-8")
    assert "sort bitvec 64" in text
    assert text.rstrip().splitlines()[-1].startswith(text.rstrip().splitlines()[-1].split()[0])
    # Structure is sane: ends with a `bad <id>` line.
    assert text.rstrip().split()[-2] == "bad"


def test_identity_rejects_unsupported_spec() -> None:
    # Synthesize a QuestionSpec subclass the emitter doesn't know about.
    from rotor.ir.spec import QuestionSpec
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FakeSpec(QuestionSpec):
        x: int = 0

    with RISCVBinary(FIXTURE) as b:
        with pytest.raises(TypeError, match="FakeSpec"):
            IdentityEmitter(b).emit(FakeSpec())
