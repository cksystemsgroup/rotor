"""Track B.3 integration: RVC fixture reaches end-to-end.

`rvc.elf` is built with `-march=rv64imc`, so clang freely mixes
16-bit compressed and 32-bit uncompressed instructions inside each
function. This test exercises the variable-length scanner, the
RVC expander, and the size-threading lowering pipeline.
"""

from __future__ import annotations

from pathlib import Path

from rotor.binary import RISCVBinary
from rotor.ir.emitter import IdentityEmitter
from rotor.ir.spec import ReachSpec
from rotor.solvers import BitwuzlaBMC, Z3BMC

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rvc.elf"


def test_add_rvc_is_entirely_compressed() -> None:
    """add_rvc fits into two 16-bit instructions (c.addw + c.ret)
    totalling 4 bytes. The scanner must see `size=2` on both."""
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("add_rvc")
        insts = list(b.instructions(fn))
        assert len(insts) == 2
        assert all(i.size == 2 for i in insts)


def test_mixed_rvc_rv64i_sequence_decodes() -> None:
    # triple = slli (4 bytes) + c.add + c.addiw + c.ret (2 bytes each).
    # The scanner must alternate size=4 and size=2 correctly.
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("triple")
        insts = list(b.instructions(fn))
        sizes = [i.size for i in insts]
    assert sizes == [4, 2, 2, 2]


def test_add_rvc_ret_reachable_at_step_1() -> None:
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("add_rvc")
        model = IdentityEmitter(b).emit(ReachSpec(function="add_rvc", target_pc=fn.start + 2))
        z3 = Z3BMC().check_reach(model, bound=2, timeout=10.0)
        bw = BitwuzlaBMC().check_reach(model, bound=2, timeout=10.0)
    assert z3.verdict == bw.verdict == "reachable"
    assert z3.step == bw.step == 1


def test_triple_ret_reachable_at_step_3() -> None:
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("triple")
        model = IdentityEmitter(b).emit(ReachSpec(function="triple", target_pc=fn.start + 8))
        r = Z3BMC().check_reach(model, bound=4, timeout=10.0)
    assert r.verdict == "reachable"
    assert r.step == 3


def test_signbit_ret_reachable() -> None:
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("signbit")
        model = IdentityEmitter(b).emit(ReachSpec(function="signbit", target_pc=fn.start + 4))
        r = Z3BMC().check_reach(model, bound=3, timeout=10.0)
    assert r.verdict == "reachable"
    assert r.step == 1
