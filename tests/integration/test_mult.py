"""Track B.1 integration: RV64M fixture reaches end-to-end through
the full solver stack.

`mult.elf` is built with `-march=rv64im` so the compiler emits
`mul` / `mulw` / `divuw` instead of library calls. This test
exercises the decoder, ISA lowering, witness simulator, and
multiple solver backends against the same fixture.
"""

from __future__ import annotations

from pathlib import Path

from rotor.binary import RISCVBinary
from rotor.ir.emitter import IdentityEmitter
from rotor.ir.spec import ReachSpec
from rotor.solvers import BitwuzlaBMC, Z3BMC

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mult.elf"


def test_mul_add_ret_reachable_in_two_steps() -> None:
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("mul_add")
        model = IdentityEmitter(b).emit(ReachSpec(function="mul_add", target_pc=fn.start + 8))
        z3 = Z3BMC().check_reach(model, bound=4, timeout=10.0)
        bw = BitwuzlaBMC().check_reach(model, bound=4, timeout=10.0)
    assert z3.verdict == "reachable" and z3.step == 2
    assert bw.verdict == "reachable" and bw.step == 2


def test_divmod_ret_reachable() -> None:
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("divmod")
        model = IdentityEmitter(b).emit(ReachSpec(function="divmod", target_pc=fn.start + 16))
        r = Z3BMC().check_reach(model, bound=8, timeout=10.0)
    assert r.verdict == "reachable"
    assert r.step == 4


def test_mul64_ret_reachable() -> None:
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("mul64")
        model = IdentityEmitter(b).emit(ReachSpec(function="mul64", target_pc=fn.start + 4))
        r = Z3BMC().check_reach(model, bound=2, timeout=10.0)
    assert r.verdict == "reachable"
    assert r.step == 1


def test_bitwuzla_agrees_with_z3_on_rv64m() -> None:
    # Concrete value check: bitwuzla and z3 must agree on the witness
    # produced by a reach-the-ret query on a function that uses mul
    # and divuw in sequence.
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("divmod")
        model = IdentityEmitter(b).emit(
            ReachSpec(function="divmod", target_pc=fn.start + 16)
        )
        z3 = Z3BMC().check_reach(model, bound=6, timeout=10.0)
        bw = BitwuzlaBMC().check_reach(model, bound=6, timeout=10.0)
    assert z3.verdict == bw.verdict == "reachable"
    assert z3.step == bw.step
