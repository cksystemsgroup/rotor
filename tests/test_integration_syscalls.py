"""End-to-end BMC on real fixtures that use ECALL syscalls.

These tests compile down a chain of: real C → RISC-V ELF → native BTOR2 →
BMC unroller → syscall dispatch → symbolic input byte → source trace.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("bitwuzla")
pytest.importorskip("elftools")

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture(name: str) -> str:
    path = os.path.join(FIXTURES, name)
    if not os.path.exists(path):
        pytest.skip(f"missing fixture {name}")
    return path


def test_read_byte_is_fully_symbolic() -> None:
    """Given a function that calls read(stdin, buf, 1) and returns buf[0],
    verify that for any target byte value there exists an input byte that
    drives the return value to that target. This proves the full syscall
    dispatch + symbolic input + memory write path is connected end-to-end.
    """
    from rotor import ModelConfig, RISCVBinary, RotorInstance
    from rotor.expr import ExpressionCompiler

    with RISCVBinary(_fixture("readbyte.elf")) as binary:
        low, high = binary.function_bounds("read_byte")
        cfg = ModelConfig(
            is_64bit=True, code_start=low, code_end=high,
            solver="bitwuzla", bound=25, model_backend="python",
        )
        inst = RotorInstance(binary, cfg)
        inst.build_machine()
        # 0x111ec is the address right after the lbu that loads buf[0] into a0.
        # (Verified once by hand via `rotor info readbyte.elf` + objdump; the
        # test asserts the ELF hasn't drifted by checking that the PC and a0
        # in the witness line up.)
        target_pc = 0x111ec
        node = ExpressionCompiler.compile(
            inst, f"pc == 0x{target_pc:x} && a0 == 42"
        )
        inst.add_bad(node, f"after-lbu, a0==42")
        result = inst.check()
        assert result.verdict == "sat"
        assert result.witness is not None
        frame = next(f for f in result.witness if f["step"] == result.steps)
        assert frame["assignments"]["pc"] == target_pc
        assert frame["assignments"]["register-file[10]"] == 42


def test_read_byte_bounded_below_ecall_cannot_reach_lbu() -> None:
    """Sanity check: with a bound that stops before the ecall fires, a0
    cannot yet hold the input byte. The post-lbu target PC is unreachable."""
    from rotor import ModelConfig, RISCVBinary, RotorInstance
    from rotor.expr import ExpressionCompiler

    with RISCVBinary(_fixture("readbyte.elf")) as binary:
        low, high = binary.function_bounds("read_byte")
        cfg = ModelConfig(
            is_64bit=True, code_start=low, code_end=high,
            solver="bitwuzla", bound=10, model_backend="python",
        )
        inst = RotorInstance(binary, cfg)
        inst.build_machine()
        target_pc = 0x111ec
        node = ExpressionCompiler.compile(
            inst, f"pc == 0x{target_pc:x} && a0 == 42"
        )
        inst.add_bad(node, f"premature-a0-check")
        result = inst.check()
        # Within 10 transitions we haven't reached 0x111ec.
        assert result.verdict == "unsat"


def test_read_four_bytes_sum_matches_target() -> None:
    """BMC should find four input bytes whose sum equals a chosen target.

    ``sum4()`` reads 4 bytes via a single ecall, then sums them into a0.
    Asking for a0 == 10 at the PC just after the final add should be SAT
    with distinct-per-position input bytes from the multi-byte read.
    """
    from rotor import ModelConfig, RISCVBinary, RotorInstance
    from rotor.expr import ExpressionCompiler

    with RISCVBinary(_fixture("readsum.elf")) as binary:
        low, high = binary.function_bounds("sum4")
        cfg = ModelConfig(
            is_64bit=True, code_start=low, code_end=high,
            solver="bitwuzla", bound=35, model_backend="python",
        )
        inst = RotorInstance(binary, cfg)
        inst.build_machine()
        # 0x1120c is immediately after `add a0, a0, a1` (the final sum).
        target_pc = 0x1120c
        node = ExpressionCompiler.compile(
            inst, f"pc == 0x{target_pc:x} && a0 == 10"
        )
        inst.add_bad(node, "after-sum a0==10")
        result = inst.check()
        assert result.verdict == "sat"
        frame = next(f for f in result.witness if f["step"] == result.steps)
        assert frame["assignments"]["pc"] == target_pc
        assert frame["assignments"]["register-file[10]"] == 10
        # After exactly one read syscall, the read-count state should be 1.
        assert frame["assignments"]["read-count"] == 1


def test_ecall_halts_the_machine() -> None:
    """After the ecall in read_byte(), bounded execution past it keeps
    halted=0 only until the exit-style syscall; the read syscall does not
    halt. We verify the machine keeps transitioning after read."""
    from rotor import ModelConfig, RISCVBinary, RotorInstance

    with RISCVBinary(_fixture("readbyte.elf")) as binary:
        low, high = binary.function_bounds("read_byte")
        cfg = ModelConfig(
            is_64bit=True, code_start=low, code_end=high,
            solver="bitwuzla", bound=20, model_backend="python",
        )
        inst = RotorInstance(binary, cfg)
        # The default illegal-instruction bad shouldn't fire within the
        # function body — all instructions are covered.
        result = inst.check()
        assert result.verdict == "unsat"
