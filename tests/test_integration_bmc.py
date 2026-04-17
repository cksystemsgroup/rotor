"""End-to-end BMC on real RISC-V ELF fixtures.

Skipped when Bitwuzla or the fixtures are unavailable. These tests exercise
the full pipeline: ELF → native BTOR2 → BMC unroller → witness decomposition
→ DWARF-annotated SourceTrace with disassembled instructions.
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


# ──────────────────────────────────────────────────────────────────────────
# UNSAT cases — function bodies execute without tripping illegal-instruction
# ──────────────────────────────────────────────────────────────────────────


def test_add2_executes_cleanly_up_to_ret() -> None:
    """The 13 instructions of add2() should not trip illegal-instruction
    bounded below the ret, regardless of initial register state."""
    from rotor import ModelConfig, RISCVBinary, RotorInstance

    with RISCVBinary(_fixture("add2.elf")) as binary:
        low, high = binary.function_bounds("add2")
        cfg = ModelConfig(
            is_64bit=True, code_start=low, code_end=high,
            solver="bitwuzla", bound=12, model_backend="python",
        )
        inst = RotorInstance(binary, cfg)
        result = inst.check()
        assert result.verdict == "unsat", f"{result.verdict}: {result.stderr}"


def test_sum_executes_cleanly_for_nine_steps() -> None:
    """The function prologue of sum_to() is covered by our ISA subset."""
    from rotor import ModelConfig, RISCVBinary, RotorInstance

    with RISCVBinary(_fixture("sum.elf")) as binary:
        low, high = binary.function_bounds("sum_to")
        cfg = ModelConfig(
            is_64bit=True, code_start=low, code_end=high,
            solver="bitwuzla", bound=9, model_backend="python",
        )
        inst = RotorInstance(binary, cfg)
        result = inst.check()
        assert result.verdict == "unsat"


def test_counter_eight_steps_unsat() -> None:
    """Original fixture still passes after the ISA additions."""
    from rotor import ModelConfig, RISCVBinary, RotorInstance

    with RISCVBinary(_fixture("counter.elf")) as binary:
        low, high = binary.function_bounds("counter")
        cfg = ModelConfig(
            is_64bit=True, code_start=low, code_end=high,
            solver="bitwuzla", bound=8, model_backend="python",
        )
        inst = RotorInstance(binary, cfg)
        result = inst.check()
        assert result.verdict == "unsat"


# ──────────────────────────────────────────────────────────────────────────
# SAT case — ret with uninitialized ra drives PC to 0 → illegal-instruction
# ──────────────────────────────────────────────────────────────────────────


def test_add2_ret_with_zero_ra_trips_illegal() -> None:
    """In a bare machine, `ret` reads ra=0 and jumps to PC=0 which has no
    valid instruction. BMC should find this illegal-instruction reachable."""
    from rotor import ModelConfig, RISCVBinary, RotorInstance

    with RISCVBinary(_fixture("add2.elf")) as binary:
        low, high = binary.function_bounds("add2")
        cfg = ModelConfig(
            is_64bit=True, code_start=low, code_end=high,
            solver="bitwuzla", bound=20, model_backend="python",
        )
        inst = RotorInstance(binary, cfg)
        result = inst.check()
        assert result.verdict == "sat"
        # The illegal fires at step 13 (just after ret).
        assert result.steps is not None and 12 <= result.steps <= 15


# ──────────────────────────────────────────────────────────────────────────
# Source trace rendering from a real BMC witness
# ──────────────────────────────────────────────────────────────────────────


def test_add2_witness_renders_as_source_trace() -> None:
    from rotor import ModelConfig, RISCVBinary, RotorInstance, SourceTrace

    with RISCVBinary(_fixture("add2.elf")) as binary:
        low, high = binary.function_bounds("add2")
        cfg = ModelConfig(
            is_64bit=True, code_start=low, code_end=high,
            solver="bitwuzla", bound=15, model_backend="python",
        )
        inst = RotorInstance(binary, cfg)
        result = inst.check()
        assert result.verdict == "sat"
        trace = SourceTrace(inst.get_witness(), binary)
        assert len(trace) > 10
        # The first 12 steps should all be inside the add2 function.
        for step in trace.steps[:12]:
            assert step.function_name == "add2"
            assert step.location is not None and "add2.c" in step.location.file
        # Disassembly should produce RISC-V mnemonics, not raw bytes.
        md = trace.as_markdown()
        for mnemonic in ("addi", "sd", "lw", "addw", "ret"):
            assert mnemonic in md


# ──────────────────────────────────────────────────────────────────────────
# Symbolic-input exercise: prove the add2 output under given inputs
# ──────────────────────────────────────────────────────────────────────────


def test_add2_with_symbolic_inputs_computes_sum() -> None:
    """With symbolic a0 and a1, find (a0, a1) such that at step 9 the result
    in a0 equals 7. This exercises the full BMC + ExpressionCompiler +
    array-witness-decomposition path."""
    from rotor import ModelConfig, RISCVBinary, RotorInstance
    from rotor.expr import ExpressionCompiler

    with RISCVBinary(_fixture("add2.elf")) as binary:
        low, high = binary.function_bounds("add2")
        cfg = ModelConfig(
            is_64bit=True, code_start=low, code_end=high,
            solver="bitwuzla", bound=9, model_backend="python",
            init_registers_to_zero=False,
        )
        inst = RotorInstance(binary, cfg)
        inst.build_machine()
        # Reaching pc=0x1117c (the ld after addw) means addw has executed.
        addw_done_pc = 0x1117c
        condition = f"pc == 0x{addw_done_pc:x} && a0 == 7"
        bad = ExpressionCompiler.compile(inst, condition)
        inst.add_bad(bad, condition)
        result = inst.check()
        assert result.verdict == "sat"
        assert result.steps == 9
        frame = next(f for f in result.witness if f["step"] == 9)
        # At step 9, a0 should indeed be 7.
        assert frame["assignments"]["register-file[10]"] == 7
        # Verify that the symbolic inputs satisfy sum = 7 (truncating to 32
        # bits and sign-extending, since addw is word-wise).
        a0_init = result.witness[0]["assignments"]["register-file[10]"]
        a1_init = result.witness[0]["assignments"]["register-file[11]"]
        truncated = (a0_init + a1_init) & 0xFFFFFFFF
        # addw sign-extends 32-bit result to 64; map back to int32.
        if truncated & 0x80000000:
            truncated -= 0x100000000
        assert truncated == 7 or truncated == 7 - (1 << 32)
