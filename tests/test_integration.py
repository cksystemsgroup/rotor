"""End-to-end integration test against a real RISC-V ELF fixture.

This test compiles-in-advance a small C program to ``tests/fixtures/counter.elf``
and drives the full pipeline: binary loading, DWARF extraction, native BTOR2
model generation, printer/parser round-trip, code-segment initialization,
expression compilation, and witness-driven source-trace rendering.

The fixture is built during CI by the command:

    clang --target=riscv64-unknown-elf -march=rv64im -mabi=lp64 \\
          -ffreestanding -nostdlib -static -g -O0 counter.c \\
          -o tests/fixtures/counter.elf

Tests are skipped if the fixture is missing.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("elftools")

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "counter.elf"
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(FIXTURE),
    reason="tests/fixtures/counter.elf not built (needs clang --target=riscv64)",
)


# ──────────────────────────────────────────────────────────────────────────
# Phase 1: ELF + DWARF loading
# ──────────────────────────────────────────────────────────────────────────


def test_load_binary_headers() -> None:
    from rotor import RISCVBinary

    with RISCVBinary(FIXTURE) as binary:
        assert binary.is_64bit
        assert binary.entry > 0
        assert binary.code is not None
        assert binary.code.size > 0


def test_symbol_table_has_counter_and_start() -> None:
    from rotor import RISCVBinary

    with RISCVBinary(FIXTURE) as binary:
        assert "counter" in binary.symbols
        assert "_start" in binary.symbols
        assert binary.symbols["counter"].kind == "func"


def test_function_bounds_from_dwarf() -> None:
    from rotor import RISCVBinary

    with RISCVBinary(FIXTURE) as binary:
        low, high = binary.function_bounds("counter")
        assert high > low
        # The counter function should be a handful of instructions, not KBs.
        assert high - low < 512


def test_dwarf_line_map_covers_source() -> None:
    from rotor import RISCVBinary

    with RISCVBinary(FIXTURE) as binary:
        low, high = binary.function_bounds("counter")
        locations = []
        for pc in range(low, high, 4):
            loc = binary.pc_to_source(pc)
            if loc is not None:
                locations.append(loc)
        assert locations, "expected at least one PC → source mapping"
        # All resolved source locations should point at counter.c.
        assert all("counter.c" in loc.file for loc in locations)


def test_dwarf_extracts_locals_and_parameters() -> None:
    from rotor import RISCVBinary

    with RISCVBinary(FIXTURE) as binary:
        fi = binary.function_by_name("counter")
        assert fi is not None
        names = {v.name for v in fi.locals} | {v.name for v in fi.parameters}
        assert "n" in names
        assert "sum" in names


# ──────────────────────────────────────────────────────────────────────────
# Phase 2/3: Native BTOR2 model construction
# ──────────────────────────────────────────────────────────────────────────


def test_build_native_model_on_fixture() -> None:
    from rotor import ModelConfig, RISCVBinary, RotorInstance

    with RISCVBinary(FIXTURE) as binary:
        low, high = binary.function_bounds("counter")
        cfg = ModelConfig(
            is_64bit=True,
            code_start=low,
            code_end=high,
            bound=50,
            model_backend="python",
        )
        inst = RotorInstance(binary, cfg)
        inst.build_machine()
        assert inst.model is not None
        nodes = inst.model.dag.nodes()
        assert len(nodes) > 200, "native model should be non-trivial"
        # Code segment must be populated with writes (one per byte).
        write_nodes = [n for n in nodes if n.op == "write"]
        assert len(write_nodes) >= (high - low), (
            f"expected ≥{high - low} code-byte writes, got {len(write_nodes)}"
        )


def test_emit_btor2_roundtrips_through_parser() -> None:
    from rotor import ModelConfig, RISCVBinary, RotorInstance
    from rotor.btor2 import BTOR2Printer, parse_btor2

    with RISCVBinary(FIXTURE) as binary:
        low, high = binary.function_bounds("counter")
        cfg = ModelConfig(
            is_64bit=True,
            code_start=low,
            code_end=high,
            bound=50,
            model_backend="python",
        )
        inst = RotorInstance(binary, cfg)
        inst.build_machine()
        text = inst.emit_btor2()

    # Round-trip through the parser + printer should preserve line count.
    dag2 = parse_btor2(text)
    text2 = BTOR2Printer().render(dag2)
    assert text.count("\n") == text2.count("\n")


# ──────────────────────────────────────────────────────────────────────────
# Phase 7: API — expression compilation against the real model
# ──────────────────────────────────────────────────────────────────────────


def test_expression_compiler_resolves_symbols() -> None:
    from rotor import ModelConfig, RISCVBinary, RotorInstance
    from rotor.expr import ExpressionCompiler

    with RISCVBinary(FIXTURE) as binary:
        low, high = binary.function_bounds("counter")
        cfg = ModelConfig(
            is_64bit=True,
            code_start=low,
            code_end=high,
            model_backend="python",
        )
        inst = RotorInstance(binary, cfg)
        inst.build_machine()
        # Each of these should compile to a 1-bit predicate rooted at the
        # expected BTOR2 op.
        for expr, expected_op in [
            (f"pc == 0x{low:x}", "eq"),
            ("a0 < 0", "slt"),
            ("a0 == 0 || a1 == 0", "or"),
            (f"pc == 0x{low + 8:x} && a0 == 42", "and"),
        ]:
            node = ExpressionCompiler.compile(inst, expr)
            assert node.sort.width == 1
            assert node.op == expected_op, f"{expr!r}: expected {expected_op}, got {node.op}"


# ──────────────────────────────────────────────────────────────────────────
# Phase 5: Source-trace rendering against the real DWARF info
# ──────────────────────────────────────────────────────────────────────────


def test_source_trace_from_synthetic_witness() -> None:
    """Construct a minimal trace by hand and verify it renders against DWARF."""
    from rotor import RISCVBinary, SourceTrace
    from rotor.trace import MachineState

    with RISCVBinary(FIXTURE) as binary:
        low, high = binary.function_bounds("counter")
        # Pick PCs that hit different source lines inside counter().
        sampled = []
        for pc in range(low, high, 4):
            loc = binary.pc_to_source(pc)
            if loc is not None:
                sampled.append((pc, loc))
            if len(sampled) >= 3:
                break

        states = [
            MachineState(step=i, pc=pc, registers={10: i, 11: 10})
            for i, (pc, _) in enumerate(sampled)
        ]
        trace = SourceTrace(states, binary)
        assert len(trace) == len(sampled)
        assert all(step.function_name == "counter" for step in trace.steps)

        md = trace.as_markdown()
        assert "counter.c" in md
        json_out = trace.as_json()
        assert "counter.c" in json_out


# ──────────────────────────────────────────────────────────────────────────
# Phase 3: RotorInstance.check against a stub solver (no external tool)
# ──────────────────────────────────────────────────────────────────────────


def test_instance_check_with_missing_solver_returns_unknown() -> None:
    """Without an installed solver, check() should degrade gracefully."""
    from rotor import ModelConfig, RISCVBinary, RotorInstance

    with RISCVBinary(FIXTURE) as binary:
        low, high = binary.function_bounds("counter")
        cfg = ModelConfig(
            is_64bit=True,
            code_start=low,
            code_end=high,
            solver="btormc",  # not on PATH in this env
            bound=5,
            model_backend="python",
        )
        inst = RotorInstance(binary, cfg)
        result = inst.check()
        assert result.verdict == "unknown"
        assert "not found" in result.stderr or "no backends" in result.stderr
