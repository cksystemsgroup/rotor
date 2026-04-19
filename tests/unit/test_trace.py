from pathlib import Path

from rotor.binary import RISCVBinary
from rotor.dwarf import DwarfLineMap
from rotor.trace import build_trace

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "add2.elf"


def test_markdown_shape_on_add2() -> None:
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("add2")
        dwarf = DwarfLineMap(FIXTURE)
        t = build_trace(
            binary=b,
            function="add2",
            target_pc=fn.start + 4,
            verdict="reachable",
            bound=1,
            reached_at=1,
            elapsed=0.008,
            backend="z3-bmc",
            initial_regs={"x10": 3, "x11": 4, "x1": 0xFEEDBEEF},
            dwarf=dwarf,
        )
        md = t.to_markdown()

    assert md.startswith("# Counterexample: can_reach(add2")
    assert "**verdict**: reachable at step 1" in md
    assert "## Execution trace" in md
    assert "## Initial register values (witness)" in md
    # Both instructions appear and ret is pretty-printed.
    assert "addw a0, a0, a1" in md
    assert " ret" in md
    # Initial witness values rendered in hex (ra non-zero).
    assert "0x00000000feedbeef" in md
    # DWARF file present (add2.c) — only if DWARF info exists in the fixture.
    assert "add2.c" in md


def test_trace_without_dwarf() -> None:
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("add2")
        t = build_trace(
            binary=b,
            function="add2",
            target_pc=fn.start + 4,
            verdict="reachable",
            bound=1,
            reached_at=1,
            elapsed=0.001,
            backend="z3-bmc",
            initial_regs={},
            dwarf=None,
        )
        md = t.to_markdown()

    assert "| -                    |" in md or " | -" in md   # blank source column
