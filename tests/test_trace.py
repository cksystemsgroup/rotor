"""Tests for :class:`SourceTrace` rendering without a real binary.

Uses a fake :class:`RISCVBinary`-like object to exercise the trace builder
and its formatters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from rotor.binary import DWARFLocation, FunctionInfo, SourceLocation, VariableInfo
from rotor.trace import MachineState, SourceTrace


@dataclass
class _FakeBinary:
    """Tiny stub implementing the handful of methods SourceTrace needs."""

    line_map: dict[int, SourceLocation]
    functions: list[FunctionInfo]
    live: dict[int, list[VariableInfo]]

    def pc_to_source(self, pc: int) -> SourceLocation | None:
        return self.line_map.get(pc)

    def function_at(self, pc: int) -> FunctionInfo | None:
        for fi in self.functions:
            if fi.contains(pc):
                return fi
        return None

    def live_variables_at(self, pc: int) -> list[VariableInfo]:
        return self.live.get(pc, [])

    def resolve_variable(self, var, registers, memory, frame_base: int = 0):
        # Fake resolution: treat every variable as coming from register 10
        # with an offset equal to the hash of its name modulo 8.
        return registers.get(10, 0)

    def disassemble(self, pc: int) -> str:
        return f"0x{pc:08x}: nop"


def _fake_binary() -> _FakeBinary:
    return _FakeBinary(
        line_map={
            0x1000: SourceLocation("main.c", 3, 5),
            0x1004: SourceLocation("main.c", 4, 5),
            0x1008: SourceLocation("main.c", 5, 5),
        },
        functions=[FunctionInfo(name="main", low_pc=0x1000, high_pc=0x1010)],
        live={
            0x1000: [VariableInfo("x", "int", 4, DWARFLocation(kind="register", register=10))],
            0x1004: [VariableInfo("x", "int", 4, DWARFLocation(kind="register", register=10))],
            0x1008: [VariableInfo("x", "int", 4, DWARFLocation(kind="register", register=10))],
        },
    )


def test_trace_step_ordering() -> None:
    binary = _fake_binary()
    states = [
        MachineState(step=0, pc=0x1000, registers={10: 0}),
        MachineState(step=1, pc=0x1004, registers={10: 7}),
        MachineState(step=2, pc=0x1008, registers={10: 7}),
    ]
    trace = SourceTrace(states, binary)  # type: ignore[arg-type]
    assert len(trace) == 3
    assert trace.steps[0].function_name == "main"
    assert trace.steps[0].variables["x"] == 0
    assert trace.steps[1].variables["x"] == 7
    assert "x" in trace.steps[1].changed
    assert "x" not in trace.steps[2].changed


def test_trace_markdown() -> None:
    binary = _fake_binary()
    states = [MachineState(step=0, pc=0x1000, registers={10: 42})]
    trace = SourceTrace(states, binary)  # type: ignore[arg-type]
    md = trace.as_markdown()
    assert "| step |" in md
    assert "main" in md


def test_trace_json() -> None:
    binary = _fake_binary()
    states = [MachineState(step=0, pc=0x1000, registers={10: 42})]
    trace = SourceTrace(states, binary)  # type: ignore[arg-type]
    doc = json.loads(trace.as_json())
    assert isinstance(doc, list) and doc[0]["variables"]["x"] == 42


def test_trace_sarif_structure() -> None:
    binary = _fake_binary()
    states = [MachineState(step=0, pc=0x1000, registers={10: 42})]
    trace = SourceTrace(states, binary)  # type: ignore[arg-type]
    sarif = trace.as_sarif()
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["tool"]["driver"]["name"] == "rotor"
