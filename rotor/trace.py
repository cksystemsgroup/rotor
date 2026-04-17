"""Phase 5: Source-level trace.

Transforms a sequence of :class:`MachineState` (derived from a solver
witness) into a human-readable, source-annotated :class:`SourceTrace` using
DWARF line/variable information from a :class:`RISCVBinary`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from rotor.binary import RISCVBinary, SourceLocation


# ──────────────────────────────────────────────────────────────────────────
# Machine state and source step
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class MachineState:
    step: int
    pc: int
    registers: dict[int, int] = field(default_factory=dict)
    memory: dict[int, int] = field(default_factory=dict)


@dataclass
class SourceStep:
    step: int
    pc: int
    location: "SourceLocation | None"
    function_name: str
    instruction: str
    variables: dict[str, int] = field(default_factory=dict)
    changed: set[str] = field(default_factory=set)

    def describe(self) -> str:
        loc = str(self.location) if self.location else "<unknown>"
        changed = (
            " [" + ", ".join(sorted(self.changed)) + "]" if self.changed else ""
        )
        return (
            f"step {self.step:3d}  {loc}  {self.function_name}  "
            f"pc=0x{self.pc:x}  {self.instruction}{changed}"
        )


# ──────────────────────────────────────────────────────────────────────────
# SourceTrace
# ──────────────────────────────────────────────────────────────────────────


class SourceTrace:
    """A source-annotated view of a concrete execution trace.

    Construct from a list of :class:`MachineState` frames and a
    :class:`RISCVBinary`: the binary's DWARF info is used to resolve each
    frame's PC to a source location, identify the enclosing function, list
    live variables, and track which variables changed between frames.
    """

    def __init__(
        self, machine_states: list[MachineState], binary: "RISCVBinary"
    ) -> None:
        self.binary = binary
        self.steps: list[SourceStep] = []
        prev: dict[str, int] = {}
        for state in machine_states:
            loc = binary.pc_to_source(state.pc)
            func = binary.function_at(state.pc)
            live = binary.live_variables_at(state.pc)
            variables: dict[str, int] = {}
            for var in live:
                value = binary.resolve_variable(var, state.registers, state.memory)
                if value is not None:
                    variables[var.name] = value
            changed = {
                name for name, value in variables.items()
                if prev.get(name) != value
            }
            self.steps.append(
                SourceStep(
                    step=state.step,
                    pc=state.pc,
                    location=loc,
                    function_name=func.name if func else "?",
                    instruction=binary.disassemble(state.pc),
                    variables=variables,
                    changed=changed,
                )
            )
            prev = variables

    # ---------------------------------------------------------- dunder

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __str__(self) -> str:
        return "\n".join(step.describe() for step in self.steps)

    # ---------------------------------------------------------- formatters

    def as_markdown(self) -> str:
        """Render the trace as a Markdown table."""
        if not self.steps:
            return "_empty trace_"
        lines = [
            "| step | location | function | pc | instruction | changed |",
            "| ---: | :------- | :------- | :- | :---------- | :------ |",
        ]
        for step in self.steps:
            loc = str(step.location) if step.location else ""
            changed = ", ".join(sorted(step.changed))
            lines.append(
                f"| {step.step} | {loc} | {step.function_name} | "
                f"0x{step.pc:x} | `{step.instruction}` | {changed} |"
            )
        return "\n".join(lines)

    def as_sarif(self) -> dict:
        """Render a SARIF 2.1.0 ``threadFlow`` fragment."""
        locations = []
        for step in self.steps:
            if step.location is None:
                continue
            locations.append(
                {
                    "location": {
                        "physicalLocation": {
                            "artifactLocation": {"uri": step.location.file},
                            "region": {
                                "startLine": step.location.line,
                                "startColumn": max(1, step.location.column),
                            },
                        },
                        "message": {
                            "text": f"{step.function_name}: {step.instruction}"
                        },
                    },
                    "nestingLevel": 0,
                    "executionOrder": step.step,
                    "state": {k: {"text": str(v)} for k, v in step.variables.items()},
                }
            )
        return {
            "version": "2.1.0",
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "runs": [
                {
                    "tool": {"driver": {"name": "rotor", "version": "0.1.0"}},
                    "results": [
                        {
                            "ruleId": "rotor.trace",
                            "message": {"text": "Rotor counterexample trace"},
                            "codeFlows": [{"threadFlows": [{"locations": locations}]}],
                        }
                    ],
                }
            ],
        }

    def as_json(self) -> str:
        """Render the trace as JSON (serializable; useful for LLM consumption)."""
        payload = []
        for step in self.steps:
            payload.append(
                {
                    "step": step.step,
                    "pc": step.pc,
                    "location": (
                        None
                        if step.location is None
                        else {
                            "file": step.location.file,
                            "line": step.location.line,
                            "column": step.location.column,
                        }
                    ),
                    "function": step.function_name,
                    "instruction": step.instruction,
                    "variables": step.variables,
                    "changed": sorted(step.changed),
                }
            )
        return json.dumps(payload, indent=2)

    def as_gdb_script(self) -> str:
        """Render a GDB script that drives a debugger along the trace."""
        lines = ["# rotor counterexample replay"]
        for step in self.steps:
            lines.append(f"# step {step.step}")
            if step.location is not None:
                lines.append(f"break {step.location.file}:{step.location.line}")
        lines.append("run")
        return "\n".join(lines)
