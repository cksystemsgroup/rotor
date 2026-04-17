"""BTOR2 witness parsing.

Turns the text witness emitted by BtorMC (and similar BMC backends) into a
list of :class:`~rotor.trace.MachineState` frames suitable for rendering as
a :class:`~rotor.trace.SourceTrace`.

Witness format (per the BTOR2 specification):

    sat
    b0 [b1 ...]      ← the bad properties that fired
    #0               ← frame 0 header
    <nid> <value> <symbol?>
    ...
    @0               ← input frame 0 header
    <nid> <value> <symbol?>
    ...
    #1
    ...
    .                ← end of witness
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rotor.trace import MachineState


_FRAME_STATE = re.compile(r"^#(\d+)")
_FRAME_INPUT = re.compile(r"^@(\d+)")
_ASSIGNMENT = re.compile(r"^(\d+)\s+([01]+|[0-9a-fA-F]+)(?:\s+(\S+))?\s*$")
_ARRAY_ASSIGNMENT = re.compile(r"^(\d+)\s+\[([01]+)\]\s+([01]+)(?:\s+(\S+))?\s*$")


@dataclass
class WitnessAssignment:
    nid: int
    value: int
    symbol: str = ""
    # For array writes: the index component of the assignment, if present.
    index: int | None = None


@dataclass
class WitnessFrame:
    step: int
    kind: str  # 'state' or 'input'
    assignments: list[WitnessAssignment] = field(default_factory=list)


@dataclass
class Witness:
    verdict: str = "sat"
    bad_properties: list[str] = field(default_factory=list)
    frames: list[WitnessFrame] = field(default_factory=list)

    def state_frames(self) -> list[WitnessFrame]:
        return [f for f in self.frames if f.kind == "state"]

    def input_frames(self) -> list[WitnessFrame]:
        return [f for f in self.frames if f.kind == "input"]


def parse_btor2_witness(text: str) -> Witness:
    """Parse a BTOR2 text witness into a :class:`Witness`."""
    w = Witness(verdict="unknown")
    current: WitnessFrame | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in ("sat", "unsat", "unknown"):
            w.verdict = line
            continue
        if line == ".":
            if current is not None:
                w.frames.append(current)
                current = None
            continue
        if line.startswith("b") and all(c.isdigit() for c in line[1:].split()[0] if c):
            w.bad_properties = line.split()
            continue
        state_match = _FRAME_STATE.match(line)
        if state_match:
            if current is not None:
                w.frames.append(current)
            current = WitnessFrame(step=int(state_match.group(1)), kind="state")
            continue
        input_match = _FRAME_INPUT.match(line)
        if input_match:
            if current is not None:
                w.frames.append(current)
            current = WitnessFrame(step=int(input_match.group(1)), kind="input")
            continue

        array_match = _ARRAY_ASSIGNMENT.match(line)
        if array_match and current is not None:
            nid = int(array_match.group(1))
            idx = int(array_match.group(2), 2)
            value = int(array_match.group(3), 2)
            symbol = array_match.group(4) or ""
            current.assignments.append(
                WitnessAssignment(nid=nid, value=value, symbol=symbol, index=idx)
            )
            continue

        assign_match = _ASSIGNMENT.match(line)
        if assign_match and current is not None:
            nid = int(assign_match.group(1))
            raw_val = assign_match.group(2)
            symbol = assign_match.group(3) or ""
            value = int(raw_val, 2) if set(raw_val) <= {"0", "1"} else int(raw_val, 16)
            current.assignments.append(
                WitnessAssignment(nid=nid, value=value, symbol=symbol)
            )
            continue

    if current is not None:
        w.frames.append(current)
    return w


# ──────────────────────────────────────────────────────────────────────────
# Reconstruct MachineState frames
# ──────────────────────────────────────────────────────────────────────────


def reconstruct_machine_states(
    witness: Witness,
    pc_symbol: str = "pc",
    register_file_symbol: str = "register-file",
) -> list[MachineState]:
    """Turn a parsed :class:`Witness` into a list of :class:`MachineState`.

    We identify frames by their ``#k`` step number and pull the PC and any
    register-file updates from the state assignments. Memory is preserved in
    a sparse byte dictionary only when ``register_file_symbol`` matches the
    model's register-file state node (the default covers the Python builder).
    """
    states: list[MachineState] = []
    registers: dict[int, int] = {}
    memory: dict[int, int] = {}

    for frame in witness.state_frames():
        pc_val = 0
        for a in frame.assignments:
            sym = a.symbol
            if sym == pc_symbol:
                pc_val = a.value
            elif sym == register_file_symbol and a.index is not None:
                registers[a.index] = a.value
            elif sym.endswith("-pc"):
                pc_val = a.value
        states.append(
            MachineState(
                step=frame.step,
                pc=pc_val,
                registers=dict(registers),
                memory=dict(memory),
            )
        )
    return states
