"""Tests for :mod:`rotor.witness`."""

from __future__ import annotations

from rotor.witness import parse_btor2_witness, reconstruct_machine_states


_SAMPLE_WITNESS = """\
sat
b0
#0
10 0 pc
11 [00000] 00000000000000000000000000000000 register-file
#1
10 100 pc
11 [00001] 00000000000000000000000000000001 register-file
.
"""


def test_parse_basic_frames() -> None:
    w = parse_btor2_witness(_SAMPLE_WITNESS)
    assert w.verdict == "sat"
    states = w.state_frames()
    assert len(states) == 2
    assert states[0].step == 0
    assert states[1].step == 1
    # The pc assignment in frame 1 should be parsed as binary 100 = 4.
    pc_frame1 = [a for a in states[1].assignments if a.symbol == "pc"][0]
    assert pc_frame1.value == 4


def test_array_assignment() -> None:
    w = parse_btor2_witness(_SAMPLE_WITNESS)
    states = w.state_frames()
    frame1_rf = [a for a in states[1].assignments if a.symbol == "register-file"][0]
    assert frame1_rf.index == 1
    assert frame1_rf.value == 1


def test_reconstruct_machine_states() -> None:
    w = parse_btor2_witness(_SAMPLE_WITNESS)
    states = reconstruct_machine_states(w)
    assert len(states) == 2
    assert states[0].pc == 0
    assert states[1].pc == 4
    assert states[1].registers[1] == 1


def test_parse_unsat() -> None:
    w = parse_btor2_witness("unsat\n")
    assert w.verdict == "unsat"
    assert w.frames == []


def test_parse_handles_hex_values() -> None:
    text = """\
sat
#0
10 ff pc
.
"""
    w = parse_btor2_witness(text)
    states = w.state_frames()
    assert states[0].assignments[0].value == 0xFF
