"""Question specifications for rotor.

A QuestionSpec is what an emitter compiles into a BTOR2 Model. M1/M2/M3
ship only ReachSpec; future milestones introduce VerifySpec,
FindInputSpec, EquivalenceSpec.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionSpec:
    """Marker base for all emitter-consumable question specs."""


@dataclass(frozen=True)
class ReachSpec(QuestionSpec):
    """can_reach obligation: is target_pc reachable within the BMC bound?"""
    function: str
    target_pc: int
