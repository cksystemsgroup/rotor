"""L0-equivalence harness.

Every IR that ships in rotor must produce, for each (binary, question)
entry in the corpus, the same verdict (and reached step for `reachable`)
that L0's IdentityEmitter produces. This is the correctness oracle
that lets IR optimizations ship without compromising trust.

M4 registers only IdentityEmitter, so the test is effectively a
self-consistency check today. L1 (DAG), L2 (SSA-BV), L3 (BVDD)
extend the `EMITTERS` tuple as they land.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from rotor import EngineConfig, IdentityEmitter, RotorEngine
from rotor.binary import RISCVBinary
from rotor.ir.emitter import BTOR2Emitter
from tests.equivalence.corpus import CORPUS, CorpusEntry

REPO_ROOT = Path(__file__).resolve().parents[2]

EmitterFactory = Callable[[RISCVBinary], BTOR2Emitter]

EMITTERS: tuple[tuple[str, EmitterFactory], ...] = (
    ("identity", IdentityEmitter),
    # L1/L2/L3 factories slot in here.
)


@pytest.fixture(scope="module", params=[e[0] for e in EMITTERS], ids=[e[0] for e in EMITTERS])
def emitter_factory(request) -> EmitterFactory:  # noqa: ANN001
    return dict(EMITTERS)[request.param]


@pytest.mark.parametrize("entry", CORPUS, ids=[e.name for e in CORPUS])
def test_emitter_matches_l0(entry: CorpusEntry, emitter_factory: EmitterFactory) -> None:
    binary_path = REPO_ROOT / entry.binary_relpath
    with RISCVBinary(binary_path) as binary:
        fn = binary.function(entry.function)
        target_pc = fn.start + entry.target_offset

        engine = RotorEngine(
            binary,
            config=EngineConfig(emitter_factory=emitter_factory, default_bound=entry.bound),
        )
        result = engine.check_reach(
            function=entry.function,
            target_pc=target_pc,
            bound=entry.bound,
        )

    assert result.verdict == entry.expected_verdict, (
        f"{emitter_factory.__name__} gave {result.verdict} on {entry.name}; "
        f"L0 expects {entry.expected_verdict}"
    )
    if entry.expected_verdict == "reachable":
        assert result.step == entry.expected_step, (
            f"{emitter_factory.__name__} reached at step {result.step} on {entry.name}; "
            f"L0 expects step {entry.expected_step}"
        )
