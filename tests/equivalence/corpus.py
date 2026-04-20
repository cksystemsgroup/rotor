"""L0-equivalence corpus.

Each entry is a tuple (binary_relpath, function, target_offset, bound,
expected_verdict, expected_step). target_offset is added to
fn.start to get the absolute target PC. expected_step is the step at
which bad holds for `reachable` verdicts, or None for `unreachable`.

Keep the corpus small but structurally diverse: straight-line,
conditional branch (both arms), within-bound unreachability, and
trivial 0-step reachability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CorpusEntry:
    name: str
    binary_relpath: str
    function: str
    target_offset: int
    bound: int
    expected_verdict: str
    expected_step: Optional[int]


FIXTURE = "tests/fixtures/add2.elf"

CORPUS: tuple[CorpusEntry, ...] = (
    CorpusEntry("add2-entry-trivial",        FIXTURE, "add2", 0x00, 0, "reachable",   0),
    CorpusEntry("add2-ret-one-step",         FIXTURE, "add2", 0x04, 1, "reachable",   1),
    CorpusEntry("add2-unreach-within-bound", FIXTURE, "add2", 0x08, 1, "unreachable", None),

    CorpusEntry("sign-entry-trivial",        FIXTURE, "sign", 0x00, 0, "reachable",   0),
    CorpusEntry("sign-li-branch",            FIXTURE, "sign", 0x10, 2, "reachable",   1),
    CorpusEntry("sign-ret2-via-positive",    FIXTURE, "sign", 0x14, 2, "reachable",   2),
    CorpusEntry("sign-ret1-via-negative",    FIXTURE, "sign", 0x0C, 3, "reachable",   3),
    CorpusEntry("sign-unreach-within-bound", FIXTURE, "sign", 0x04, 0, "unreachable", None),
)
