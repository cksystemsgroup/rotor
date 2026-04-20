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


ADD2 = "tests/fixtures/add2.elf"
BRANCHES = "tests/fixtures/branches.elf"

CORPUS: tuple[CorpusEntry, ...] = (
    # add2.elf — straight-line + simple conditional fixture (M1)
    CorpusEntry("add2-entry-trivial",        ADD2, "add2", 0x00, 0, "reachable",   0),
    CorpusEntry("add2-ret-one-step",         ADD2, "add2", 0x04, 1, "reachable",   1),
    CorpusEntry("add2-unreach-within-bound", ADD2, "add2", 0x08, 1, "unreachable", None),

    CorpusEntry("sign-entry-trivial",        ADD2, "sign", 0x00, 0, "reachable",   0),
    CorpusEntry("sign-li-branch",            ADD2, "sign", 0x10, 2, "reachable",   1),
    CorpusEntry("sign-ret2-via-positive",    ADD2, "sign", 0x14, 2, "reachable",   2),
    CorpusEntry("sign-ret1-via-negative",    ADD2, "sign", 0x0C, 3, "reachable",   3),
    CorpusEntry("sign-unreach-within-bound", ADD2, "sign", 0x04, 0, "unreachable", None),

    # branches.elf — exercises beq / bge / bltu / ori / li / mv / jal / ret (M5)
    CorpusEntry("branches-entry-trivial",    BRANCHES, "branches", 0x00, 0, "reachable",   0),
    CorpusEntry("branches-after-mv-and-li",  BRANCHES, "branches", 0x08, 2, "reachable",   2),
    CorpusEntry("branches-ret-reachable",    BRANCHES, "branches", 0x20, 7, "reachable",   7),
    CorpusEntry("branches-jal-reachable",    BRANCHES, "branches", 0x28, 6, "reachable",   6),
    CorpusEntry("branches-c4-unreach-at-2",  BRANCHES, "branches", 0x14, 2, "unreachable", None),
)
