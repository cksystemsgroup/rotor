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
MEMOPS = "tests/fixtures/memops.elf"
RODATA = "tests/fixtures/rodata.elf"
MULT = "tests/fixtures/mult.elf"
RVC = "tests/fixtures/rvc.elf"
BITOPS = "tests/fixtures/bitops.elf"

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

    # memops.elf — exercises the M6 memory model (lw / sw / store-then-load).
    CorpusEntry("load_sum-entry-trivial",    MEMOPS, "load_sum",  0x00, 0, "reachable",   0),
    CorpusEntry("load_sum-second-lw",        MEMOPS, "load_sum",  0x04, 1, "reachable",   1),
    CorpusEntry("load_sum-addw",             MEMOPS, "load_sum",  0x08, 2, "reachable",   2),
    CorpusEntry("load_sum-ret",              MEMOPS, "load_sum",  0x0C, 3, "reachable",   3),
    CorpusEntry("load_sum-unreach-at-0",     MEMOPS, "load_sum",  0x04, 0, "unreachable", None),

    CorpusEntry("roundtrip-entry-trivial",   MEMOPS, "roundtrip", 0x00, 0, "reachable",   0),
    CorpusEntry("roundtrip-after-sw",        MEMOPS, "roundtrip", 0x04, 1, "reachable",   1),
    CorpusEntry("roundtrip-after-lw",        MEMOPS, "roundtrip", 0x08, 2, "reachable",   2),
    CorpusEntry("roundtrip-ret",             MEMOPS, "roundtrip", 0x0C, 3, "reachable",   3),

    # rodata.elf — exercises ELF segment init + lw from a rodata table.
    CorpusEntry("pick-entry-trivial",        RODATA, "pick", 0x00, 0, "reachable",   0),
    CorpusEntry("pick-after-auipc",          RODATA, "pick", 0x08, 2, "reachable",   2),
    CorpusEntry("pick-lw",                   RODATA, "pick", 0x14, 5, "reachable",   5),
    CorpusEntry("pick-ret",                  RODATA, "pick", 0x18, 6, "reachable",   6),
    CorpusEntry("pick-unreach-at-4",         RODATA, "pick", 0x10, 3, "unreachable", None),

    # mult.elf — exercises the RV64M extension (mul / divuw / mulw).
    CorpusEntry("mul_add-entry-trivial",     MULT, "mul_add", 0x00, 0, "reachable",   0),
    CorpusEntry("mul_add-ret",               MULT, "mul_add", 0x08, 2, "reachable",   2),
    CorpusEntry("divmod-entry-trivial",      MULT, "divmod",  0x00, 0, "reachable",   0),
    CorpusEntry("divmod-ret",                MULT, "divmod",  0x10, 4, "reachable",   4),
    CorpusEntry("mul64-entry-trivial",       MULT, "mul64",   0x00, 0, "reachable",   0),
    CorpusEntry("mul64-ret",                 MULT, "mul64",   0x04, 1, "reachable",   1),

    # rvc.elf — exercises the RVC (compressed) decoder + variable-length scanner.
    CorpusEntry("add_rvc-entry-trivial",     RVC, "add_rvc", 0x00, 0, "reachable",   0),
    CorpusEntry("add_rvc-ret",               RVC, "add_rvc", 0x02, 1, "reachable",   1),
    CorpusEntry("triple-entry-trivial",      RVC, "triple",  0x00, 0, "reachable",   0),
    CorpusEntry("triple-ret",                RVC, "triple",  0x08, 3, "reachable",   3),
    CorpusEntry("signbit-ret",               RVC, "signbit", 0x04, 1, "reachable",   1),

    # bitops.elf — Track B.4 capstone fixture combining RV64M + RVC
    # in a realistic mini bit-manipulation library.
    CorpusEntry("is_power_of_two-entry",     BITOPS, "is_power_of_two", 0x00, 0, "reachable", 0),
    CorpusEntry("is_power_of_two-ret",       BITOPS, "is_power_of_two", 0x0c, 4, "reachable", 4),
    CorpusEntry("popcount-entry",            BITOPS, "popcount",        0x00, 0, "reachable", 0),
    CorpusEntry("shifted_mul-entry",         BITOPS, "shifted_mul",     0x00, 0, "reachable", 0),
    CorpusEntry("shifted_mul-ret",           BITOPS, "shifted_mul",     0x08, 2, "reachable", 2),
)
