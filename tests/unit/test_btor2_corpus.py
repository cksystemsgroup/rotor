"""Golden-corpus parse/re-emit tests.

For each public-style BTOR2 file under ``tests/golden_corpus/``:

* ``from_text`` produces no diagnostics.
* ``to_text(parsed) -> from_text -> to_text`` is byte-identical
  (canonicalization fixpoint).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rotor.btor2 import from_text, to_text

CORPUS_DIR = Path(__file__).resolve().parent.parent / "golden_corpus"
CORPUS_FILES = sorted(CORPUS_DIR.glob("*.btor2"))


@pytest.mark.parametrize("path", CORPUS_FILES, ids=[p.name for p in CORPUS_FILES])
def test_corpus_parses_without_diagnostics(path: Path):
    parsed = from_text(path.read_text())
    assert parsed.diagnostics == [], parsed.diagnostics
    assert parsed.model.lines, "expected at least one line"


@pytest.mark.parametrize("path", CORPUS_FILES, ids=[p.name for p in CORPUS_FILES])
def test_corpus_canonical_fixpoint(path: Path):
    once = to_text(from_text(path.read_text()).model)
    twice = to_text(from_text(once).model)
    assert once == twice
