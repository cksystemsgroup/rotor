"""Tests for :mod:`rotor.engine` utilities."""

from __future__ import annotations

import pytest

from rotor.engine import _topological_sort


def test_toposort_linear() -> None:
    graph = {"a": ["b"], "b": ["c"], "c": []}
    order = _topological_sort(graph)
    assert order.index("c") < order.index("b") < order.index("a")


def test_toposort_diamond() -> None:
    graph = {
        "top": ["left", "right"],
        "left": ["bottom"],
        "right": ["bottom"],
        "bottom": [],
    }
    order = _topological_sort(graph)
    assert order.index("bottom") < order.index("left")
    assert order.index("bottom") < order.index("right")
    assert order.index("left") < order.index("top")
    assert order.index("right") < order.index("top")


def test_toposort_cycle_raises() -> None:
    graph = {"a": ["b"], "b": ["a"]}
    with pytest.raises(ValueError):
        _topological_sort(graph)


def test_path_condition_empty() -> None:
    from rotor.engine import PathCondition

    pc = PathCondition.empty()
    assert pc.conjuncts == []
    pc2 = pc.extend(lambda inst: None)  # type: ignore[arg-type]
    assert len(pc2.conjuncts) == 1
