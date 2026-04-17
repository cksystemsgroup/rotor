"""Tests for :class:`RotorInstance` (model-backend agnostic)."""

from __future__ import annotations

from rotor.btor2 import BTOR2Printer, NodeDAG, RISCVMachineBuilder
from rotor.instance import ModelConfig


def test_python_builder_produces_sorts_and_states() -> None:
    cfg = ModelConfig(is_64bit=True, cores=1)
    builder = RISCVMachineBuilder(cfg)
    model = builder.build()
    assert model.dag is builder.dag
    assert "register-file" in model.state_nodes
    # Memory segments
    for seg in ("code", "data", "heap", "stack"):
        assert seg in model.state_nodes
    # Sorts present in the DAG
    descriptions = {s.describe() for s in model.dag.sorts()}
    assert any("bitvec 64" in d for d in descriptions)
    assert any("bitvec 1" in d for d in descriptions)
    assert any(d.startswith("array") for d in descriptions)


def test_python_builder_register_value_node() -> None:
    cfg = ModelConfig(is_64bit=True, cores=1)
    builder = RISCVMachineBuilder(cfg)
    builder.build()
    a0 = builder.register_value(core=0, reg=10)
    # Reading from the same core/reg should be deduplicated.
    a0b = builder.register_value(core=0, reg=10)
    assert a0 is a0b


def test_python_builder_emits_btor2() -> None:
    cfg = ModelConfig(is_64bit=True, cores=1)
    builder = RISCVMachineBuilder(cfg)
    builder.build()
    text = BTOR2Printer().render(builder.dag)
    assert "sort bitvec 1" in text
    assert "sort bitvec 64" in text
    assert " state " in text or "state " in text
    assert " init " in text or "init " in text


def test_dag_nid_monotone() -> None:
    dag = NodeDAG()
    a = dag.intern_sort("bitvec", width=8)
    b = dag.intern_sort("bitvec", width=16)
    assert b.nid > a.nid
