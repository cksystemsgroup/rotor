"""Tests for :mod:`rotor.expr`."""

from __future__ import annotations

from typing import Any

from rotor.btor2.builder import RISCVMachineBuilder
from rotor.expr import ExpressionCompiler, _tokenize
from rotor.instance import ModelConfig


def _mk_instance():
    """Build a minimal RotorInstance-like shim backed by the Python builder."""

    class Stub:
        def __init__(self) -> None:
            builder = RISCVMachineBuilder(ModelConfig(is_64bit=True, cores=1))
            self._model = builder.build()
            self._model.builder = builder
            self.config = builder.config

        @property
        def model(self):
            return self._model

    return Stub()


def test_tokenize_ints_and_ops() -> None:
    toks = _tokenize("pc == 0x1040")
    kinds = [t.kind for t in toks if t.kind != "end"]
    assert kinds == ["ident", "op", "int"]
    assert toks[0].text == "pc"
    assert toks[2].value == 0x1040


def test_tokenize_signed_literal() -> None:
    toks = _tokenize("-1")
    kinds = [t.kind for t in toks if t.kind != "end"]
    assert kinds == ["int"]
    assert toks[0].value == -1


def test_compile_pc_equals() -> None:
    stub = _mk_instance()
    node = ExpressionCompiler.compile(stub, "pc == 0x1040")
    assert node.sort.width == 1
    assert node.op == "eq"


def test_compile_register_compare() -> None:
    stub = _mk_instance()
    node = ExpressionCompiler.compile(stub, "a0 < a1")
    assert node.op == "slt"


def test_compile_bitmask() -> None:
    stub = _mk_instance()
    node = ExpressionCompiler.compile(stub, "x5 & 0xff != 0")
    # Top-level op is the != comparison; the & is a subexpression.
    assert node.op == "neq"


def test_compile_logical_combination() -> None:
    stub = _mk_instance()
    node = ExpressionCompiler.compile(stub, "a0 == 0 || a1 == 0")
    assert node.op == "or"


def test_unknown_identifier_raises() -> None:
    stub = _mk_instance()
    try:
        ExpressionCompiler.compile(stub, "unknown_var + 1")
    except ValueError as err:
        assert "unknown_var" in str(err)
    else:
        raise AssertionError("expected ValueError")
