"""Phase 4: RotorEngine — orchestration over multiple RotorInstances."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable, Iterable

from rotor.binary import RISCVBinary
from rotor.btor2 import Node
from rotor.instance import ModelConfig, RotorInstance
from rotor.solvers.base import CheckResult


# ──────────────────────────────────────────────────────────────────────────
# Specifications + path conditions (used by compositional / concolic flows)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Specification:
    """Per-function pre/post condition pair, with a negated form for bad()."""

    precondition: Node
    postcondition: Node
    postcondition_negated: Node


@dataclass
class PathCondition:
    """A conjunction of branch predicates walked so far during concolic
    exploration. Each element is a BTOR2 node (or a deferred constructor)."""

    conjuncts: list[Node | Callable[[RotorInstance], Node]] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "PathCondition":
        return cls([])

    def extend(self, clause: Node | Callable[[RotorInstance], Node]) -> "PathCondition":
        return PathCondition(self.conjuncts + [clause])

    def as_node(self, inst: RotorInstance) -> Node:
        assert inst.model.builder is not None, "path condition requires Python builder"
        builder = inst.model.builder  # type: ignore[attr-defined]
        nodes: list[Node] = []
        for clause in self.conjuncts:
            nodes.append(clause(inst) if callable(clause) else clause)
        if not nodes:
            assert builder.NID_TRUE is not None
            return builder.NID_TRUE
        acc = nodes[0]
        for node in nodes[1:]:
            acc = builder.and_(acc, node, "path clause")
        return acc


# ──────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────


class RotorEngine:
    """High-level engine: creates and coordinates :class:`RotorInstance` objects."""

    def __init__(
        self,
        binary_path: str,
        default_solver: str = "bitwuzla",
        default_bound: int = 1000,
    ) -> None:
        self.binary = RISCVBinary(binary_path)
        self.default_solver = default_solver
        self.default_bound = default_bound

    # ---------------------------------------------------------- instance mint

    def create_instance(
        self,
        function: str | None = None,
        config: ModelConfig | None = None,
    ) -> RotorInstance:
        cfg = config or ModelConfig(
            solver=self.default_solver, bound=self.default_bound
        )
        if function:
            low, high = self.binary.function_bounds(function)
            cfg.code_start = low
            cfg.code_end = high
        inst = RotorInstance(self.binary, cfg)
        inst.build_machine()
        return inst

    # --------------------------------------------------------------- portfolio

    def run_portfolio(
        self,
        instances: list[RotorInstance],
        timeout: float = 300.0,
    ) -> list[CheckResult]:
        """Run ``instances`` in parallel; stop early on the first conclusive one."""
        results: list[CheckResult] = []
        if not instances:
            return results
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(instances)
        ) as pool:
            futures = {
                pool.submit(inst.check, inst.config.bound): inst
                for inst in instances
            }
            try:
                for future in concurrent.futures.as_completed(
                    futures, timeout=timeout
                ):
                    result = future.result()
                    results.append(result)
                    if result.is_conclusive():
                        for other in futures:
                            if other is not future:
                                other.cancel()
                        break
            except concurrent.futures.TimeoutError:
                pass
        return results

    # ------------------------------------------------- compositional verify

    def verify_compositionally(
        self,
        call_graph: dict[str, list[str]],
        specs: dict[str, Specification],
        bound: int = 1000,
    ) -> dict[str, CheckResult]:
        """Verify each function independently, callees before callers.

        Verified callee postconditions become constraints when checking
        callers — this avoids state-space blow-up from inlining.
        """
        results: dict[str, CheckResult] = {}
        for function in _topological_sort(call_graph):
            inst = self.create_instance(function)
            spec = specs[function]
            inst.add_constraint(spec.precondition, "precondition")
            for callee in call_graph.get(function, []):
                prior = results.get(callee)
                if prior is not None and prior.verdict == "unsat" and callee in specs:
                    inst.add_constraint(
                        specs[callee].postcondition,
                        f"callee-{callee}-postcondition",
                    )
            inst.add_bad(spec.postcondition_negated, "postcondition-violation")
            results[function] = inst.check(bound)
        return results

    # ----------------------------------------------------- concolic explore

    def explore_paths(
        self,
        function: str,
        max_paths: int = 100,
        bound: int = 1000,
    ) -> list:
        """Concolic exploration skeleton.

        The strategy is: check an empty path condition; for each SAT result,
        record the trace, then extend the worklist with negated branch
        decisions to explore unvisited paths. A full implementation requires
        branch-point extraction from the solver, which depends on the solver
        backend; this method returns the traces that the current backends
        produce directly.
        """
        from rotor.trace import SourceTrace

        paths: list[SourceTrace] = []
        worklist: list[PathCondition] = [PathCondition.empty()]
        while worklist and len(paths) < max_paths:
            condition = worklist.pop()
            inst = self.create_instance(function)
            if condition.conjuncts:
                inst.add_constraint(condition.as_node(inst), "path-condition")
            result = inst.check(bound)
            if result.verdict == "sat":
                witness = inst.get_witness()
                paths.append(SourceTrace(witness, self.binary))
                # Branch point negation is solver-specific: CheckResult has
                # `branch_points` as a hook. If empty we can't fork further.
                for branch in result.branch_points:
                    worklist.append(condition.extend(branch))
        return paths

    # ------------------------------------------------------------ utilities

    def _estimate_state_space(self, function: str) -> int:
        """Rough state-space estimate for solver-selection heuristics."""
        low, high = self.binary.function_bounds(function)
        return max(1, high - low)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """Callee-before-caller topological ordering."""
    order: list[str] = []
    permanent: set[str] = set()
    temporary: set[str] = set()

    def visit(node: str) -> None:
        if node in permanent:
            return
        if node in temporary:
            raise ValueError(f"cycle in call graph at {node}")
        temporary.add(node)
        for dep in graph.get(node, []):
            visit(dep)
        temporary.discard(node)
        permanent.add(node)
        order.append(node)

    for node in graph:
        visit(node)
    return order
