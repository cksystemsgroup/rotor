"""Command-line entry point for Python Rotor.

Subcommands:

    info         print ELF/DWARF summary + supported-opcode coverage
    disasm       disassemble a function or a PC range
    btor2        emit BTOR2 text for a binary / function
    analyze      scan for unsupported instructions in a function
    check        BMC reachability of the illegal-instruction property
    reach        BMC: can this condition be reached? (RotorAPI.can_reach)
    find-input   BMC: find an input that triggers a condition
    verify       prove an invariant (bounded or unbounded via k-induction)
    equivalent   check two binaries produce identical outputs
    crotor-check check whether the external C Rotor binary is callable

All subcommands follow a common output convention:

    * verdict is always the first line
    * source-annotated traces (markdown) follow on SAT / reachable outcomes
    * proofs follow on UNSAT / holds outcomes
    * a non-zero exit code is returned for UNKNOWN or errors
"""

from __future__ import annotations

import argparse
import sys

from rotor.binary import RISCVBinary
from rotor.btor2 import CRotorBackend
from rotor.instance import ModelConfig, RotorInstance


# ──────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────


def _print(line: str = "", **kwargs) -> None:
    print(line, **kwargs)


def _exit_code_for(verdict: str) -> int:
    """Map rotor verdicts to POSIX exit codes.

    sat / reachable / found / differs / violated : 1  — something found
    unsat / unreachable / equivalent / holds      : 0  — proved safe/clean
    other (unknown, error)                        : 2
    """
    if verdict in ("sat", "reachable", "found", "differs", "violated"):
        return 1
    if verdict in ("unsat", "unreachable", "equivalent", "holds"):
        return 0
    return 2


def _maybe_print_trace(trace) -> None:
    if trace is not None:
        _print()
        _print(trace.as_markdown())


def _maybe_print_proof(proof: str | None, label: str = "proof") -> None:
    if proof:
        _print()
        _print(f"{label}:")
        for line in proof.splitlines():
            _print(f"  {line}")


def _warn_unsupported(binary: RISCVBinary, low: int, high: int) -> None:
    """Emit a stderr warning if the selected code range contains any
    instructions outside the native builder's subset."""
    from rotor.riscv import scan_unsupported_instructions, format_issues

    issues = scan_unsupported_instructions(binary, low, high)
    if issues:
        sys.stderr.write(format_issues(issues) + "\n")


def _resolve_function_range(
    binary: RISCVBinary, function: str | None,
    start: int | None, end: int | None,
) -> tuple[int, int]:
    if function:
        low, high = binary.function_bounds(function)
    else:
        assert binary.code is not None, "binary has no .text"
        low = start if start is not None else binary.code.start
        high = end if end is not None else binary.code.start + binary.code.size
    if start is not None:
        low = start
    if end is not None:
        high = end
    return low, high


# ──────────────────────────────────────────────────────────────────────────
# info / disasm / analyze
# ──────────────────────────────────────────────────────────────────────────


def _cmd_info(args: argparse.Namespace) -> int:
    with RISCVBinary(args.path) as binary:
        _print(f"path      : {binary._path}")
        _print(f"is_64bit  : {binary.is_64bit}")
        _print(f"entry     : 0x{binary.entry:x}")
        for name in ("code", "data", "rodata", "bss"):
            seg = getattr(binary, name)
            if seg:
                _print(
                    f"{name:9s} : 0x{seg.start:x}..0x{seg.start + seg.size:x} "
                    f"({seg.size} bytes)"
                )
        _print(f"symbols   : {len(binary.symbols)}")
        if args.functions:
            _print()
            _print("functions:")
            for name, sym in sorted(
                ((n, s) for n, s in binary.symbols.items() if s.kind == "func"),
                key=lambda kv: kv[1].address,
            ):
                _print(
                    f"  0x{sym.address:08x} +{sym.size:<4d} {name}"
                )
    return 0


def _cmd_disasm(args: argparse.Namespace) -> int:
    with RISCVBinary(args.path) as binary:
        low, high = _resolve_function_range(binary, args.function, args.start, args.end)
        for pc in range(low, high, 4):
            loc = binary.pc_to_source(pc)
            loc_str = f"   ; {loc}" if loc else ""
            _print(f"  0x{pc:08x}: {binary.disassemble(pc)}{loc_str}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from rotor.riscv import scan_unsupported_instructions

    with RISCVBinary(args.path) as binary:
        low, high = _resolve_function_range(binary, args.function, args.start, args.end)
        issues = scan_unsupported_instructions(binary, low, high)
        _print(
            f"range    : 0x{low:x}..0x{high:x} ({(high - low) // 4} instrs)"
        )
        _print(f"status   : {'clean' if not issues else f'{len(issues)} unsupported'}")
        for issue in issues:
            _print(f"  {issue}")
    return 0 if not issues else 1


# ──────────────────────────────────────────────────────────────────────────
# BTOR2 emission
# ──────────────────────────────────────────────────────────────────────────


def _cmd_btor2(args: argparse.Namespace) -> int:
    with RISCVBinary(args.path) as binary:
        low, high = _resolve_function_range(binary, args.function, args.start, args.end)
        config = ModelConfig(
            is_64bit=binary.is_64bit,
            code_start=low,
            code_end=high,
        )
        inst = RotorInstance(binary, config)
        inst.build_machine()
        btor2_text = inst.emit_btor2()
    if args.output == "-":
        sys.stdout.write(btor2_text)
    else:
        with open(args.output, "w") as fp:
            fp.write(btor2_text)
    return 0


# ──────────────────────────────────────────────────────────────────────────
# Low-level check
# ──────────────────────────────────────────────────────────────────────────


def _cmd_check(args: argparse.Namespace) -> int:
    with RISCVBinary(args.path) as binary:
        low, high = _resolve_function_range(binary, args.function, args.start, args.end)
        _warn_unsupported(binary, low, high)
        config = ModelConfig(
            is_64bit=binary.is_64bit,
            code_start=low,
            code_end=high,
            solver=args.solver,
            bound=args.bound,
        )
        inst = RotorInstance(binary, config)
        result = inst.check()
    _print(f"verdict  : {result.verdict}")
    _print(f"solver   : {result.solver}")
    _print(f"elapsed  : {result.elapsed:.3f}s")
    if result.steps is not None:
        _print(f"steps    : {result.steps}")
    if result.stderr and args.verbose:
        _print(f"stderr   : {result.stderr.strip()[:400]}")
    return _exit_code_for(result.verdict)


# ──────────────────────────────────────────────────────────────────────────
# RotorAPI subcommands
# ──────────────────────────────────────────────────────────────────────────


def _api(args: argparse.Namespace):
    """Construct a RotorAPI from common flags."""
    from rotor.api import RotorAPI

    return RotorAPI(
        args.path,
        default_solver=args.solver,
        default_bound=args.bound,
    )


def _cmd_reach(args: argparse.Namespace) -> int:
    api = _api(args)
    with api.binary:
        low, high = _resolve_function_range(api.binary, args.function, None, None)
        _warn_unsupported(api.binary, low, high)
        result = api.can_reach(
            function=args.function,
            condition=args.condition,
            bound=args.bound,
            unbounded=args.unbounded,
        )
    _print(f"verdict  : {result.verdict}")
    _print(f"elapsed  : {result.elapsed:.3f}s")
    _maybe_print_trace(result.trace)
    _maybe_print_proof(result.proof, "proof of unreachability")
    return _exit_code_for(result.verdict)


def _cmd_find_input(args: argparse.Namespace) -> int:
    api = _api(args)
    with api.binary:
        low, high = _resolve_function_range(api.binary, args.function, None, None)
        _warn_unsupported(api.binary, low, high)
        result = api.find_input(
            function=args.function,
            output_condition=args.condition,
            bound=args.bound,
        )
    _print(f"verdict  : {result.verdict}")
    _print(f"elapsed  : {result.elapsed:.3f}s")
    if result.input_bytes is not None:
        _print(f"input    : {result.input_bytes!r}")
    _maybe_print_trace(result.trace)
    return _exit_code_for(result.verdict)


def _cmd_verify(args: argparse.Namespace) -> int:
    api = _api(args)
    with api.binary:
        low, high = _resolve_function_range(api.binary, args.function, None, None)
        _warn_unsupported(api.binary, low, high)
        result = api.verify(
            function=args.function,
            invariant=args.invariant,
            bound=args.bound,
            unbounded=args.unbounded,
        )
    _print(f"verdict  : {result.verdict}")
    _print(f"unbounded: {result.unbounded}")
    _print(f"elapsed  : {result.elapsed:.3f}s")
    _maybe_print_proof(result.proof, "inductive invariant")
    _maybe_print_trace(result.counterexample)
    return _exit_code_for(result.verdict)


def _cmd_equivalent(args: argparse.Namespace) -> int:
    api = _api(args)
    with api.binary:
        result = api.are_equivalent(
            other_binary=args.other,
            function=args.function,
            bound=args.bound,
            unbounded=args.unbounded,
        )
    _print(f"verdict  : {result.verdict}")
    _print(f"elapsed  : {result.elapsed:.3f}s")
    _maybe_print_trace(result.trace_a)
    _maybe_print_proof(result.proof, "equivalence proof")
    return _exit_code_for(result.verdict)


def _cmd_crotor_check(args: argparse.Namespace) -> int:
    backend = CRotorBackend(args.binary)
    if backend.available():
        _print(f"C Rotor available: {backend.binary}")
        return 0
    sys.stderr.write(f"C Rotor not found: {backend.binary}\n")
    return 1


# ──────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────


def _add_function_range_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--function", help="narrow the analysis to a named function")
    p.add_argument("--start", type=lambda s: int(s, 0),
                   help="start PC (default: function or .text start)")
    p.add_argument("--end", type=lambda s: int(s, 0),
                   help="end PC (default: function or .text end)")


def _add_solver_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--solver", default="bitwuzla",
                   choices=["bitwuzla", "btormc", "ic3", "kind", "portfolio"])
    p.add_argument("--bound", type=int, default=100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rotor",
        description="Python Rotor — verify RISC-V ELF binaries with BTOR2/BMC.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="print ELF/DWARF summary")
    p_info.add_argument("path")
    p_info.add_argument("--functions", action="store_true",
                        help="list functions in the symbol table")
    p_info.set_defaults(func=_cmd_info)

    p_disasm = sub.add_parser("disasm", help="disassemble a function or range")
    p_disasm.add_argument("path")
    _add_function_range_args(p_disasm)
    p_disasm.set_defaults(func=_cmd_disasm)

    p_analyze = sub.add_parser(
        "analyze", help="scan for unsupported instructions in a function",
    )
    p_analyze.add_argument("path")
    _add_function_range_args(p_analyze)
    p_analyze.set_defaults(func=_cmd_analyze)

    p_btor2 = sub.add_parser("btor2", help="emit BTOR2 text for a binary")
    p_btor2.add_argument("path")
    p_btor2.add_argument("-o", "--output", default="-",
                         help="output file (default: stdout)")
    _add_function_range_args(p_btor2)
    p_btor2.set_defaults(func=_cmd_btor2)

    p_check = sub.add_parser(
        "check", help="BMC reachability of the illegal-instruction property",
    )
    p_check.add_argument("path")
    _add_function_range_args(p_check)
    _add_solver_args(p_check)
    p_check.add_argument("--verbose", action="store_true")
    p_check.set_defaults(func=_cmd_check)

    p_reach = sub.add_parser(
        "reach",
        help="BMC: can CONDITION be reached during execution? (RotorAPI.can_reach)",
    )
    p_reach.add_argument("path")
    p_reach.add_argument("--function", required=True)
    p_reach.add_argument("--condition", required=True,
                         help='condition expression, e.g. "a0 < 0"')
    p_reach.add_argument("--unbounded", action="store_true")
    _add_solver_args(p_reach)
    p_reach.set_defaults(func=_cmd_reach)

    p_find = sub.add_parser(
        "find-input",
        help="find an input that causes CONDITION to hold at some step",
    )
    p_find.add_argument("path")
    p_find.add_argument("--function", required=True)
    p_find.add_argument("--condition", required=True)
    _add_solver_args(p_find)
    p_find.set_defaults(func=_cmd_find_input)

    p_verify = sub.add_parser(
        "verify",
        help="prove INVARIANT always holds (unbounded via k-induction)",
    )
    p_verify.add_argument("path")
    p_verify.add_argument("--function", required=True)
    p_verify.add_argument("--invariant", required=True)
    p_verify.add_argument("--unbounded", action="store_true")
    _add_solver_args(p_verify)
    p_verify.set_defaults(func=_cmd_verify)

    p_eq = sub.add_parser(
        "equivalent",
        help="check that two binaries produce identical output for a function",
    )
    p_eq.add_argument("path")
    p_eq.add_argument("--other", required=True, help="second ELF to compare")
    p_eq.add_argument("--function", required=True)
    p_eq.add_argument("--unbounded", action="store_true")
    _add_solver_args(p_eq)
    p_eq.set_defaults(func=_cmd_equivalent)

    p_cr = sub.add_parser("crotor-check", help="check that C Rotor is installed")
    p_cr.add_argument("--binary", default="rotor")
    p_cr.set_defaults(func=_cmd_crotor_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyError as err:
        sys.stderr.write(f"error: {err}\n")
        return 2
    except NotImplementedError as err:
        sys.stderr.write(f"error: {err}\n")
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
