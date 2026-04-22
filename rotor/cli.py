"""Command-line interface for rotor.

Subcommands map 1:1 to API verbs:

    rotor info <elf> [--functions]          (metadata inspection)
    rotor disasm <elf> --function <name>    (RISC-V disassembly)
    rotor reach  <elf> --function <name> --target <pc> [--bound N] [--trace FILE]

Plus two utilities that operate on BTOR2 text directly (debugging and
benchmarking seam; rotor's compile pipeline is unchanged):

    rotor btor2-roundtrip <file.btor2>      (parse then re-emit)
    rotor solve-btor2     <file.btor2> [--bound N]... [--timeout T]

Exit codes follow the PLAN convention:

    0    safe / proved / equivalent   (e.g. reach returned `unreachable`)
    1    reached / found / differs    (e.g. reach returned `reachable`)
    2    unknown / error

Output is plain text to stdout; counterexample markdown goes to stderr
by default, or to a file via `--trace`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, TextIO

from rotor.api import RotorAPI
from rotor.binary import Function, RISCVBinary
from rotor.btor2.parser import Diagnostic, ParseResult, from_path as parse_btor2_file
from rotor.btor2.printer import to_text as btor2_to_text
from rotor.btor2.riscv.decoder import decode
from rotor.dwarf import DwarfLineMap
from rotor.riscv.disasm import disasm
from rotor.solvers.portfolio import Portfolio
from rotor.solvers.z3bv import Z3BMC

EXIT_OK = 0
EXIT_FOUND = 1
EXIT_UNKNOWN = 2


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rotor",
        description="A BTOR2 compiler for RISC-V questions.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    info = sub.add_parser("info", help="Summarize a RISC-V ELF")
    info.add_argument("elf", type=Path)
    info.add_argument("--functions", action="store_true",
                      help="List every function symbol with its PC range.")
    info.set_defaults(func=cmd_info)

    da = sub.add_parser("disasm", help="Disassemble a function")
    da.add_argument("elf", type=Path)
    da.add_argument("--function", required=True, help="Function symbol name.")
    da.set_defaults(func=cmd_disasm)

    rc = sub.add_parser("reach", help="Check reachability of a target PC")
    rc.add_argument("elf", type=Path)
    rc.add_argument("--function", required=True)
    rc.add_argument("--target", required=True,
                    help="Target PC as hex (e.g. 0x100b4) or decimal.")
    rc.add_argument("--bound", type=int, default=20,
                    help="BMC unroll bound (default: 20).")
    mode = rc.add_mutually_exclusive_group()
    mode.add_argument("--unbounded", action="store_true",
                      help="Use Z3 Spacer (PDR/IC3) instead of bounded BMC. "
                           "May answer `proved` with an inductive invariant.")
    mode.add_argument("--cegar", action="store_true",
                      help="Use counterexample-guided abstraction refinement. "
                           "Starts with every register havoc'd and refines on "
                           "spurious counterexamples.")
    mode.add_argument("--portfolio", action="store_true",
                      help="Race every available solver backend in parallel "
                           "(Z3 BMC/Spacer, Bitwuzla if installed, rIC3/btormc "
                           "if on PATH). First globally-conclusive verdict wins.")
    rc.add_argument("--trace", type=Path,
                    help="Write counterexample markdown to this path "
                         "instead of stderr.")
    rc.set_defaults(func=cmd_reach)

    vf = sub.add_parser(
        "verify",
        help="Verify a register-comparison predicate at every ret of a function.",
    )
    vf.add_argument("elf", type=Path)
    vf.add_argument("--function", required=True)
    vf.add_argument("--register", required=True,
                    help="Register to read at each ret — ABI name (e.g. a0) "
                         "or index (0..31).")
    vf.add_argument("--op", required=True,
                    choices=["eq", "neq", "slt", "slte", "sgt", "sgte",
                             "ult", "ulte", "ugt", "ugte"],
                    help="Comparison operator. Signedness is encoded in the name.")
    vf.add_argument("--value", required=True,
                    help="Right-hand side of the predicate (hex 0x.. or decimal, "
                         "signed).")
    vf.add_argument("--bound", type=int, default=20,
                    help="BMC unroll bound (default: 20).")
    vf.add_argument("--unbounded", action="store_true",
                    help="Use Z3 Spacer for unbounded verification.")
    vf.set_defaults(func=cmd_verify)

    rt = sub.add_parser(
        "btor2-roundtrip",
        help="Parse a BTOR2 file and re-emit it (diagnostics on stderr).",
    )
    rt.add_argument("file", type=Path)
    rt.set_defaults(func=cmd_btor2_roundtrip)

    sb = sub.add_parser(
        "solve-btor2",
        help="Solve the reachability (bad states) of a BTOR2 file via rotor's portfolio.",
    )
    sb.add_argument("file", type=Path)
    sb.add_argument(
        "--bound", type=int, action="append",
        help="BMC unroll bound; repeat to race multiple bounds (default: 20).",
    )
    sb.add_argument(
        "--timeout", type=float, default=None,
        help="Per-entry solver timeout in seconds (default: none).",
    )
    sb.set_defaults(func=cmd_solve_btor2)

    return p


def main(argv: Optional[Sequence[str]] = None,
         stdout: Optional[TextIO] = None,
         stderr: Optional[TextIO] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    try:
        return args.func(args, out, err)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=err)
        return EXIT_UNKNOWN
    except KeyError as exc:
        print(f"error: {exc}", file=err)
        return EXIT_UNKNOWN
    except ValueError as exc:
        print(f"error: {exc}", file=err)
        return EXIT_UNKNOWN


# ---------------------------------------------------------------------------
# subcommand implementations
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    with RISCVBinary(args.elf) as b:
        functions = sorted(b.functions.values(), key=lambda f: f.start)
        code_start = min((f.start for f in functions), default=0)
        code_end = max((f.end for f in functions), default=0)
        print(f"path      : {args.elf}", file=out)
        print(f"is_64bit  : {b.is_64bit}", file=out)
        print(f"entry     : 0x{b.entry:x}", file=out)
        if functions:
            print(
                f"code      : 0x{code_start:x}..0x{code_end:x} "
                f"({code_end - code_start} bytes)",
                file=out,
            )
        if args.functions:
            print("functions :", file=out)
            for fn in functions:
                print(f"  0x{fn.start:08x} +{fn.end - fn.start:<4} {fn.name}", file=out)
    return EXIT_OK


def cmd_disasm(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    with RISCVBinary(args.elf) as b:
        fn: Function = b.function(args.function)
        dwarf = DwarfLineMap(args.elf)
        for inst in b.instructions(fn):
            d = decode(inst.word)
            text = disasm(d) if d is not None else "<unsupported>"
            src = dwarf.lookup(inst.pc)
            src_str = f"  ; {Path(src.file).name}:{src.line}" if src is not None else ""
            print(f"  0x{inst.pc:08x}: {text:<22}{src_str}", file=out)
    return EXIT_OK


def cmd_reach(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from rotor.solvers import default_portfolio
    target = _parse_int(args.target)
    portfolio = default_portfolio(bound=args.bound) if args.portfolio else None
    with RotorAPI(args.elf, default_bound=args.bound, portfolio=portfolio) as api:
        if args.cegar:
            r = api.cegar_reach(function=args.function, target_pc=target)
        else:
            r = api.can_reach(
                function=args.function, target_pc=target,
                bound=args.bound, unbounded=args.unbounded,
            )

    print(f"verdict  : {r.verdict}", file=out)
    print(f"bound    : {r.bound}", file=out)
    if r.step is not None:
        print(f"step     : {r.step}", file=out)
    print(f"elapsed  : {r.elapsed * 1000:.1f}ms", file=out)
    print(f"backend  : {r.backend}", file=out)
    if r.invariant is not None:
        print(f"invariant: {r.invariant}", file=out)

    if r.trace is not None:
        md = r.trace.to_markdown()
        if args.trace is not None:
            args.trace.write_text(md)
            print(f"trace    : {args.trace}", file=out)
        else:
            print(md, file=err)

    if r.verdict == "reachable":
        return EXIT_FOUND
    if r.verdict in ("unreachable", "proved"):
        return EXIT_OK
    return EXIT_UNKNOWN


_ABI_TO_INDEX = {
    "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
    "t0": 5, "t1": 6, "t2": 7, "s0": 8, "fp": 8, "s1": 9,
    "a0": 10, "a1": 11, "a2": 12, "a3": 13, "a4": 14,
    "a5": 15, "a6": 16, "a7": 17,
    "s2": 18, "s3": 19, "s4": 20, "s5": 21, "s6": 22,
    "s7": 23, "s8": 24, "s9": 25, "s10": 26, "s11": 27,
    "t3": 28, "t4": 29, "t5": 30, "t6": 31,
}


def _parse_register(spec: str) -> int:
    """Accept either an ABI name (`a0`, `ra`, ...) or an index (`10`, `x10`)."""
    s = spec.strip().lower()
    if s in _ABI_TO_INDEX:
        return _ABI_TO_INDEX[s]
    if s.startswith("x") and s[1:].isdigit():
        return int(s[1:])
    if s.isdigit():
        return int(s)
    raise ValueError(f"unrecognized register spec: {spec!r}")


def cmd_verify(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    register = _parse_register(args.register)
    rhs = _parse_int(args.value)
    with RotorAPI(args.elf, default_bound=args.bound) as api:
        r = api.verify(
            function=args.function,
            register=register,
            comparison=args.op,
            rhs=rhs,
            bound=args.bound,
            unbounded=args.unbounded,
        )

    print(f"verdict  : {r.verdict}", file=out)
    print(f"bound    : {r.bound}", file=out)
    if r.step is not None:
        print(f"step     : {r.step}", file=out)
    print(f"elapsed  : {r.elapsed * 1000:.1f}ms", file=out)
    print(f"backend  : {r.backend}", file=out)
    if r.invariant is not None:
        print(f"invariant: {r.invariant}", file=out)
    if r.initial_regs:
        # Show the CEX initial regs compactly so users can see which
        # input makes the predicate fail.
        nonzero = {k: v for k, v in r.initial_regs.items() if v}
        if nonzero:
            print(f"initial  : "
                  + ", ".join(f"{k}=0x{v:x}" for k, v in sorted(nonzero.items())),
                  file=out)

    # Verify exit codes mirror reach, with the polarity stated explicitly:
    # reachable (predicate can fail)  → EXIT_FOUND (1)
    # unreachable/proved (predicate holds)  → EXIT_OK (0)
    # unknown → EXIT_UNKNOWN (2)
    if r.verdict == "reachable":
        return EXIT_FOUND
    if r.verdict in ("unreachable", "proved"):
        return EXIT_OK
    return EXIT_UNKNOWN


def cmd_btor2_roundtrip(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Parse a BTOR2 file and re-emit it, surfacing diagnostics on stderr.

    Useful for delta-debugging the parser / emitter pair and for
    normalizing benchmark files into rotor's canonical (dense-id,
    constd-only) form.
    """
    r = parse_btor2_file(args.file)
    _print_diagnostics(r, err)
    out.write(btor2_to_text(r.model))
    return EXIT_OK if r.ok else EXIT_UNKNOWN


def cmd_solve_btor2(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Solve a BTOR2 file's reachability via the Z3BMC portfolio.

    The default is a single entry at bound 20 (mirroring `rotor reach`).
    Passing `--bound` multiple times builds a race across those bounds;
    the portfolio short-circuits on the first globally-conclusive verdict
    (`reachable` or `proved`) and otherwise returns the deepest
    `unreachable`.
    """
    r = parse_btor2_file(args.file)
    _print_diagnostics(r, err)
    if not r.ok:
        return EXIT_UNKNOWN

    bounds = args.bound or [20]
    portfolio = Portfolio()
    for b in bounds:
        portfolio.add(Z3BMC(), bound=b, timeout=args.timeout)
    result = portfolio.check_reach(r.model)

    print(f"verdict  : {result.verdict}", file=out)
    print(f"bound    : {result.bound}", file=out)
    if result.step is not None:
        print(f"step     : {result.step}", file=out)
    print(f"elapsed  : {result.elapsed * 1000:.1f}ms", file=out)
    print(f"backend  : {result.backend}", file=out)
    if result.invariant is not None:
        print(f"invariant: {result.invariant}", file=out)
    if result.reason is not None:
        print(f"reason   : {result.reason}", file=out)

    if result.verdict == "reachable":
        return EXIT_FOUND
    if result.verdict in ("unreachable", "proved"):
        return EXIT_OK
    return EXIT_UNKNOWN


def _print_diagnostics(r: ParseResult, err: TextIO) -> None:
    for d in r.diagnostics:
        print(f"{d.severity}: line {d.line_no}: {d.message}", file=err)


def _parse_int(s: str) -> int:
    return int(s, 16) if s.lower().startswith("0x") else int(s)


if __name__ == "__main__":                 # pragma: no cover
    raise SystemExit(main())
