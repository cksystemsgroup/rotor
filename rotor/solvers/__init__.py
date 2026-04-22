"""Solver backends. Each backend consumes a BTOR2 Model and answers
reach/verify/... questions.

    Z3BMC        — bounded BMC via Z3:         `reachable` | `unreachable` | `unknown`
    BitwuzlaBMC  — bounded BMC via Bitwuzla:   typically 3–5× faster on BV workloads
    CVC5BMC      — bounded BMC via CVC5:       uncorrelated with Z3/Bitwuzla
    Z3Spacer     — unbounded PDR via Z3:       `reachable` | `proved` | `unknown`
    Pono         — subprocess bridge to Pono (Stanford):
                     bmc / ind / ic3ia / mbic3 / ic3sa / interp modes;
                     strong QF_ABV support via smt-switch

Pono reports `unknown` with a clear reason when its binary isn't
on PATH, so portfolios that include it work in environments that
have only a subset of the tools installed.

History: rotor 0.x shipped separate subprocess bridges to rIC3
(`rotor.solvers.Ric3`) and BtorMC (`rotor.solvers.BtorMC`). Pono
supersedes both — its `--engine mbic3` / `--engine ic3ia` stand
in for rIC3, and `--engine bmc` stands in for BtorMC. One
maintained adapter replaced two unexercised ones. Migrations:

    Ric3(extra_args=[...])  →  Pono(mode="mbic3")
                          or:  Pono(mode="ic3ia")
    BtorMC(mode="bmc")      →  Pono(mode="bmc")
    BtorMC(mode="kind")     →  Pono(mode="ind")
"""

from typing import Optional

from rotor.solvers.base import SolverBackend, SolverResult, Verdict
from rotor.solvers.pono import Pono
from rotor.solvers.portfolio import Portfolio, PortfolioEntry
from rotor.solvers.z3bv import Z3BMC
from rotor.solvers.z3spacer import Z3Spacer

# Optional in-process backends. Their underlying solver libraries
# aren't listed as hard requirements in pyproject.toml — rotor
# should import cleanly on a minimal install (just z3-solver +
# pyelftools). When the optional package is missing, the name is
# bound to `None` and `default_portfolio()` skips it; direct
# callers should prefer `from rotor.solvers.bitwuzla import
# BitwuzlaBMC` / `from rotor.solvers.cvc5bmc import CVC5BMC`, which
# raise a clearer ImportError pointing at the missing package.

try:
    from rotor.solvers.bitwuzla import BitwuzlaBMC
except ImportError:                                  # pragma: no cover
    BitwuzlaBMC = None  # type: ignore[assignment]

try:
    from rotor.solvers.cvc5bmc import CVC5BMC
except ImportError:                                  # pragma: no cover
    CVC5BMC = None  # type: ignore[assignment]


def default_portfolio(bound: int = 20, timeout: Optional[float] = 30.0) -> Portfolio:
    """Build a portfolio that races every available solver backend.

    Always-available (in-process Python): Z3BMC, Z3Spacer. Added
    when their pip packages are importable: BitwuzlaBMC, CVC5BMC.
    Pono is added (as two racers — `bmc` and `ic3ia`) if its
    binary is on PATH.

    Unavailable backends are omitted rather than added as guaranteed-
    `unknown` racers — a noise contributor in the race would dilute
    the portfolio's verdict strength.

    `bound` applies to the bounded engines; unbounded engines
    (Z3Spacer, Pono IC3IA) ignore it. `timeout` is per-entry.
    """
    p = Portfolio()
    p.add(Z3BMC(), bound=bound, timeout=timeout)
    p.add(Z3Spacer(), bound=0, timeout=timeout)
    if BitwuzlaBMC is not None:
        p.add(BitwuzlaBMC(), bound=bound, timeout=timeout)
    if CVC5BMC is not None:
        p.add(CVC5BMC(), bound=bound, timeout=timeout)
    # Pono: two distinct engines in the race when installed — its
    # `bmc` mode competes with Bitwuzla/CVC5/Z3 on BV, and its
    # `ic3ia` mode competes with Z3Spacer on unbounded PDR. They
    # count as separate Portfolio entries because they're genuinely
    # different algorithms sharing one binary.
    pono_bmc = Pono(mode="bmc")
    if pono_bmc.available:
        p.add(pono_bmc, bound=bound, timeout=timeout)
        p.add(Pono(mode="ic3ia"), bound=0, timeout=timeout)
    return p


__all__ = [
    "SolverBackend",
    "SolverResult",
    "Verdict",
    "Z3BMC",
    "Z3Spacer",
    "BitwuzlaBMC",
    "CVC5BMC",
    "Pono",
    "Portfolio",
    "PortfolioEntry",
    "default_portfolio",
]
