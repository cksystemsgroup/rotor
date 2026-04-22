"""Unit tests for `rotor.solvers.default_portfolio`.

The helper builds a Portfolio that races every available backend.
"Available" has two meanings:

  - In-process Python backends (Z3BMC, Z3Spacer) are always present.
  - Bitwuzla's and CVC5's Python bindings are optional (`pip install
    bitwuzla` / `pip install cvc5`) — the helper probes importability.
  - Subprocess backends (Pono, Ric3, BtorMC) check `shutil.which(...)`.

Unavailable backends are omitted rather than added as guaranteed-
`unknown` racers — a noise-contributor in the race would dilute the
portfolio's verdict strength.
"""

from __future__ import annotations

import shutil

from rotor.solvers import (
    BitwuzlaBMC,
    BtorMC,
    CVC5BMC,
    Pono,
    Portfolio,
    Ric3,
    Z3BMC,
    Z3Spacer,
    default_portfolio,
)


def _backend_types(p: Portfolio) -> list[type]:
    return [type(e.backend) for e in p.entries]


def test_default_portfolio_always_includes_z3_engines() -> None:
    p = default_portfolio()
    assert isinstance(p, Portfolio)
    types = _backend_types(p)
    assert Z3BMC in types
    assert Z3Spacer in types


def test_default_portfolio_includes_bitwuzla_when_importable() -> None:
    try:
        import bitwuzla                              # noqa: F401
    except ImportError:
        return                                        # environment-dependent
    p = default_portfolio()
    assert BitwuzlaBMC in _backend_types(p)


def test_default_portfolio_includes_cvc5_when_importable() -> None:
    try:
        import cvc5                                   # noqa: F401
    except ImportError:
        return
    p = default_portfolio()
    assert CVC5BMC in _backend_types(p)


def test_default_portfolio_skips_subprocess_when_binary_missing() -> None:
    p = default_portfolio()
    types = _backend_types(p)
    if shutil.which("pono") is None:
        assert Pono not in types
    if shutil.which("rIC3") is None:
        assert Ric3 not in types
    if shutil.which("btormc") is None:
        assert BtorMC not in types


def test_default_portfolio_adds_two_pono_engines_when_available() -> None:
    # When pono is installed, the portfolio adds both `bmc` (bounded)
    # and `ic3ia` (unbounded) as separate racers. Verify by checking
    # entry names when the binary is present; skip otherwise.
    if shutil.which("pono") is None:
        return
    p = default_portfolio()
    pono_modes = [
        e.backend.name for e in p.entries if isinstance(e.backend, Pono)
    ]
    assert "pono-bmc" in pono_modes
    assert "pono-ic3ia" in pono_modes


def test_default_portfolio_forwards_bound_and_timeout_to_bounded_entries() -> None:
    p = default_portfolio(bound=7, timeout=12.5)
    # Bounded engines (Z3BMC, BitwuzlaBMC, CVC5BMC, BtorMC) take the given bound;
    # unbounded engines (Z3Spacer, Ric3) are added with bound=0.
    for entry in p.entries:
        assert entry.timeout == 12.5
        if isinstance(entry.backend, (Z3BMC, BitwuzlaBMC, CVC5BMC, BtorMC)):
            assert entry.bound == 7
        elif isinstance(entry.backend, (Z3Spacer, Ric3)):
            assert entry.bound == 0
