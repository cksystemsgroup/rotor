"""M3 shipping gate: the README Quick Start runs verbatim.

The test extracts each `$ rotor ...` command from the fenced console
block in README.md and executes it through rotor.cli.main, asserting
that each produces the exact stdout shown in the README (with elapsed
timing elided from the `reach` output because wall time is variable).
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from rotor.cli import main

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"


def _extract_commands_and_outputs() -> list[tuple[list[str], str]]:
    text = README.read_text()
    # The Quick Start's first console block contains all verb walkthroughs.
    match = re.search(r"```console\n(.*?)\n```", text, flags=re.DOTALL)
    assert match is not None, "no ```console block found in README"
    body = match.group(1)
    entries: list[tuple[list[str], str]] = []
    current_cmd: list[str] | None = None
    current_out: list[str] = []
    for line in body.splitlines():
        if line.startswith("$ "):
            if current_cmd is not None:
                entries.append((current_cmd, "\n".join(current_out).rstrip() + "\n"))
            current_cmd = line[2:].split()
            current_out = []
        elif current_cmd is not None:
            current_out.append(line)
    if current_cmd is not None:
        entries.append((current_cmd, "\n".join(current_out).rstrip() + "\n"))
    return entries


def _run(argv: list[str]) -> tuple[int, str, str]:
    # The commands in the README start with "rotor"; drop it before dispatch.
    assert argv[0] == "rotor"
    out, err = io.StringIO(), io.StringIO()
    code = main(argv[1:], stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def _normalize_reach_output(s: str) -> str:
    # elapsed has variable wall time; drop that line before comparison.
    return "\n".join(
        line for line in s.splitlines() if not line.startswith("elapsed")
    ) + "\n"


def test_readme_quickstart_runs_verbatim() -> None:
    entries = _extract_commands_and_outputs()
    assert len(entries) == 3

    # 1) rotor info
    argv, expected = entries[0]
    code, out, _ = _run(argv)
    assert code == 0
    assert out == expected

    # 2) rotor disasm
    argv, expected = entries[1]
    code, out, _ = _run(argv)
    assert code == 0
    assert out == expected

    # 3) rotor reach  (exit 1 = reachable; elapsed line elided)
    argv, expected = entries[2]
    code, out, _ = _run(argv)
    assert code == 1
    assert _normalize_reach_output(out) == expected
