"""Storage-safe Eclipse deck transformations."""

from __future__ import annotations

import re


class RestartOutputEnabledError(ValueError):
    """Raised when a deck can request mass-run restart output."""


def render_punq_opm_compatibility(deck: str) -> str:
    """Add explicit table/aquifer dimensions required by current OPM Flow."""
    records = (
        ("TABDIMS", "TABDIMS\n  1 1 50 50 /\n\n"),
        ("AQUDIMS", "AQUDIMS\n  2 100 1 36 2 100 /\n\n"),
        ("NUMRES", "NUMRES\n  1 /\n\n"),
        ("UNIFOUT", "UNIFOUT\n\n"),
    )
    active = _active_text(deck)
    missing = [
        record
        for keyword, record in records
        if not re.search(rf"(?im)^\s*{keyword}\s*$", active)
    ]
    rendered = deck
    if missing:
        grid = re.search(r"(?im)^\s*GRID\s*(?:\r?\n|$)", rendered)
        if grid is None:
            raise ValueError("PUNQ deck must contain a GRID section")
        rendered = rendered[: grid.start()] + "".join(missing) + rendered[grid.start() :]

    aquct = re.search(
        r"(?ims)(^\s*AQUCT\s*$)(.*?)(^\s*AQUANCON\s*$)", rendered
    )
    if aquct is not None:
        body_lines = [line.strip() for line in aquct.group(2).splitlines() if line.strip()]
        if not body_lines or body_lines[-1] != "/":
            replacement = aquct.group(1) + aquct.group(2).rstrip() + "\n/\n\n" + aquct.group(3)
            rendered = rendered[: aquct.start()] + replacement + rendered[aquct.end() :]
    rendered = re.sub(
        r"(?im)('PRO\*'\s+'SHUT')\s+6\*\s+120(?:\.0+)?\s*/",
        r"\1 'ORAT' 100.0 4* 120.0 /",
        rendered,
        count=1,
    )
    rendered = re.sub(
        r"(?ims)^\s*WCUTBACK\s*$.*?^\s*/\s*$",
        "-- FMGEO: WCUTBACK removed; OPM Flow 2026.04 does not implement it.\n",
        rendered,
        count=1,
    )
    rendered = re.sub(r"(?im)^\s*SEPARATE\s*$\s*", "", rendered, count=1)
    return rendered


def ensure_summary_keywords(deck: str, keywords: list[str] | tuple[str, ...]) -> str:
    """Add missing field summary requests at the start of the SUMMARY section."""
    normalized = [keyword.strip().upper() for keyword in keywords]
    if any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", keyword) for keyword in normalized):
        raise ValueError("summary keywords must be non-empty Eclipse identifiers")
    summary = re.search(r"(?im)^\s*SUMMARY\s*$", _active_text(deck))
    schedule = re.search(r"(?im)^\s*SCHEDULE\s*$", _active_text(deck))
    if summary is None or schedule is None or schedule.start() <= summary.end():
        raise ValueError("deck must contain SUMMARY before SCHEDULE")
    active_section = _active_text(deck)[summary.end() : schedule.start()]
    missing = [
        keyword
        for keyword in normalized
        if not re.search(rf"(?im)^\s*{re.escape(keyword)}(?:\s|/|$)", active_section)
    ]
    if not missing:
        return deck
    marker = re.search(r"(?im)^\s*SUMMARY\s*(?:\r?\n|$)", deck)
    if marker is None:  # pragma: no cover - guarded by the active-text search
        raise ValueError("deck must contain a SUMMARY section")
    insertion = "".join(f"{keyword}\n" for keyword in missing)
    return deck[: marker.end()] + insertion + deck[marker.end() :]


def _active_text(deck: str) -> str:
    return "\n".join(line.split("--", maxsplit=1)[0] for line in deck.splitlines())


def strip_restart_output(deck: str) -> str:
    """Remove active RPTRST records while preserving the rest of a text deck."""
    lines = deck.splitlines(keepends=True)
    output: list[str] = []
    skipping = False
    for line in lines:
        active = line.split("--", maxsplit=1)[0].strip()
        if not skipping and re.match(r"(?i)^RPTRST(?:\s|/|$)", active):
            skipping = "/" not in active
            continue
        if skipping:
            if "/" in active:
                skipping = False
            continue
        output.append(line)
    sanitized = "".join(output)
    sanitized = re.sub(
        r"(?ims)^[ \t]*RPTSOL[ \t]*$\s*(?=[^/]*\bRESTART\b)[^/]*/"
        r"[ \t]*(?:\r?\n[ \t]*/[ \t]*)?",
        "",
        sanitized,
    )
    assert_restart_disabled(sanitized)
    return sanitized


def assert_restart_disabled(deck: str) -> None:
    """Reject active schedule restart-output requests."""
    active = _active_text(deck)
    if re.search(r"(?im)^\s*RPTRST(?:\s|/|$)", active):
        raise RestartOutputEnabledError("active RPTRST request is forbidden")
    for match in re.finditer(r"(?ims)^\s*RPTSOL\s*$\s*(.*?)/", active):
        if re.search(r"(?i)\bRESTART\b", match.group(1)):
            raise RestartOutputEnabledError("RPTSOL RESTART request is forbidden")
    for match in re.finditer(r"(?ims)^\s*RPTSCHED\s*$\s*(.*?)/", active):
        if re.search(r"(?i)\bRESTART\b", match.group(1)):
            raise RestartOutputEnabledError("RPTSCHED RESTART request is forbidden")
