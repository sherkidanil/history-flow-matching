"""Storage-safe Eclipse deck transformations."""

from __future__ import annotations

import re


class RestartOutputEnabledError(ValueError):
    """Raised when a deck can request mass-run restart output."""


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
    assert_restart_disabled(sanitized)
    return sanitized


def assert_restart_disabled(deck: str) -> None:
    """Reject active schedule restart-output requests."""
    active = _active_text(deck)
    if re.search(r"(?im)^\s*RPTRST(?:\s|/|$)", active):
        raise RestartOutputEnabledError("active RPTRST request is forbidden")
    for match in re.finditer(r"(?ims)^\s*RPTSCHED\s*$\s*(.*?)/", active):
        if re.search(r"(?i)\bRESTART\b", match.group(1)):
            raise RestartOutputEnabledError("RPTSCHED RESTART request is forbidden")
