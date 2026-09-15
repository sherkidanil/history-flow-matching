from __future__ import annotations

import pytest

from fmgeo.forward.deck import (
    RestartOutputEnabledError,
    assert_restart_disabled,
    ensure_summary_keywords,
    render_punq_opm_compatibility,
    strip_restart_output,
)


def test_strip_restart_output_removes_active_rptrst_block() -> None:
    deck = """
SCHEDULE
-- RPTRST
-- BASIC=2 /
RPTRST
  BASIC=2 /
WELSPECS
  'P1' 'G' 1 1 1* 'OIL' /
/
"""

    sanitized = strip_restart_output(deck)

    assert "BASIC=2" not in sanitized.replace("-- BASIC=2", "")
    assert_restart_disabled(sanitized)
    assert "WELSPECS" in sanitized


def test_restart_assertion_rejects_unsanitized_deck() -> None:
    with pytest.raises(RestartOutputEnabledError, match="RPTRST"):
        assert_restart_disabled("SCHEDULE\nRPTRST\n BASIC=2 /\n")


def test_ensure_summary_keywords_adds_only_missing_requests() -> None:
    deck = "RUNSPEC\nSUMMARY\nFOPR\nSCHEDULE\n"

    rendered = ensure_summary_keywords(deck, ["FOPT", "FOPR"])

    assert rendered.count("FOPR") == 1
    assert "SUMMARY\nFOPT\nFOPR\n" in rendered


def test_punq_compatibility_adds_required_runspec_dimensions_once() -> None:
    deck = (
        "RUNSPEC\nDIMENS\n 19 28 5 /\nGRID\nSOLUTION\n"
        "AQUCT\n 1 2 3 4 5 6 7 8 9 1 1 /\n"
        "AQUANCON\n 1 1 1 1 1 1 1 'I+' 1.0 /\n/\n"
        "SUMMARY\nSEPARATE\nFOPT\n"
        "SCHEDULE\nWCONPROD\n'PRO*' 'SHUT' 6* 120.0 /\n/\n"
        "WCUTBACK\n'PRO*' 1* 200.0 2* 0.75 'OIL' 120.0 /\n/\n"
    )

    first = render_punq_opm_compatibility(deck)
    second = render_punq_opm_compatibility(first)

    assert first == second
    assert "TABDIMS\n  1 1 50 50 /\n" in first
    assert "AQUDIMS\n  2 100 1 36 2 100 /\n" in first
    assert first.index("AQUDIMS") < first.index("GRID")
    assert "UNIFOUT" in first
    assert "\nSEPARATE\n" not in first
    assert "/\n\nAQUANCON" in first
    assert "'PRO*' 'SHUT' 'ORAT' 100.0 4* 120.0 /" in first
    assert "FMGEO: WCUTBACK removed" in first
    assert "\nWCUTBACK\n" not in first
