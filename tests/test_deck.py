from __future__ import annotations

import pytest

from fmgeo.forward.deck import (
    RestartOutputEnabledError,
    assert_restart_disabled,
    ensure_summary_keywords,
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
