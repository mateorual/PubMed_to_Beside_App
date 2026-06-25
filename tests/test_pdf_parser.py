"""Tests for Gray Zone PDF parser text heuristics."""

from __future__ import annotations

from utils.pdf_parser import parse_case_text


def test_parse_case_text_extracts_sections() -> None:
    """Parser should extract case text, numbered questions, and expert blocks."""
    text = """
    A patient has locally advanced cancer after prior therapy.

    1. What systemic therapy is appropriate?
    2. What dose would you recommend?

    Expert Opinion
    Dr. Jane Doe, University Hospital. I recommend 50 Gy in 25 fractions with chemotherapy.
    """
    parsed = parse_case_text(text, "case_test")
    assert parsed["case_id"] == "case_test"
    assert len(parsed["questions"]) == 2
    assert parsed["expert_opinions"]
    assert "50 Gy" in parsed["expert_opinions"][0]["recommended_doses"]
