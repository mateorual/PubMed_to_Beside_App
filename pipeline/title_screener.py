"""Stage 2b: title-level relevance screening before abstract scoring."""

from __future__ import annotations

import re
from typing import Any


def screen_titles(
    papers: list[dict[str, Any]],
    case_text: str,
    questions: list[str],
    threshold: float = 0.12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Screen papers by title lexical overlap; return (passed_papers, trace).

    Threshold is a fraction of query terms that must appear in the title.
    A low default (0.12) keeps recall high — abstract scoring is the strict gate.
    """
    terms = _content_terms(f"{case_text} {' '.join(questions)}")
    trace: list[dict[str, Any]] = []
    for paper in papers:
        title = paper.get("title", "")
        title_terms = _content_terms(title)
        if not title_terms:
            score = 0.0
            rationale = "No title text available."
        else:
            overlap = len(terms & title_terms)
            score = round(overlap / max(len(terms), 1), 4)
            rationale = f"{overlap} of {len(terms)} query terms matched in title."
        passed = score >= threshold
        trace.append(
            {
                "pmid": paper.get("pmid", ""),
                "title": title,
                "title_score": score,
                "title_passed": passed,
                "title_rationale": rationale,
            }
        )
    passed_pmids = {row["pmid"] for row in trace if row["title_passed"]}
    passed_papers = [p for p in papers if p.get("pmid", "") in passed_pmids]
    return passed_papers, trace


def _content_terms(text: str) -> set[str]:
    """Extract lowercased content tokens (length >= 4, stop-words removed)."""
    stop = {
        "patient", "treatment", "therapy", "question", "clinical", "would",
        "with", "from", "this", "that", "their", "have", "been", "were",
        "case", "after", "before", "during", "versus", "compared", "using",
        "after", "about", "what", "which", "does", "when", "will", "should",
    }
    return {
        t.lower()
        for t in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text)
        if t.lower() not in stop
    }
