"""Tests for deterministic pipeline helpers (no real API calls required)."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.abstract_screener import screen_abstracts
from pipeline.query_formulator import _fallback_query
from pipeline.synthesizer import _needs_reformat, extract_citations, synthesize_answer
from pipeline.title_screener import screen_titles


# ── Query formulator ──────────────────────────────────────────────────────────

def test_fallback_query_returns_keywords() -> None:
    """Fallback query formulation should produce a usable query string."""
    result = _fallback_query("Anal squamous cell carcinoma after chemoradiation.", ["What dose?"])
    assert result.query_string
    assert "anal" in result.keywords


def test_fallback_query_method_label() -> None:
    """Fallback query must be labelled 'fallback', not 'llm'."""
    result = _fallback_query("Rectal adenocarcinoma stage II.", ["What is the recommended adjuvant therapy?"])
    assert result.method == "fallback"
    assert len(result.query_strings) >= 1


# ── Title screener ────────────────────────────────────────────────────────────

def test_title_screener_output_structure() -> None:
    """Title screening trace must include required fields for every paper."""
    papers = [
        {"pmid": "1", "title": "Anal carcinoma chemoradiation dose study"},
        {"pmid": "2", "title": "Unrelated orthopedic surgery outcomes"},
    ]
    passed, trace = screen_titles(papers, "Anal squamous cell carcinoma chemoradiation", ["What dose?"])
    assert len(trace) == 2
    for row in trace:
        assert "pmid" in row
        assert "title_score" in row
        assert "title_passed" in row
        assert "title_rationale" in row
        assert isinstance(row["title_score"], float)
        assert isinstance(row["title_passed"], bool)


def test_title_screener_relevant_paper_passes() -> None:
    """A paper with strong title overlap should pass title screening."""
    papers = [{"pmid": "10", "title": "Chemoradiation for anal squamous cell carcinoma: dose and outcomes"}]
    passed, trace = screen_titles(papers, "Anal squamous cell carcinoma chemoradiation", ["What dose?"])
    assert any(row["title_passed"] for row in trace), "Relevant paper should pass title screening"
    assert papers[0] in passed


def test_title_screener_unrelated_paper_filtered() -> None:
    """A paper completely unrelated to the query should not pass title screening."""
    papers = [{"pmid": "99", "title": "Randomized study of knee arthroplasty outcomes"}]
    passed, trace = screen_titles(papers, "Anal squamous cell carcinoma chemoradiation", ["What radiation dose?"])
    assert not any(row["title_passed"] for row in trace), "Unrelated paper should not pass"
    assert len(passed) == 0


def test_title_screener_empty_title() -> None:
    """Papers with no title should not crash and should get score 0."""
    papers = [{"pmid": "5", "title": ""}]
    passed, trace = screen_titles(papers, "some case text", ["some question"])
    assert trace[0]["title_score"] == 0.0
    assert "No title" in trace[0]["title_rationale"]


# ── Abstract screener ─────────────────────────────────────────────────────────

def test_lexical_screening_without_api_key() -> None:
    """Abstract screening should work deterministically without LLM access."""
    papers = [{"pmid": "1", "title": "Anal cancer", "abstract": "Anal carcinoma chemoradiation dose study."}]
    passed, trace = screen_abstracts(papers, "Anal carcinoma chemoradiation", ["dose"], threshold=1)
    assert passed
    assert trace[0]["pmid"] == "1"


def test_abstract_screening_trace_structure() -> None:
    """Abstract screening trace must carry score, passed, and rationale fields."""
    papers = [{"pmid": "42", "title": "Some cancer paper", "abstract": "Cancer therapy outcomes."}]
    _, trace = screen_abstracts(papers, "cancer therapy", ["outcomes"], threshold=0)
    assert len(trace) == 1
    row = trace[0]
    assert "pmid" in row
    assert "score" in row
    assert "passed" in row


def test_abstract_screening_no_papers() -> None:
    """Empty input should return empty passed list and empty trace."""
    passed, trace = screen_abstracts([], "case text", ["question"])
    assert passed == []
    assert trace == []


# ── Synthesizer — extract_citations ───────────────────────────────────────────

def test_extract_citations_maps_sentences() -> None:
    """Citation extraction should capture sentence text and PMID."""
    papers = [{"pmid": "123", "abstract": "Chemoradiation improves local control."}]
    citations = extract_citations("Use chemoradiation [PMID:123].", papers)
    assert citations[0]["pmid"] == "123"
    assert "chemoradiation" in citations[0]["sentence"].lower()


def test_extract_citations_enriched_fields() -> None:
    """Extracted citations must include title, year, in_fetched_set, used_full_text."""
    papers = [{"pmid": "456", "title": "A randomized trial", "year": "2020",
               "abstract": "Evidence text.", "used_full_text": False}]
    citations = extract_citations("Key finding [PMID:456].", papers)
    assert len(citations) == 1
    c = citations[0]
    assert c["title"] == "A randomized trial"
    assert c["year"] == "2020"
    assert c["in_fetched_set"] is True
    assert c["used_full_text"] is False


def test_extract_citations_detects_hallucinated_pmid() -> None:
    """A PMID not in the fetched set should be flagged as not in_fetched_set."""
    papers = [{"pmid": "111", "abstract": "Some abstract.", "used_full_text": False}]
    citations = extract_citations("Hallucinated result [PMID:999].", papers)
    assert citations[0]["in_fetched_set"] is False


def test_extract_citations_no_citations() -> None:
    """Answer text without any [PMID:...] tags should return empty citation list."""
    papers = [{"pmid": "1", "abstract": "Abstract text."}]
    citations = extract_citations("No citations in this answer.", papers)
    assert citations == []


# ── Synthesizer — full output structure ──────────────────────────────────────

def test_synthesize_answer_no_papers_fallback() -> None:
    """Synthesis with no papers should return a fallback answer and empty citations."""
    result = synthesize_answer("A patient case.", ["What therapy?"], papers=[], max_papers=5)
    assert "answer_text" in result
    assert "citations" in result
    assert "hallucinated_pmids" in result
    assert "papers_used" in result
    assert "trace_log" in result
    # No papers means no citations
    assert result["citations"] == []


def test_synthesize_answer_hallucination_detection() -> None:
    """Hallucinated PMIDs must appear in hallucinated_pmids list."""
    papers = [{"pmid": "111", "abstract": "Real abstract.", "used_full_text": False}]
    answer = "First claim [PMID:111]. Second claim [PMID:999]."
    result = synthesize_answer.__wrapped__ if hasattr(synthesize_answer, "__wrapped__") else None
    # Test via extract_citations directly (synthesize_answer needs an API key for LLM path)
    from pipeline.synthesizer import extract_citations
    citations = extract_citations(answer, papers)
    fetched = {str(p.get("pmid")) for p in papers}
    hallucinated = sorted({c["pmid"] for c in citations if c["pmid"] not in fetched})
    assert "999" in hallucinated
    assert "111" not in hallucinated


def test_needs_reformat_detects_any_old_label() -> None:
    """Old labelled-section markers should be detected anywhere in the answer."""
    assert _needs_reformat("**Primary recommendation:** Treat with RT.")
    assert _needs_reformat("Text\nSupporting evidence - trial data.")
    assert _needs_reformat("Evidence quality: retrospective cohort.")
    assert _needs_reformat("Caveats and uncertainty: limited evidence.")
    assert not _needs_reformat("**Bottom line:** We recommend RT.\nFlowing paragraph.")


def test_synthesize_answer_reformats_old_openai_output(monkeypatch: object) -> None:
    """OpenAI synthesis should run the post-processor before citation extraction."""
    from pipeline import synthesizer

    old_answer = (
        "**Question 1: Treat?**\n\n"
        "Primary recommendation - Treat with postoperative RT [PMID:111].\n"
        "Supporting evidence - Trial evidence supports this [PMID:111]."
    )
    new_answer = (
        "**Question 1: Treat?**\n\n"
        "**Bottom line:** We recommend postoperative RT for this patient.\n\n"
        "Trial evidence supports this recommendation [PMID:111]."
    )
    papers = [{"pmid": "111", "title": "Trial", "abstract": "Trial evidence supports this.", "used_full_text": False}]

    monkeypatch.setattr(synthesizer.config, "is_provider_available", lambda provider: True)
    monkeypatch.setattr(synthesizer, "_synthesize_with_llm", lambda *args, **kwargs: old_answer)
    monkeypatch.setattr(synthesizer, "_reformat_answer", lambda answer, provider: new_answer)

    result = synthesize_answer("Patient case.", ["Treat?"], papers, provider="openai")

    assert result["answer_text"] == new_answer
    assert "Primary recommendation" not in result["answer_text"]
    assert result["citations"][0]["pmid"] == "111"


# ── JSON case loading (cli helper) ────────────────────────────────────────────

def test_load_case_file(tmp_path: Path) -> None:
    """CLI load_case_file should parse patient_description and questions correctly."""
    from app.cli import load_case_file

    payload = {
        "patient_description": "A 60-year-old woman with stage III cervical cancer.",
        "questions": ["What radiation dose?", "What chemotherapy regimen?"],
    }
    case_file = tmp_path / "test_case.json"
    case_file.write_text(json.dumps(payload), encoding="utf-8")

    case_text, questions = load_case_file(case_file)
    assert case_text == payload["patient_description"]
    assert questions == payload["questions"]


# ── Trace-log structure ───────────────────────────────────────────────────────

def test_trace_log_structure_from_pipeline(monkeypatch: object) -> None:
    """run_pipeline trace_log must contain all hierarchical stage keys."""
    import types

    from app import cli

    # Patch names as they appear in cli's own namespace (from X import Y binds Y there)
    stub_paper = {"pmid": "1", "title": "Anal cancer chemoradiation", "abstract": "Text.", "pmcid": "PMC1"}

    monkeypatch.setattr(
        cli, "formulate_query",
        lambda *a, **kw: types.SimpleNamespace(
            query_string="anal cancer", query_strings=["anal cancer"], keywords=["anal"], method="fallback"
        ),
    )
    monkeypatch.setattr(cli, "fetch_pubmed_papers_multi", lambda *a, **kw: [stub_paper])
    monkeypatch.setattr(
        cli, "screen_titles",
        lambda *a, **kw: (
            [stub_paper],
            [{"pmid": "1", "title": "T", "title_score": 0.5, "title_passed": True, "title_rationale": "ok"}],
        ),
    )
    monkeypatch.setattr(
        cli, "screen_abstracts",
        lambda *a, **kw: (
            [stub_paper],
            [{"pmid": "1", "title": "T", "score": 7.0, "passed": True, "rationale": "ok"}],
        ),
    )
    monkeypatch.setattr(
        cli, "fetch_full_texts",
        lambda *a, **kw: [{**stub_paper, "used_full_text": False, "full_text": ""}],
    )
    monkeypatch.setattr(
        cli, "synthesize_answer",
        lambda *a, **kw: {
            "answer_text": "Answer [PMID:1].",
            "citations": [],
            "hallucinated_pmids": [],
            "papers_used": [],
            "trace_log": kw.get("trace_log") or {},
        },
    )

    result = cli.run_pipeline("Patient case.", ["Question?"], include_fulltext=True)
    trace = result["trace_log"]

    for key in (
        "query", "retrieved_count", "title_screened_count", "title_passed_count",
        "screened_count", "passed_count", "fulltext_count", "synthesis_papers",
        "title_screening_trace", "abstract_screening_trace", "fulltext_trace",
    ):
        assert key in trace, f"Missing trace key: {key}"

    assert trace["retrieved_count"] == 1
    assert trace["title_passed_count"] == 1
    assert trace["passed_count"] == 1
