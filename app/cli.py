"""Command-line interface for the PubMed to Bedside MVP."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from pipeline.abstract_screener import screen_abstracts
from pipeline.fulltext_fetcher import fetch_full_texts
from pipeline.pubmed_fetcher import fetch_pubmed_papers_multi
from pipeline.query_formulator import formulate_query
from pipeline.synthesizer import synthesize_answer
from pipeline.title_screener import screen_titles

LOGGER = logging.getLogger(__name__)


def run_pipeline(
    case_text: str,
    questions: list[str],
    max_results: int = config.MAX_PUBMED_RESULTS,
    include_fulltext: bool = True,
    max_synthesis_papers: int = config.MAX_SYNTHESIS_PAPERS,
    provider: str = "openai",
    images: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the full pipeline: query -> retrieve -> title screen -> abstract screen -> full text -> synthesize."""
    # Query formulation and screening always use OpenAI when available — reliable JSON output
    # and MeSH knowledge that small local models lack. Provider is used only for synthesis.
    retrieval_provider = "openai" if config.OPENAI_API_KEY else provider

    query = formulate_query(case_text, questions, provider=retrieval_provider)
    LOGGER.info("Query method: %s | %d queries | retrieval: %s | synthesis: %s",
                query.method, len(query.query_strings), retrieval_provider, provider)
    for i, qs in enumerate(query.query_strings, 1):
        LOGGER.info("  Query %d: %s", i, qs)

    papers = fetch_pubmed_papers_multi(query.query_strings, max_results=max_results)
    LOGGER.info("Retrieved %d unique papers with abstracts", len(papers))

    # Title screening is informational only — all papers proceed to abstract scoring
    title_passed, title_trace = screen_titles(papers, case_text, questions)
    LOGGER.info("Title scores annotated for %d papers", len(papers))

    passed, abstract_trace = screen_abstracts(papers, case_text, questions, provider=retrieval_provider)
    LOGGER.info("Abstract screening: %d of %d passed", len(passed), len(papers))

    synthesis_candidates = passed[:max_synthesis_papers]
    if include_fulltext:
        evidence_papers = fetch_full_texts(synthesis_candidates, max_papers=max_synthesis_papers)
    else:
        evidence_papers = [dict(p, used_full_text=False, full_text="") for p in synthesis_candidates]
    fulltext_count = sum(1 for p in evidence_papers if p.get("used_full_text"))
    LOGGER.info("Full text fetched for %d papers", fulltext_count)

    fulltext_trace = [
        {
            "pmid": p.get("pmid", ""),
            "title": p.get("title", ""),
            "year": p.get("year", ""),
            "pmcid": p.get("pmcid", ""),
            "full_text_available": bool(p.get("pmcid")),
            "used_full_text": p.get("used_full_text", False),
        }
        for p in evidence_papers
    ]

    trace_log = {
        "query": {
            "primary_query": query.query_string,
            "all_queries": query.query_strings,
            "keywords": query.keywords,
            "method": query.method,
        },
        "retrieved_count": len(papers),
        "title_screened_count": len(title_trace),
        "title_passed_count": sum(1 for r in title_trace if r["title_passed"]),
        "screened_count": len(abstract_trace),
        "passed_count": len(passed),
        "fulltext_count": fulltext_count,
        "synthesis_papers": min(len(evidence_papers), max_synthesis_papers),
        "title_screening_trace": title_trace,
        "abstract_screening_trace": abstract_trace,
        "fulltext_trace": fulltext_trace,
        # backward-compat alias used by older consumers
        "screening_trace": abstract_trace,
    }
    result = synthesize_answer(
        case_text, questions, evidence_papers,
        trace_log=trace_log, max_papers=max_synthesis_papers, provider=provider, images=images,
        patient_profile=getattr(query, "patient_profile", ""),
    )
    result["case_text"] = case_text
    result["questions"] = questions
    return result


def load_case_file(path: Path) -> tuple[str, list[str]]:
    """Load patient text and questions from a parsed case JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("patient_description", ""), list(data.get("questions", []))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="Raw patient case text.")
    parser.add_argument("--questions", nargs="+", help="One or more clinical questions.")
    parser.add_argument("--case_file", type=Path, help="Path to parsed Gray Zone case JSON.")
    parser.add_argument("--max_results", type=int, default=config.MAX_PUBMED_RESULTS)
    parser.add_argument("--output", type=Path, default=config.EVALUATION_RESULTS_DIR / "result.json")
    parser.add_argument("--no_fulltext", action="store_true", help="Skip PMC full-text fetching.")
    return parser


def main() -> None:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args()
    config.configure_logging()
    config.ensure_directories()

    if args.case_file:
        case_text, questions = load_case_file(args.case_file)
    else:
        case_text = args.case or ""
        questions = args.questions or []
    if not case_text or not questions:
        parser.error("Provide --case and --questions, or --case_file with patient_description and questions.")

    result = run_pipeline(
        case_text=case_text,
        questions=questions,
        max_results=args.max_results,
        include_fulltext=not args.no_fulltext,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    trace = result["trace_log"]
    query_lines = [f"  [{i+1}] {q}" for i, q in enumerate(trace["query"]["all_queries"])]
    print(
        "\n".join(
            [
                f"Queries ({trace['query']['method']}):",
                *query_lines,
                f"Retrieved:          {trace['retrieved_count']}",
                f"Title passed:       {trace['title_passed_count']}",
                f"Abstract passed:    {trace['passed_count']}",
                f"Full text fetched:  {trace['fulltext_count']}",
                f"Sent to synthesis:  {trace['synthesis_papers']}",
                "",
                result["answer_text"],
                "",
                f"Saved: {args.output}",
            ]
        )
    )


if __name__ == "__main__":
    main()
