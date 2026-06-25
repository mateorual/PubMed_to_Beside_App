"""Stage 3: screen abstracts for relevance to the clinical case."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import config

LOGGER = logging.getLogger(__name__)


def screen_abstracts(
    papers: list[dict[str, Any]],
    case_text: str,
    questions: list[str],
    threshold: float = config.ABSTRACT_THRESHOLD,
    batch_size: int = 5,
    provider: str = "openai",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank abstracts by relevance and return passed papers plus trace rows."""
    if not papers:
        return [], []
    if config.is_provider_available(provider):
        try:
            trace = _screen_with_llm(papers, case_text, questions, threshold, batch_size, provider)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("LLM screening failed; using lexical fallback: %s", exc)
            trace = _screen_with_lexical_overlap(papers, case_text, questions, threshold)
    else:
        trace = _screen_with_lexical_overlap(papers, case_text, questions, threshold)

    score_by_pmid = {row["pmid"]: row for row in trace}
    ranked = sorted(
        ({**paper, "abstract_score": score_by_pmid.get(paper.get("pmid"), {}).get("score", 0.0)} for paper in papers),
        key=lambda item: item.get("abstract_score", 0.0),
        reverse=True,
    )
    passed = [paper for paper in ranked if score_by_pmid.get(paper.get("pmid"), {}).get("passed")]
    return passed, trace


def _screen_with_llm(
    papers: list[dict[str, Any]],
    case_text: str,
    questions: list[str],
    threshold: float,
    batch_size: int,
    provider: str = "openai",
) -> list[dict[str, Any]]:
    """Score abstracts with the LLM in batches."""
    trace: list[dict[str, Any]] = []
    for start in range(0, len(papers), batch_size):
        batch = papers[start : start + batch_size]
        scores = _score_batch(case_text, questions, batch, provider)
        score_map = {str(item.get("pmid")): item for item in scores}
        for paper in batch:
            pmid = str(paper.get("pmid", ""))
            item = score_map.get(pmid, {})
            score = float(item.get("score", 0))
            trace.append(
                {
                    "pmid": pmid,
                    "title": paper.get("title", ""),
                    "score": score,
                    "passed": score >= threshold,
                    "rationale": item.get("rationale", ""),
                }
            )
        time.sleep(0.5)
    return trace


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _score_batch(
    case_text: str,
    questions: list[str],
    papers: list[dict[str, Any]],
    provider: str = "openai",
) -> list[dict[str, Any]]:
    """Call the LLM to score one abstract batch."""
    client = config.get_llm_client(provider)
    compact_papers = [
        {
            "pmid": paper.get("pmid"),
            "title": paper.get("title"),
            "abstract": str(paper.get("abstract", ""))[:2500],
        }
        for paper in papers
    ]
    create_kwargs: dict[str, Any] = {
        "model": config.get_model_name(provider, "screening"),
        "messages": [
            {"role": "system", "content": config.ABSTRACT_SCREENING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"patient_case": case_text, "clinical_questions": questions, "papers": compact_papers},
                    ensure_ascii=False,
                ),
            },
        ],
    }
    # Ollama's JSON mode is unreliable for Llama 3.1 8B — omit it and parse best-effort
    if provider != "ollama":
        create_kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**create_kwargs)
    raw = response.choices[0].message.content or "{}"
    # Best-effort JSON extraction: find the first {...} block in case the model wraps output
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    payload = json.loads(json_match.group(0) if json_match else "{}") if json_match else {}
    return list(payload.get("scores", []))


def _screen_with_lexical_overlap(
    papers: list[dict[str, Any]],
    case_text: str,
    questions: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    """Score abstracts with a deterministic lexical-overlap fallback."""
    terms = _content_terms(f"{case_text} {' '.join(questions)}")
    trace = []
    for paper in papers:
        abstract_terms = _content_terms(f"{paper.get('title', '')} {paper.get('abstract', '')}")
        overlap = len(terms & abstract_terms)
        score = min(10.0, round((overlap / max(len(terms), 1)) * 20, 2))
        trace.append(
            {
                "pmid": paper.get("pmid", ""),
                "title": paper.get("title", ""),
                "score": score,
                "passed": score >= threshold,
                "rationale": "Lexical-overlap fallback score.",
            }
        )
    return trace


def _content_terms(text: str) -> set[str]:
    """Extract simple content terms from text."""
    import re

    stop = {"patient", "treatment", "therapy", "question", "clinical", "would", "with", "from", "this"}
    return {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text) if token.lower() not in stop}
