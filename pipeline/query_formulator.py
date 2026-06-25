"""Stage 1: formulate a PubMed query from a case and clinical questions."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import config

LOGGER = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structured output from query formulation."""

    query_string: str        # primary query (first in query_strings)
    query_strings: list[str] # all queries, most specific first
    keywords: list[str]
    method: str
    patient_profile: str = ""  # compact structured summary of patient features for synthesis


def formulate_query(case_text: str, questions: list[str], provider: str = "openai") -> QueryResult:
    """Create 1–3 complementary PubMed queries and a keyword trace."""
    if config.is_provider_available(provider):
        try:
            return _formulate_query_with_llm(case_text, questions, provider)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("LLM query formulation failed; using fallback: %s", exc)
    return _fallback_query(case_text, questions)


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _formulate_query_with_llm(case_text: str, questions: list[str], provider: str = "openai") -> QueryResult:
    """Call the LLM to formulate 1–3 complementary literature queries."""
    client = config.get_llm_client(provider)
    response = client.chat.completions.create(
        model=config.get_model_name(provider, "query"),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": config.QUERY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"patient_case": case_text, "clinical_questions": questions},
                    ensure_ascii=False,
                ),
            },
        ],
    )
    payload = json.loads(response.choices[0].message.content or "{}")

    # Accept either "query_strings" (list) or legacy "query_string" (single string)
    raw_queries: list[str] = payload.get("query_strings") or []
    if not raw_queries and payload.get("query_string"):
        raw_queries = [str(payload["query_string"])]

    query_strings = [str(q).strip() for q in raw_queries if str(q).strip()]
    if not query_strings:
        raise ValueError("LLM returned no valid query strings")

    keywords = [str(k).strip() for k in payload.get("keywords", []) if str(k).strip()]
    patient_profile = str(payload.get("patient_profile", "")).strip()
    LOGGER.info("LLM formulated %d queries; primary: %s", len(query_strings), query_strings[0])
    return QueryResult(
        query_string=query_strings[0],
        query_strings=query_strings,
        keywords=keywords,
        method="llm",
        patient_profile=patient_profile,
    )


def _fallback_query(case_text: str, questions: list[str]) -> QueryResult:
    """Build two simple queries from frequent clinical tokens."""
    text = f"{case_text} {' '.join(questions)}".lower()
    stopwords = {
        "patient", "with", "would", "what", "should", "recommend", "therapy",
        "treatment", "because", "there", "about", "which", "year", "years",
        "case", "clinical",
    }
    tokens = re.findall(r"[a-z][a-z0-9-]{3,}", text)
    counts = Counter(token for token in tokens if token not in stopwords)
    keywords = [term for term, _ in counts.most_common(8)]

    # Specific query: top 5 terms with AND
    primary = " AND ".join(f'"{term}"' if "-" in term else term for term in keywords[:5])
    # Broader query: top 3 terms (more recall)
    broader = " AND ".join(f'"{term}"' if "-" in term else term for term in keywords[:3])

    query_strings = [q for q in [primary, broader] if q]
    if not query_strings:
        query_strings = ["radiation oncology"]

    return QueryResult(
        query_string=query_strings[0],
        query_strings=query_strings,
        keywords=keywords,
        method="fallback",
    )
