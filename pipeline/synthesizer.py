"""Stage 5: synthesize an answer with PMID citations."""

from __future__ import annotations

import base64
import logging
import re
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import config

LOGGER = logging.getLogger(__name__)
PMID_PATTERN = re.compile(r"\[PMID:\s*(\d+)\]")

# Few-shot example injected into the messages array to enforce flowing prose format.
# PMIDs use non-numeric labels (FEWSHOT1 etc.) so they cannot match PMID_PATTERN
# and will never appear in extracted citations.
_FEW_SHOT_USER = """PATIENT CASE:
A 65-year-old woman with pT2N2M0 breast cancer (4/15 positive axillary nodes, clear margins >5 mm, no extranodal extension) underwent mastectomy. No significant comorbidities.

CLINICAL QUESTIONS — there are exactly 1 question below. Produce exactly 1 answer section.
1. Would you recommend post-mastectomy radiation therapy?

LITERATURE CONTEXT — answer exclusively from these papers:

PMID: FEWSHOT1
Authors: McGale et al.
Title: Post-mastectomy radiotherapy for node-positive breast cancer: EBCTCG meta-analysis
Year: 2014
Evidence: Meta-analysis of 1,133 women with 4+ positive axillary nodes. Post-mastectomy RT reduced 10-year locoregional recurrence from 32.1% to 13.0% and improved 20-year breast cancer mortality (RR 0.87, p=0.01). Benefit was consistent regardless of systemic therapy use."""

_FEW_SHOT_ASSISTANT = """**Question 1: Would you recommend post-mastectomy radiation therapy?**

**Bottom line:** We recommend post-mastectomy radiation therapy for this patient with 4 positive axillary nodes.

McGale et al. [PMID:FEWSHOT1] in a meta-analysis of 1,133 women with 4+ positive nodes demonstrated that post-mastectomy RT reduced 10-year locoregional recurrence from 32.1% to 13.0% and improved 20-year breast cancer mortality (RR 0.87, p=0.01), with benefit consistent across systemic therapy regimens — directly applicable to this patient's pN2 disease. This patient's clear margins (>5 mm) and absence of extranodal extension are favorable features but do not negate the strong indication driven by the nodal burden [PMID:FEWSHOT1]. The main caveat is that McGale et al. [PMID:FEWSHOT1] predates modern systemic therapies; whether the absolute benefit is preserved with contemporary regimens remains debated, and the decision should be reviewed in a multidisciplinary setting."""


def synthesize_answer(
    case_text: str,
    questions: list[str],
    papers: list[dict[str, Any]],
    trace_log: dict[str, Any] | None = None,
    max_papers: int = config.MAX_SYNTHESIS_PAPERS,
    provider: str = "openai",
    images: list[dict[str, Any]] | None = None,
    patient_profile: str = "",
) -> dict[str, Any]:
    """Generate the final answer and citation audit."""
    papers = papers[:max_papers]
    if config.is_provider_available(provider) and papers:
        try:
            answer_text = _synthesize_with_llm(case_text, questions, papers, provider, images, patient_profile)
            if provider == "openai" and _needs_reformat(answer_text):
                LOGGER.info("Old section format detected — running reformat pass")
                answer_text = _reformat_answer(answer_text, provider)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("LLM synthesis failed; using fallback answer: %s", exc)
            answer_text = _fallback_answer(questions, papers)
    else:
        answer_text = _fallback_answer(questions, papers)

    citations = extract_citations(answer_text, papers)
    fetched_pmids = {str(paper.get("pmid")) for paper in papers}
    hallucinated = sorted({item["pmid"] for item in citations if item["pmid"] not in fetched_pmids})
    return {
        "answer_text": answer_text,
        "citations": citations,
        "hallucinated_pmids": hallucinated,
        "papers_used": [_paper_summary(paper) for paper in papers],
        "trace_log": trace_log or {},
    }


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _synthesize_with_llm(
    case_text: str,
    questions: list[str],
    papers: list[dict[str, Any]],
    provider: str = "openai",
    images: list[dict[str, Any]] | None = None,
    patient_profile: str = "",
) -> str:
    """Call the LLM to synthesize a cited clinical answer."""
    is_ollama = provider == "ollama"
    client = config.get_llm_client(provider)

    # Local 8B models need a smaller context budget and a simpler prompt
    paper_cap = config.MAX_SYNTHESIS_PAPERS_OLLAMA if is_ollama else len(papers)
    token_budget = config.MAX_CONTEXT_TOKENS_OLLAMA if is_ollama else 100_000
    system_prompt = config.SYNTHESIS_SYSTEM_PROMPT_OLLAMA if is_ollama else config.SYNTHESIS_SYSTEM_PROMPT

    context = build_context(papers[:paper_cap], max_tokens=token_budget)
    n_q = len(questions)
    questions_block = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))

    profile_block = f"PATIENT PROFILE:\n{patient_profile}\n\n" if patient_profile else ""

    if is_ollama:
        # For small local models: put context first, questions last.
        # Recency bias means the model answers whatever appears closest to generation time.
        # With questions buried before a large context block, Llama ignores them and
        # summarizes papers instead.
        user_message = (
            f"LITERATURE CONTEXT (use ONLY these papers):\n\n{context}\n\n"
            f"{profile_block}"
            f"PATIENT CASE:\n{case_text}\n\n"
            f"Now answer EACH of the following {n_q} question{'s' if n_q != 1 else ''} "
            f"using the papers above. One section per question, in order:\n{questions_block}"
        )
    else:
        user_message = (
            f"{profile_block}"
            f"PATIENT CASE:\n{case_text}\n\n"
            f"CLINICAL QUESTIONS — there are exactly {n_q} question{'s' if n_q != 1 else ''} below. "
            f"Produce exactly {n_q} answer section{'s' if n_q != 1 else ''}. "
            f"Do NOT split any compound question into sub-questions:\n{questions_block}\n\n"
            f"LITERATURE CONTEXT — answer exclusively from these papers:\n\n{context}"
        )

    # Build multimodal content for OpenAI vision (GPT-4o); text-only for all other providers
    if provider == "openai" and images:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_message}]
        for img in images:
            ext = img["name"].rsplit(".", 1)[-1].lower()
            mime = "image/png" if ext == "png" else "image/jpeg"
            b64 = base64.b64encode(img["bytes"]).decode()
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
            })
        LOGGER.info("Vision synthesis: %d image(s) attached", len(images))
    else:
        user_content = user_message  # type: ignore[assignment]

    # For OpenAI, inject a few-shot example so GPT-4o mirrors the flowing prose format.
    # The example uses non-numeric PMIDs (FEWSHOT1) that cannot be extracted by PMID_PATTERN.
    if provider == "openai":
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _FEW_SHOT_USER},
            {"role": "assistant", "content": _FEW_SHOT_ASSISTANT},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    response = client.chat.completions.create(
        model=config.get_model_name(provider, "synthesis"),
        messages=messages,
    )
    return response.choices[0].message.content or ""


def build_context(papers: list[dict[str, Any]], max_tokens: int = 100_000) -> str:
    """Assemble a bounded literature context for synthesis."""
    chunks = []
    for paper in papers:
        body = str(paper.get("full_text") or paper.get("abstract") or "")
        if paper.get("full_text"):
            body = _truncate_full_text(body, target_words=3500)
        authors: list[str] = paper.get("authors") or []
        if authors:
            first_last = authors[0].split()[-1]  # last name of first author
            author_line = f"{first_last} et al." if len(authors) > 1 else first_last
        else:
            author_line = "Unknown"
        chunks.append(
            "\n".join(
                [
                    f"PMID: {paper.get('pmid', '')}",
                    f"Authors: {author_line}",
                    f"Title: {paper.get('title', '')}",
                    f"Year: {paper.get('year', '')}",
                    f"Evidence: {body}",
                ]
            )
        )
    context = "\n\n---\n\n".join(chunks)
    return _truncate_to_token_budget(context, max_tokens)


def extract_citations(answer_text: str, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract inline PMID citations, their answer sentences, and source metadata."""
    pmid_to_paper = {str(paper.get("pmid")): paper for paper in papers}
    citations: list[dict[str, Any]] = []
    sentences = re.split(r"(?<=[.!?])\s+", answer_text)
    for sentence in sentences:
        for pmid in PMID_PATTERN.findall(sentence):
            paper = pmid_to_paper.get(pmid, {})
            in_set = pmid in pmid_to_paper
            used_full = paper.get("used_full_text", False)
            evidence_body = str(paper.get("full_text") or paper.get("abstract", ""))
            citations.append(
                {
                    "pmid": pmid,
                    "title": paper.get("title", ""),
                    "year": str(paper.get("year", "")),
                    "sentence": sentence.strip(),
                    "evidence_quote": _best_evidence_quote(sentence, evidence_body),
                    "in_fetched_set": in_set,
                    "used_full_text": used_full,
                }
            )
    return citations


# Matches section labels like "Primary recommendation —", "**Supporting evidence —**", etc.
_SECTION_LABEL_RE = re.compile(
    r"^\*{0,2}(Primary recommendation|Supporting evidence|Evidence quality"
    r"|Caveats(?:\s+and\s+uncertainty)?)\*{0,2}\s*[—–\-]+\s*",
    re.IGNORECASE | re.MULTILINE,
)


def _needs_reformat(text: str) -> bool:
    """Return True if the answer contains the old labelled-section structure."""
    return bool(_SECTION_LABEL_RE.search(text))


def _reformat_answer(answer_text: str, provider: str) -> str:  # noqa: ARG001
    """
    Deterministic reformatter — no LLM call.
    Strips 4-section labels and merges content into Bottom line + flowing paragraph
    per question block. Works regardless of whether headers have ** bold markers.
    """
    # Split on "Question N" boundary — handles both plain and **bold** headers
    q_blocks = re.split(r"(?=\*{0,2}Question\s+\d+\b)", answer_text, flags=re.IGNORECASE)
    out_blocks: list[str] = []

    for block in q_blocks:
        if not block.strip():
            continue

        # First line is the question header (bold or plain)
        parts = block.split("\n", 1)
        raw_header = parts[0].strip()
        body = parts[1] if len(parts) > 1 else ""

        # Verify it's actually a question header; otherwise pass through unchanged
        if not re.match(r"\*{0,2}Question\s+\d+\b", raw_header, re.IGNORECASE):
            out_blocks.append(block.strip())
            continue

        # Normalise to bold (strip existing * and re-apply)
        header = f"**{raw_header.strip('*').strip()}**"

        # Parse body paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        bottom_line = ""
        prose_parts: list[str] = []

        for para in paragraphs:
            lm = _SECTION_LABEL_RE.match(para)
            if lm:
                label = lm.group(1).lower()
                content = para[lm.end():].strip()
                if "primary recommendation" in label:
                    # First sentence → Bottom line; rest → prose
                    sentences = re.split(r"(?<=[.!?])\s+", content, maxsplit=1)
                    bottom_line = sentences[0].strip()
                    if len(sentences) > 1 and sentences[1].strip():
                        prose_parts.append(sentences[1].strip())
                else:
                    if content:
                        prose_parts.append(content)
            else:
                if para:
                    prose_parts.append(para)

        prose = " ".join(prose_parts)

        if bottom_line:
            out_blocks.append(f"{header}\n\n**Bottom line:** {bottom_line}\n\n{prose}")
        else:
            # No Primary recommendation — strip labels and join remaining content
            cleaned_parts: list[str] = []
            for para in paragraphs:
                lm2 = _SECTION_LABEL_RE.match(para)
                cleaned_parts.append(para[lm2.end():].strip() if lm2 else para)
            out_blocks.append(f"{header}\n\n{' '.join(cleaned_parts)}")

    result = "\n\n".join(out_blocks)
    LOGGER.info("Deterministic reformat complete (%d question blocks)", len(out_blocks))
    return result


def _fallback_answer(questions: list[str], papers: list[dict[str, Any]]) -> str:
    """Create a transparent non-LLM answer from retrieved abstracts."""
    if not papers:
        return "No relevant papers with abstracts were available, so no evidence-based synthesis could be generated."
    top_pmids = ", ".join(f"[PMID:{paper.get('pmid')}]" for paper in papers[:5] if paper.get("pmid"))
    sections = []
    for idx, question in enumerate(questions, start=1):
        sections.append(
            f"Question {idx}: {question}\n"
            f"The available MVP fallback cannot make a clinical recommendation without an LLM synthesis pass. "
            f"Review the retrieved evidence directly: {top_pmids}."
        )
    return "\n\n".join(sections)


def _best_evidence_quote(sentence: str, abstract: str) -> str:
    """Select a short abstract sentence with the highest lexical overlap."""
    if not abstract:
        return ""
    terms = {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", sentence)}
    candidates = re.split(r"(?<=[.!?])\s+", abstract)
    best = max(candidates, key=lambda item: len(terms & {t.lower() for t in re.findall(r'[A-Za-z][A-Za-z0-9-]{3,}', item)}), default="")
    return best[:500]


def _paper_summary(paper: dict[str, Any]) -> dict[str, Any]:
    """Return paper metadata plus the evidence text actually sent to synthesis."""
    used_full = paper.get("used_full_text", False)
    summary: dict[str, Any] = {
        "pmid": paper.get("pmid"),
        "pmcid": paper.get("pmcid"),
        "title": paper.get("title"),
        "year": paper.get("year"),
        "authors": paper.get("authors", []),
        "abstract_score": paper.get("abstract_score"),
        "used_full_text": used_full,
        "abstract": paper.get("abstract", ""),
    }
    if used_full:
        # Include the truncated full text that was sent to the LLM
        summary["full_text"] = _truncate_full_text(str(paper.get("full_text", "")), target_words=3500)
    return summary


def _truncate_full_text(text: str, target_words: int = 2000) -> str:
    """Truncate full text keeping intro, middle (Results), and conclusion."""
    words = text.split()
    if len(words) <= target_words:
        return text
    n = len(words)
    # Allocate budget: 25% intro context, 50% middle (results/methods), 25% conclusion
    first = target_words // 4
    mid_size = target_words // 2
    last = target_words - first - mid_size
    mid_start = n // 3  # results typically start around the 1/3 mark
    return " ".join(
        words[:first]
        + ["[...truncated...]"]
        + words[mid_start : mid_start + mid_size]
        + ["[...truncated...]"]
        + words[-last:]
    )


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Truncate text approximately to a token budget."""
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoding.decode(tokens[:max_tokens])
    except Exception:  # noqa: BLE001
        return " ".join(text.split()[: max_tokens * 3 // 4])
