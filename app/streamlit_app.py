"""PubMed to Bedside — clinical literature assistant."""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMG_DIR = PROJECT_ROOT / "img"

import streamlit as st

import config
from pipeline.abstract_screener import screen_abstracts
from pipeline.fulltext_fetcher import fetch_full_texts
from pipeline.pubmed_fetcher import fetch_pubmed_papers_multi
from pipeline.query_formulator import formulate_query
from pipeline.synthesizer import synthesize_answer
from pipeline.title_screener import screen_titles

PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

# ── Utilities ─────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normalize Unicode whitespace and remove common PDF extraction artifacts."""
    text = unicodedata.normalize("NFKC", text)
    # Replace non-standard space variants with regular space
    text = re.sub(r"[  -​  　]", " ", text)
    # Remove PDF (cid:N) artifacts
    text = re.sub(r"\(cid:\d+\)", " ", text)
    # Collapse multiple spaces on the same line (preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


def parse_questions(raw: str) -> list[str]:
    """Parse numbered questions ('1. Q', '1) Q', '1.Q') or plain one-per-line."""
    text = raw.strip()
    if not text:
        return []
    # Allow optional whitespace after the punctuation so '3.Would' also matches
    if re.search(r"(?m)^\d+[.)]\s*\w", text):
        parts = re.split(r"(?m)(?=^\d+[.)]\s*\w)", text)
        questions = []
        for part in parts:
            q = re.sub(r"^\d+[.)]\s*", "", part.strip()).strip()
            if q:
                questions.append(q)
        return questions
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def linkify_pmids(text: str) -> str:
    """Replace [PMID:...] tags with clickable PubMed hyperlinks.

    Handles both single citations [PMID:12345678] and semicolon-separated groups
    [PMID:12345678; PMID:87654321], converting each PMID into its own link.
    """
    def _replace(m: re.Match) -> str:
        pmids = re.findall(r"\d+", m.group(0))
        return " ".join(f"[[PMID:{p}]]({PUBMED_URL.format(pmid=p)})" for p in pmids)

    return re.sub(r"\[PMID:\s*\d+(?:\s*;\s*PMID:\s*\d+)*\]", _replace, text)


def load_case_json(data: dict[str, Any]) -> tuple[str, list[str]]:
    return data.get("patient_description", ""), list(data.get("questions", []))


def save_result(result: dict[str, Any], anonymize: bool) -> Path:
    config.ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = dict(result)
    if anonymize:
        out.pop("case_text", None)
    out_path = config.EVALUATION_RESULTS_DIR / f"result_{timestamp}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PubMed to Bedside",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Tighten top spacing */
.block-container { padding-top: 1.75rem !important; padding-bottom: 3rem !important; }

/* Sidebar background */
section[data-testid="stSidebar"] > div:first-child {
    background: #f1f5f9;
    border-right: 1px solid #e2e8f0;
}

/* Sidebar text */
section[data-testid="stSidebar"] .stMarkdown p {
    color: #475569;
    font-size: 0.85rem;
}

/* App title in sidebar */
.sidebar-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #0f3460;
    letter-spacing: -0.01em;
    margin-bottom: 2px;
    line-height: 1.2;
}
.sidebar-subtitle {
    font-size: 0.95rem;
    color: #64748b;
    margin-bottom: 0;
}

/* Primary run button */
div[data-testid="stButton"] > button[kind="primary"] {
    background: #0f3460 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1rem !important;
    transition: background 0.15s ease !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1a5276 !important;
}

/* Section header style for main content */
h3 { color: #0f3460 !important; font-size: 1rem !important; margin-top: 1.5rem !important; }

/* Answer text readability */
.stMarkdown p { line-height: 1.75; }

/* Expander header */
.streamlit-expanderHeader { font-weight: 600 !important; color: #334155 !important; }

/* Subtle divider color */
hr { border-color: #e2e8f0 !important; }

/* Hide Streamlit footer */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

if "ver" not in st.session_state:
    st.session_state["ver"] = 0
if "result" not in st.session_state:
    st.session_state["result"] = None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Institutional logos
    logo_fau, logo_prl, logo_uke = st.columns(3)
    with logo_fau:
        fau_path = IMG_DIR / "FAU_logo.png"
        if fau_path.exists():
            st.image(str(fau_path), use_container_width=True)
    with logo_prl:
        prl_path = IMG_DIR / "PR_Lab.png"
        if prl_path.exists():
            st.image(str(prl_path), use_container_width=True)
    with logo_uke:
        uke_path = IMG_DIR / "ErlangenHospital.png"
        if uke_path.exists():
            st.image(str(uke_path), use_container_width=True)

    st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)

    title_col, reset_col = st.columns([4, 1])
    with title_col:
        st.markdown('<p class="sidebar-title">PubMed to Bedside</p>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-subtitle">Clinical literature synthesis</p>', unsafe_allow_html=True)
    with reset_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("New", help="Clear all fields and start a new case"):
            st.session_state["ver"] += 1
            st.session_state["result"] = None
            st.rerun()

    st.divider()

    ver = st.session_state["ver"]

    input_mode = st.radio(
        "Input",
        ["Type / paste", "Upload file"],
        key=f"mode_{ver}",
        horizontal=True,
        label_visibility="collapsed",
    )

    case_text: str = ""
    questions: list[str] = []

    if input_mode == "Type / paste":
        raw_case = st.text_area(
            "Patient case description",
            height=210,
            key=f"case_{ver}",
            placeholder="Paste the patient case here...",
        )
        case_text = normalize_text(raw_case)
        raw_q = st.text_area(
            "Clinical questions",
            height=140,
            key=f"q_{ver}",
            placeholder="1. What systemic therapy would you recommend?\n2. What radiation dose and fractionation?",
        )
        questions = parse_questions(raw_q)

    elif input_mode == "Upload file":
        uploaded = st.file_uploader(
            "Upload .txt or .json", type=["txt", "json"], key=f"up_{ver}"
        )
        if uploaded is not None:
            raw_bytes = uploaded.read().decode("utf-8", errors="replace")
            if uploaded.name.endswith(".json"):
                try:
                    data = json.loads(raw_bytes)
                    case_text, questions = load_case_json(data)
                    case_text = normalize_text(case_text)
                    st.success(f"Loaded: {uploaded.name}")
                except json.JSONDecodeError as exc:
                    st.error(f"Invalid JSON: {exc}")
            else:
                case_text = normalize_text(raw_bytes)
                raw_q = st.text_area(
                    "Clinical questions",
                    height=120,
                    key=f"q_up_{ver}",
                    placeholder="1. Question one\n2. Question two",
                )
                questions = parse_questions(raw_q)

    # Case preview
    if case_text and len(case_text) > 30:
        with st.expander("Case preview", expanded=False):
            st.caption(case_text[:500] + ("..." if len(case_text) > 500 else ""))
        if questions:
            for i, q in enumerate(questions, 1):
                st.caption(f"{i}. {q[:75]}{'...' if len(q) > 75 else ''}")

    # Clinical image upload (optional)
    attached_images: list[dict[str, Any]] = []
    with st.expander("Clinical images (optional, up to 3)", expanded=False):
        st.caption("Attach imaging (e.g. PET-CT, MRI). With OpenAI, actual images are analyzed by the model; captions provide additional context for all providers.")
        for _i in range(3):
            _img = st.file_uploader(
                f"Image {_i + 1}",
                type=["png", "jpg", "jpeg"],
                key=f"img{_i}_{ver}",
            )
            if _img is not None:
                _cap = st.text_input(
                    f"Caption for image {_i + 1}",
                    key=f"cap{_i}_{ver}",
                    placeholder="e.g. Axial PET-CT: primary anal canal mass (A), inguinal node (B)...",
                )
                attached_images.append(
                    {"bytes": _img.getvalue(), "name": _img.name, "caption": _cap, "index": _i + 1}
                )

    # Build enriched case text (captions become part of LLM context)
    if attached_images:
        _img_ctx = "\n".join(
            f"Image {d['index']}: {d['caption']}"
            for d in attached_images
            if d["caption"].strip()
        )
        case_text_for_pipeline = (
            case_text + ("\n\nCLINICAL IMAGES PROVIDED:\n" + _img_ctx if _img_ctx else "")
        )
    else:
        case_text_for_pipeline = case_text

    st.divider()

    with st.expander("Settings", expanded=False):
        max_results = st.slider(
            "Max PubMed results per query", 5, 100, 30, 5, key=f"mr_{ver}"
        )
        max_synthesis = st.slider(
            "Max papers in synthesis", 3, 30, 5, 1, key=f"ms_{ver}"
        )
        include_fulltext = st.checkbox("Fetch PMC full text", value=True, key=f"ft_{ver}")
        save_enabled = st.checkbox("Save result to disk", value=False, key=f"sv_{ver}")
        anonymize_save = st.checkbox(
            "Omit case text from saved file",
            value=True,
            disabled=not save_enabled,
            key=f"anon_{ver}",
        )

    st.markdown(
        "<p style='font-size:0.78rem; font-weight:700; color:#475569; "
        "text-transform:uppercase; letter-spacing:0.07em; margin:0.6rem 0 0.2rem;'>"
        "Model</p>",
        unsafe_allow_html=True,
    )
    provider_label = st.radio(
        "Model",
        ["OpenAI — GPT-4o", "Open Source — Llama 3.1 8B (local)"],
        key=f"prov_{ver}",
        label_visibility="collapsed",
    )
    provider = "ollama" if "Open Source" in provider_label else "openai"

    if provider == "ollama" and not config.is_provider_available("ollama"):
        st.warning("Ollama is not running. Start it with `ollama serve` in a terminal.")
    elif provider == "openai" and not config.OPENAI_API_KEY:
        st.warning("No OpenAI API key found in .env. Set OPENAI_API_KEY or switch to Open Source.")

    if provider == "ollama":
        _ollama_cap = getattr(config, "MAX_SYNTHESIS_PAPERS_OLLAMA", 3)
        st.info(
            "**Open Source mode:** Query construction and abstract filtering are handled by "
            "OpenAI GPT-4o (if available) to ensure accurate PubMed queries and reliable "
            f"paper selection. Only the final answer synthesis uses Llama 3.1 8B locally. "
            f"Synthesis is capped at {_ollama_cap} papers to fit the model's context window."
        )

    run = st.button(
        "Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=not (case_text.strip() and len(questions) > 0),
    )


# ── Main: run pipeline ────────────────────────────────────────────────────────

if run and case_text.strip() and questions:
    with st.status("Running...", expanded=True) as status:
        # Query formulation and screening always use OpenAI when available — these stages
        # require reliable JSON output and MeSH knowledge that small local models lack.
        # The user-selected model (provider) is used only for synthesis.
        retrieval_provider = "openai" if config.OPENAI_API_KEY else provider

        st.write("Formulating PubMed queries...")
        query = formulate_query(case_text_for_pipeline, questions, provider=retrieval_provider)

        st.write(f"Retrieving papers...")
        papers = fetch_pubmed_papers_multi(query.query_strings, max_results=max_results)
        st.write(f"Retrieved {len(papers)} papers. Annotating titles...")

        _, title_trace = screen_titles(papers, case_text_for_pipeline, questions)

        st.write(f"Scoring {len(papers)} abstracts...")
        passed, abstract_trace = screen_abstracts(papers, case_text_for_pipeline, questions, provider=retrieval_provider)
        st.write(f"{len(passed)} papers passed abstract screening.")

        # Only process the top-N papers that will go into synthesis — no point fetching
        # full text for papers that passed screening but won't be used.
        synthesis_candidates = passed[:max_synthesis]
        if include_fulltext and synthesis_candidates:
            st.write(f"Fetching full text for top {len(synthesis_candidates)} synthesis papers...")
            evidence_papers = fetch_full_texts(synthesis_candidates, max_papers=max_synthesis)
        else:
            evidence_papers = [dict(p, used_full_text=False, full_text="") for p in synthesis_candidates]

        fulltext_count = sum(1 for p in evidence_papers if p.get("used_full_text"))
        st.write("Synthesizing answer...")

        title_score_map = {row["pmid"]: row for row in title_trace}
        abstract_score_map = {row["pmid"]: row for row in abstract_trace}

        # Build a lookup for full-text fetch results (only synthesis candidates were fetched)
        fetched_map = {p.get("pmid"): p for p in evidence_papers}
        fulltext_trace = [
            {
                "pmid": p.get("pmid", ""),
                "title": p.get("title", ""),
                "year": p.get("year", ""),
                "pmcid": p.get("pmcid", ""),
                "full_text_available": bool(p.get("pmcid")),
                "used_full_text": fetched_map.get(p.get("pmid"), {}).get("used_full_text", False),
                "abstract_score": p.get("abstract_score"),
            }
            for p in passed
        ]

        trace_log: dict[str, Any] = {
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
            "synthesis_papers": min(len(evidence_papers), max_synthesis),
            "title_screening_trace": title_trace,
            "abstract_screening_trace": abstract_trace,
            "fulltext_trace": fulltext_trace,
            "screening_trace": abstract_trace,
        }

        vision_images = attached_images if (provider == "openai" and attached_images) else None
        result = synthesize_answer(
            case_text_for_pipeline, questions, evidence_papers,
            trace_log=trace_log, max_papers=max_synthesis, provider=provider,
            images=vision_images, patient_profile=getattr(query, "patient_profile", ""),
        )
        result["case_text"] = case_text
        result["questions"] = questions
        result["provider"] = provider
        st.session_state["result"] = result
        st.session_state["result_images"] = attached_images
        status.update(label="Complete.", state="complete", expanded=False)

        if save_enabled:
            saved_path = save_result(result, anonymize=anonymize_save)
            anon_note = " (case text omitted)" if anonymize_save else ""
            st.caption(f"Saved to `{saved_path}`{anon_note}")


# ── Main: render results ──────────────────────────────────────────────────────

result = st.session_state.get("result")

if result is None:
    st.markdown("""
<div style="display:flex; flex-direction:column; align-items:center;
            justify-content:center; padding: 4rem 2rem 2rem; color: #94a3b8;">
  <p style="font-size:1.5rem; font-weight:600; color:#cbd5e1; margin-bottom:0.5rem;">
    PubMed to Bedside
  </p>
  <p style="font-size:0.95rem; text-align:center; max-width:480px; margin-bottom:0.3rem;">
    Enter a patient case and clinical questions in the sidebar,
    then click <strong>Run Pipeline</strong> to generate an evidence-based answer.
  </p>
  <p style="font-size:0.88rem; text-align:center; max-width:480px; color:#64748b;">
    Tip: open <strong>Settings</strong> in the sidebar before running to configure the number
    of PubMed results per query and the maximum papers used in synthesis.
  </p>
</div>
""", unsafe_allow_html=True)
    st.stop()

# Hallucination warning
hallucinated = result.get("hallucinated_pmids", [])
if hallucinated:
    st.error(
        "**Hallucinated citations detected.** The following PMIDs appear in the answer "
        "but were not in the fetched evidence set — verify independently before use: "
        + ", ".join(hallucinated)
    )

papers_used: list[dict[str, Any]] = result.get("papers_used", [])
if not papers_used:
    st.warning(
        "No papers passed abstract screening. The answer below has no literature support. "
        "Try increasing the number of results in Settings."
    )

# ── Clinical images (if any were attached) ───────────────────────────────────

result_images: list[dict[str, Any]] = st.session_state.get("result_images", [])
if result_images:
    st.markdown("### Clinical Images")
    img_cols = st.columns(min(len(result_images), 3))
    for col, img_data in zip(img_cols, result_images):
        with col:
            st.image(img_data["bytes"], use_container_width=True)
            if img_data["caption"]:
                st.caption(img_data["caption"])
    st.divider()

# ── Answer ────────────────────────────────────────────────────────────────────

st.markdown("### Answer")
st.markdown(linkify_pmids(result.get("answer_text", "")))

st.divider()

# ── Source documents ──────────────────────────────────────────────────────────

with st.expander(
    f"Source documents  —  {len(papers_used)} paper{'s' if len(papers_used) != 1 else ''} used in synthesis",
    expanded=True,
):
    if not papers_used:
        st.caption("No papers reached synthesis.")
    for paper in papers_used:
        pmid = paper.get("pmid", "")
        title = paper.get("title", "Unknown title")
        year = paper.get("year", "")
        score = paper.get("abstract_score")
        used_ft = paper.get("used_full_text", False)
        pmcid = paper.get("pmcid", "")
        authors = paper.get("authors", [])

        ft_label = "Full text" if used_ft else ("PMC available" if pmcid else "Abstract only")
        score_label = f"  |  Relevance {score:.1f}/10" if score is not None else ""
        header = f"{title[:85]}{'...' if len(title) > 85 else ''}  ({year})  |  {ft_label}{score_label}"

        with st.expander(header, expanded=False):
            link_col, info_col = st.columns([2, 1])
            with link_col:
                if pmid:
                    st.markdown(f"[PubMed PMID {pmid}]({PUBMED_URL.format(pmid=pmid)})")
                if pmcid:
                    st.caption(f"PMCID: {pmcid}")
                if authors:
                    st.caption(", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""))
            with info_col:
                if not pmcid:
                    st.caption("No PMC full text available.")
                elif used_ft:
                    st.caption("Full text was fetched and used.")
                else:
                    st.caption("Full text available but not fetched (limit reached).")

            if used_ft and paper.get("full_text"):
                st.markdown("**Full text (as sent to synthesis):**")
                st.text(paper["full_text"][:2500] + ("..." if len(paper.get("full_text", "")) > 2500 else ""))
            elif paper.get("abstract"):
                st.markdown("**Abstract:**")
                st.write(paper["abstract"])

# ── Sentence-level citation audit ─────────────────────────────────────────────

citations = result.get("citations", [])
with st.expander(
    f"Sentence-level citations  —  {len(citations)} citation{'s' if len(citations) != 1 else ''}",
    expanded=False,
):
    if not citations:
        st.caption("No inline citations found in the answer.")
    else:
        for cit in citations:
            pmid = cit.get("pmid", "")
            in_set = cit.get("in_fetched_set", True)
            year = cit.get("year", "")
            title = cit.get("title", "")
            used_ft = cit.get("used_full_text", False)
            quote = cit.get("evidence_quote", "")
            sentence = cit.get("sentence", "")

            if not in_set:
                st.error(
                    f"**PMID {pmid} — not in evidence set (possible hallucination)**  \n"
                    f"Sentence: _{sentence}_"
                )
            else:
                st.markdown(
                    f"**[PMID {pmid}]({PUBMED_URL.format(pmid=pmid)})** ({year})"
                    + ("  |  Full text" if used_ft else "  |  Abstract")
                )
                if title:
                    st.caption(title)
                st.markdown(f"> {sentence}")
                if quote:
                    st.caption(f'Source excerpt: "{quote[:300]}"')
            st.divider()

# ── Pipeline trace ────────────────────────────────────────────────────────────

trace = result.get("trace_log", {})
_provider_used = result.get("provider", "openai")
_model_label = (
    f"Open Source — Llama 3.1 8B (Ollama)" if _provider_used == "ollama"
    else f"OpenAI — GPT-4o"
)
with st.expander(f"Pipeline trace  —  {_model_label}", expanded=False):

    # Pre-compute synthesis set so both the metrics and the table use the same definition
    ab_trace = trace.get("abstract_screening_trace", [])
    ft_map = {r["pmid"]: r for r in trace.get("fulltext_trace", [])}
    n_synthesis = trace.get("synthesis_papers", 0)
    passed_sorted = sorted(
        [r for r in ab_trace if r.get("passed")],
        key=lambda r: r.get("score", 0),
        reverse=True,
    )
    synthesis_pmids = {r["pmid"] for r in passed_sorted[:n_synthesis]}
    ft_available_in_synth = sum(
        1 for pmid in synthesis_pmids if ft_map.get(pmid, {}).get("full_text_available")
    )

    # Funnel summary
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Retrieved", trace.get("retrieved_count", 0))
    c2.metric("Abstract passed", trace.get("passed_count", 0))
    c3.metric("In synthesis", trace.get("synthesis_papers", 0))
    c4.metric("Full text available", ft_available_in_synth)
    c5.metric("Full text used", trace.get("fulltext_count", 0))

    # Evidence table
    if ab_trace:
        import pandas as pd

        rows = []
        for row in sorted(ab_trace, key=lambda r: r.get("score", 0), reverse=True):
            pid = row.get("pmid", "")
            in_synthesis = pid in synthesis_pmids
            ft_info = ft_map.get(pid, {})
            rows.append({
                "PMID": pid,
                "Title": row.get("title", "")[:60],
                "Abstract score": round(row.get("score", 0), 1),
                "Passed screening": "Yes" if row.get("passed") else "No",
                "Used in synth.": "Yes" if in_synthesis else "No",
                "Full text": ("Yes" if ft_info.get("used_full_text") else "No") if in_synthesis else "-",
            })
        st.markdown("**Screening decisions for all retrieved papers (sorted by relevance):**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Queries used
    all_queries = trace.get("query", {}).get("all_queries", [])
    if all_queries:
        st.markdown("**PubMed queries used:**")
        for q in all_queries:
            st.code(q, language=None)

    with st.expander("Raw JSON trace", expanded=False):
        st.json(trace)
