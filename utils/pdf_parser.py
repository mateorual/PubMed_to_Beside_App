"""Parse Gray Zone PDF cases into structured JSON files."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

LOGGER = logging.getLogger(__name__)


def parse_pdf(pdf_path: Path, overrides_path: Path | None = None) -> dict[str, Any]:
    """Parse one Gray Zone PDF into a structured case dictionary."""
    text = extract_pdf_text(pdf_path)
    parsed = parse_case_text(text, case_id=_case_id_from_path(pdf_path))
    overrides = load_manual_overrides(overrides_path or config.PARSED_CASES_DIR / "manual_overrides.json")
    return {**parsed, **overrides.get(parsed["case_id"], {})}


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract normalized text from a PDF with pdfplumber."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text(x_tolerance=1, y_tolerance=3) or "" for page in pdf.pages]
    return "\n".join(pages)


def parse_case_text(text: str, case_id: str) -> dict[str, Any]:
    """Parse raw Gray Zone text into case, questions, and expert opinion sections."""
    normalized = re.sub(r"[ \t]+", " ", text.replace("\r", "\n"))
    first_question_match = re.search(r"(?:^|\n)\s*1[.)]\s+", normalized)
    first_question_start = first_question_match.start() if first_question_match else _find_opinion_start(normalized)
    opinion_start = _find_opinion_start(normalized, start=first_question_start or 0)
    question_region_end = opinion_start or len(normalized)
    question_region = normalized[first_question_start:question_region_end] if first_question_start else ""
    question_matches = list(
        re.finditer(
            r"(?:^|\n)\s*(\d+)[.)]\s+(.+?)(?=(?:\n\s*\d+[.)]\s+)|$)",
            question_region,
            re.I | re.S,
        )
    )
    questions = [_clean(match.group(2)) for match in question_matches]
    patient_description = _clean(normalized[: first_question_start or opinion_start])
    opinions_text = normalized[opinion_start:] if opinion_start else ""
    expert_opinions = parse_expert_opinions(opinions_text)
    return {
        "case_id": case_id,
        "patient_description": patient_description,
        "questions": questions,
        "expert_opinions": expert_opinions,
        "raw_text_length": len(normalized),
    }


def parse_expert_opinions(text: str) -> list[dict[str, str]]:
    """Extract coarse expert opinion blocks from text."""
    if not text.strip():
        return []
    author_pattern = r"(?=(?:Dr\.|Prof\.|Professor)\s+[A-Z][A-Za-z .'-]+)"
    blocks = [
        block.strip()
        for block in re.split(author_pattern, text)
        if block.strip() and not re.fullmatch(r"expert opinion|opinion|response", block.strip(), re.I)
    ]
    if len(blocks) <= 1:
        blocks = [text.strip()]
    opinions = []
    for block in blocks:
        author = _extract_author(block)
        opinions.append(
            {
                "author": author,
                "institution": _extract_institution(block),
                "approach_summary": _clean(block[:1500]),
                "recommended_doses": _extract_sentences(block, r"dose|Gy|fraction|fx|cGy"),
                "systemic_therapy_opinion": _extract_sentences(block, r"systemic|chemo|immunotherapy|therapy"),
                "brachytherapy_causality_opinion": _extract_sentences(block, r"brachy|causal|causality"),
            }
        )
    return opinions


def save_parsed_case(parsed: dict[str, Any], output_dir: Path = config.PARSED_CASES_DIR) -> Path:
    """Save a parsed case dictionary as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{parsed['case_id']}.json"
    output_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def parse_cases(input_dir: Path, output_dir: Path = config.PARSED_CASES_DIR) -> list[Path]:
    """Parse every PDF in a directory and save JSON outputs."""
    outputs = []
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        try:
            outputs.append(save_parsed_case(parse_pdf(pdf_path), output_dir))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to parse %s: %s", pdf_path, exc)
    return outputs


def load_manual_overrides(path: Path) -> dict[str, Any]:
    """Load manual parser overrides if present."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _case_id_from_path(path: Path) -> str:
    """Convert a PDF filename into a case identifier."""
    stem = path.stem.lower()
    match = re.match(r"case(\d+)", stem)
    return f"case{match.group(1)}" if match else stem


def _find_opinion_start(text: str, start: int = 0) -> int:
    """Find the likely beginning of expert responses."""
    match = re.search(r"\n\s*(?:GRAY ZONE\s+)?(?:Expert Opinions?|Opinion|Response|Dr\.|Prof\.)", text[start:], re.I)
    return start + match.start() if match else 0


def _extract_author(block: str) -> str:
    """Extract an expert author name from a block."""
    match = re.search(r"(Dr\.|Prof\.|Professor)\s+([A-Z][A-Za-z .'-]+)", block)
    return _clean(match.group(0)) if match else ""


def _extract_institution(block: str) -> str:
    """Extract a likely institution line from a block."""
    match = re.search(r"((?:University|Hospital|Clinic|Cancer Center|Institute)[^\n.]{0,120})", block, re.I)
    return _clean(match.group(1)) if match else ""


def _extract_sentences(block: str, pattern: str) -> str:
    """Extract sentences matching a keyword pattern."""
    sentences = re.split(r"(?<=[.!?])\s+", _clean(block))
    selected = [sentence for sentence in sentences if re.search(pattern, sentence, re.I)]
    return " ".join(selected[:4])


def _clean(text: str) -> str:
    """Normalize whitespace in parsed text."""
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    """Run the PDF parser from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--output_dir", type=Path, default=config.PARSED_CASES_DIR)
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    config.configure_logging()
    config.ensure_directories()
    if args.pdf:
        output = save_parsed_case(parse_pdf(args.pdf), args.output_dir)
        LOGGER.info("Saved %s", output)
    else:
        outputs = parse_cases(args.input_dir, args.output_dir)
        LOGGER.info("Saved %d parsed cases", len(outputs))


if __name__ == "__main__":
    main()
