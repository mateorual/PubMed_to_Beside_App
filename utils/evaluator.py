"""Evaluate generated answers against parsed expert opinions."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rouge_score import rouge_scorer

import config

LOGGER = logging.getLogger(__name__)


def evaluate_all(cases_dir: Path, results_dir: Path) -> list[dict[str, Any]]:
    """Evaluate all matching parsed cases and pipeline result files."""
    rows = []
    for case_path in sorted(cases_dir.glob("case*.json")):
        result_path = results_dir / f"{case_path.stem}_result.json"
        if not result_path.exists():
            LOGGER.warning("Missing result for %s", case_path.stem)
            continue
        rows.append(evaluate_case(case_path, result_path))
    return rows


def evaluate_case(case_path: Path, result_path: Path) -> dict[str, Any]:
    """Evaluate one generated answer against expert opinions."""
    case = json.loads(case_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    reference = " ".join(opinion.get("approach_summary", "") for opinion in case.get("expert_opinions", []))
    answer = result.get("answer_text", "")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(reference, answer)["rougeL"]
    bert = _bertscore(answer, reference)
    return {
        "case_id": case.get("case_id", case_path.stem),
        "rougeL_precision": rouge.precision,
        "rougeL_recall": rouge.recall,
        "rougeL_fmeasure": rouge.fmeasure,
        "bertscore_precision": bert.get("precision"),
        "bertscore_recall": bert.get("recall"),
        "bertscore_f1": bert.get("f1"),
        "citation_faithfulness": citation_faithfulness(result),
    }


def citation_faithfulness(result: dict[str, Any]) -> float:
    """Compute fraction of cited PMIDs present in fetched papers."""
    cited = set(re.findall(r"\[PMID:(\d+)\]", result.get("answer_text", "")))
    if not cited:
        return 0.0
    fetched = {str(paper.get("pmid")) for paper in result.get("papers_used", [])}
    return len(cited & fetched) / len(cited)


def save_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save evaluation rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "rougeL_precision",
        "rougeL_recall",
        "rougeL_fmeasure",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "citation_faithfulness",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bertscore(answer: str, reference: str) -> dict[str, float | None]:
    """Compute BERTScore if the optional dependency/model is available."""
    if not answer or not reference:
        return {"precision": None, "recall": None, "f1": None}
    try:
        from bert_score import score

        precision, recall, f1 = score([answer], [reference], lang="en", verbose=False)
        return {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0])}
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("BERTScore unavailable: %s", exc)
        return {"precision": None, "recall": None, "f1": None}


def main() -> None:
    """Run evaluation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases_dir", type=Path, default=config.PARSED_CASES_DIR)
    parser.add_argument("--results_dir", type=Path, default=config.EVALUATION_RESULTS_DIR)
    args = parser.parse_args()
    config.configure_logging()
    rows = evaluate_all(args.cases_dir, args.results_dir)
    output = args.results_dir / "evaluation_summary.csv"
    save_summary(rows, output)
    LOGGER.info("Saved %s", output)


if __name__ == "__main__":
    main()
