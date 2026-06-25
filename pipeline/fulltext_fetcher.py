"""Stage 4: fetch available PMC full text for screened papers."""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import config

LOGGER = logging.getLogger(__name__)
PMC_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_full_texts(
    papers: list[dict[str, Any]],
    max_papers: int = config.MAX_FULLTEXT_PAPERS,
) -> list[dict[str, Any]]:
    """Populate full_text for top screened papers with PMCIDs."""
    enriched: list[dict[str, Any]] = []
    for index, paper in enumerate(papers):
        item = dict(paper)
        item["used_full_text"] = False
        if index < max_papers and paper.get("pmcid"):
            try:
                item["full_text"] = fetch_pmc_full_text(str(paper["pmcid"]))
                item["used_full_text"] = bool(item["full_text"])
                LOGGER.info("Fetched PMC full text for %s", paper.get("pmcid"))
                time.sleep(0.34 if not config.NCBI_API_KEY else 0.11)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Full-text fetch failed for %s: %s", paper.get("pmcid"), exc)
                item["full_text"] = ""
        else:
            item["full_text"] = ""
        enriched.append(item)
    return enriched


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    reraise=True,
)
def fetch_pmc_full_text(pmcid: str) -> str:
    """Fetch and parse full text XML from PMC."""
    clean_id = re.sub(r"^PMC", "", pmcid.strip(), flags=re.IGNORECASE)
    params: dict[str, Any] = {
        "db": "pmc",
        "id": clean_id,
        "rettype": "xml",
        "tool": config.NCBI_TOOL,
        "email": config.NCBI_EMAIL,
        "api_key": config.NCBI_API_KEY,
    }
    response = requests.get(PMC_EFETCH_URL, params=params, timeout=45)
    response.raise_for_status()
    return parse_pmc_body(response.text)


def parse_pmc_body(xml_text: str) -> str:
    """Extract section headers and body paragraphs from PMC XML."""
    root = ET.fromstring(xml_text)
    body = root.find(".//body")
    if body is None:
        return ""
    parts: list[str] = []
    for node in body.iter():
        tag = _strip_namespace(node.tag)
        if tag == "title":
            title = _node_text(node)
            if title:
                parts.append(f"\n## {title}")
        elif tag == "p":
            paragraph = _node_text(node)
            if paragraph:
                parts.append(paragraph)
    return "\n".join(parts).strip()


def _strip_namespace(tag: str) -> str:
    """Remove XML namespace from a tag."""
    return tag.rsplit("}", 1)[-1]


def _node_text(node: ET.Element) -> str:
    """Return normalized text for an XML node."""
    return " ".join("".join(node.itertext()).split())
