"""Stage 2: fetch PubMed records via pymed or NCBI E-utilities."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import re

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import config

LOGGER = logging.getLogger(__name__)
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def fetch_pubmed_papers(query: str, max_results: int = config.MAX_PUBMED_RESULTS) -> list[dict[str, Any]]:
    """Fetch PubMed papers and filter out records without abstracts."""
    config.ensure_directories()
    query = _sanitise_query(query)
    cache_path = _cache_path(query, max_results)
    if cache_path.exists():
        LOGGER.info("Loading PubMed results from cache: %s", cache_path)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    try:
        # E-utilities is primary because it supports sort=relevance explicitly.
        papers = _fetch_with_eutils(query, max_results)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("E-utilities fetch failed; falling back to pymed: %s", exc)
        papers = _fetch_with_pymed(query, max_results)

    papers = [paper for paper in papers if str(paper.get("abstract") or "").strip()]
    cache_path.write_text(json.dumps(papers, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Fetched %d PubMed records with abstracts", len(papers))
    return papers


def fetch_pubmed_papers_multi(
    queries: list[str], max_results: int = config.MAX_PUBMED_RESULTS
) -> list[dict[str, Any]]:
    """Fetch papers from multiple queries and deduplicate by PMID."""
    seen_pmids: set[str] = set()
    all_papers: list[dict[str, Any]] = []
    for query in queries:
        papers = fetch_pubmed_papers(query, max_results=max_results)
        added = 0
        for paper in papers:
            pmid = str(paper.get("pmid", ""))
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                all_papers.append(paper)
                added += 1
        LOGGER.info("Query %r: +%d new papers (%d total unique)", query[:70], added, len(all_papers))
    return all_papers


def _sanitise_query(query: str) -> str:
    """Strip quotes around multi-word MeSH terms; E-utilities rejects quoted MeSH tags."""
    # "Term With Spaces"[MeSH] → Term With Spaces[MeSH]
    return re.sub(r'"([^"]+?)"\[MeSH\]', r'\1[MeSH]', query)


def _cache_path(query: str, max_results: int) -> Path:
    """Return a deterministic cache path for a PubMed query."""
    digest = hashlib.sha256(f"{query}:{max_results}".encode("utf-8")).hexdigest()[:16]
    return config.CACHE_DIR / f"pubmed_{digest}.json"


def _fetch_with_pymed(query: str, max_results: int) -> list[dict[str, Any]]:
    """Fetch papers using the pymed package."""
    from pymed import PubMed

    pubmed = PubMed(tool=config.NCBI_TOOL, email=config.NCBI_EMAIL)
    results = pubmed.query(query, max_results=max_results)
    papers: list[dict[str, Any]] = []
    for article in results:
        pmid = str(getattr(article, "pubmed_id", "") or "").split("\n")[0]
        authors = []
        for author in getattr(article, "authors", []) or []:
            lastname = author.get("lastname") or ""
            firstname = author.get("firstname") or ""
            name = f"{firstname} {lastname}".strip()
            if name:
                authors.append(name)
        papers.append(
            {
                "pmid": pmid,
                "title": getattr(article, "title", "") or "",
                "abstract": getattr(article, "abstract", "") or "",
                "authors": authors,
                "year": _extract_year(getattr(article, "publication_date", None)),
                "pmcid": _extract_pmcid(getattr(article, "xml", None)),
            }
        )
    return papers


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    reraise=True,
)
def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET JSON from NCBI with retry on transport failures."""
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    reraise=True,
)
def _request_text(url: str, params: dict[str, Any]) -> str:
    """GET text from NCBI with retry on transport failures."""
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.text


def _fetch_with_eutils(query: str, max_results: int) -> list[dict[str, Any]]:
    """Fetch papers using the NCBI E-utilities REST API."""
    base_params: dict[str, Any] = {
        "tool": config.NCBI_TOOL,
        "email": config.NCBI_EMAIL,
        "api_key": config.NCBI_API_KEY,
    }
    search = _request_json(
        f"{EUTILS_BASE}/esearch.fcgi",
        {
            **base_params,
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance",
        },
    )
    pmids = search.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []
    time.sleep(0.34 if not config.NCBI_API_KEY else 0.11)
    xml_text = _request_text(
        f"{EUTILS_BASE}/efetch.fcgi",
        {**base_params, "db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
    )
    return _parse_pubmed_xml(xml_text)


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse PubMed efetch XML into paper dictionaries."""
    root = ET.fromstring(xml_text)
    papers: list[dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article.find(".//PMID"))
        title = " ".join(article.findtext(".//ArticleTitle", default="").split())
        abstract_parts = [_text(node) for node in article.findall(".//Abstract/AbstractText")]
        abstract = "\n".join(part for part in abstract_parts if part)
        authors = []
        for author in article.findall(".//Author"):
            collective = _text(author.find("CollectiveName"))
            if collective:
                authors.append(collective)
                continue
            name = " ".join(part for part in [_text(author.find("ForeName")), _text(author.find("LastName"))] if part)
            if name:
                authors.append(name)
        papers.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": _text(article.find(".//PubDate/Year")),
                "pmcid": _extract_pmcid(article),
            }
        )
    return papers


def _text(node: ET.Element | None) -> str:
    """Return concatenated text from an XML node."""
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _extract_pmcid(xml_obj: Any) -> str | None:
    """Extract a PMCID from XML-like pymed or ElementTree objects."""
    if xml_obj is None:
        return None
    try:
        root = ET.fromstring(xml_obj) if isinstance(xml_obj, str) else xml_obj
        for article_id in root.findall(".//ArticleId"):
            if article_id.attrib.get("IdType", "").lower() == "pmc":
                text = _text(article_id)
                return text if text.startswith("PMC") else f"PMC{text}"
    except Exception:  # noqa: BLE001
        return None
    return None


def _extract_year(value: Any) -> str:
    """Extract a publication year from a date-like value."""
    if not value:
        return ""
    match = __import__("re").search(r"\d{4}", str(value))
    return match.group(0) if match else ""
