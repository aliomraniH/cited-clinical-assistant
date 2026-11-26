"""Pull recent PubMed records through NCBI E-utilities.

Only public endpoints are used. Set NCBI_API_KEY to raise the rate limit from
3 to 10 requests per second; nothing else identifying is sent.
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "cited-clinical-assistant"
BATCH = 200
BASE_FILTER = "english[Language] AND hasabstract"


@dataclass
class Record:
    pmid: str
    title: str
    abstract: str
    journal: str
    pub_date: str  # ISO-ish: YYYY, YYYY-MM or YYYY-MM-DD
    doi: str | None
    publication_types: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    n_authors: int = 0


class EUtils:
    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = api_key if api_key is not None else os.environ.get("NCBI_API_KEY")
        self.session = session or requests.Session()
        self.min_interval = 0.11 if self.api_key else 0.34
        self._last = 0.0

    def _get(self, endpoint: str, params: dict) -> requests.Response:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        params = {**params, "tool": TOOL_NAME}
        if self.api_key:
            params["api_key"] = self.api_key
        for attempt in range(4):
            resp = self.session.get(f"{EUTILS}/{endpoint}", params=params, timeout=60)
            self._last = time.monotonic()
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            return resp
        resp.raise_for_status()
        return resp

    def search(self, query: str, days: int, retmax: int = 5000) -> list[str]:
        """PMIDs added to PubMed (Entrez date) in the last `days` days."""
        params = {
            "db": "pubmed",
            "term": f"{query} AND {BASE_FILTER}",
            "reldate": days,
            "datetype": "edat",
            "retmax": retmax,
            "retmode": "json",
        }
        data = self._get("esearch.fcgi", params).json()
        return list(data["esearchresult"].get("idlist", []))

    def fetch(self, pmids: list[str]) -> list[Record]:
        records: list[Record] = []
        for i in range(0, len(pmids), BATCH):
            chunk = pmids[i : i + BATCH]
            resp = self._get("efetch.fcgi", {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"})
            records.extend(parse_pubmed_xml(resp.content))
        return records


def _text(el: ET.Element | None) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}


def _date(article: ET.Element) -> str:
    """Best available publication date: electronic date first, then journal issue date."""
    ad = article.find("ArticleDate")
    if ad is not None and ad.findtext("Year"):
        return f"{ad.findtext('Year')}-{int(ad.findtext('Month', '1')):02d}-{int(ad.findtext('Day', '1')):02d}"
    pd = article.find("Journal/JournalIssue/PubDate")
    if pd is None:
        return ""
    year = pd.findtext("Year")
    if not year:
        medline = pd.findtext("MedlineDate", "")
        return medline[:4] if medline[:4].isdigit() else ""
    month = pd.findtext("Month")
    if not month:
        return year
    m = int(month) if month.isdigit() else _MONTHS.get(month[:3])
    if not m:
        return year
    day = pd.findtext("Day")
    return f"{year}-{m:02d}-{int(day):02d}" if day and day.isdigit() else f"{year}-{m:02d}"


def parse_pubmed_xml(xml: bytes) -> list[Record]:
    root = ET.fromstring(xml)
    out: list[Record] = []
    for art in root.iter("PubmedArticle"):
        cit = art.find("MedlineCitation")
        if cit is None:
            continue
        article = cit.find("Article")
        if article is None:
            continue
        parts = []
        for ab in article.findall("Abstract/AbstractText"):
            label = ab.get("Label")
            txt = _text(ab)
            if txt:
                parts.append(f"{label}: {txt}" if label else txt)
        doi = None
        for aid in art.findall("PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = (aid.text or "").strip() or None
        out.append(
            Record(
                pmid=_text(cit.find("PMID")),
                title=_text(article.find("ArticleTitle")),
                abstract="\n".join(parts),
                journal=_text(article.find("Journal/Title")),
                pub_date=_date(article),
                doi=doi,
                publication_types=[_text(p) for p in article.findall("PublicationTypeList/PublicationType")],
                mesh_terms=[_text(m) for m in cit.findall("MeshHeadingList/MeshHeading/DescriptorName")],
                n_authors=len(article.findall("AuthorList/Author")),
            )
        )
    return out
