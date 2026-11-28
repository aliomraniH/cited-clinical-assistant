"""Sources beyond PubMed abstracts: PMC and Europe PMC open-access full text,
ClinicalTrials.gov records, and VA/DoD clinical practice guidelines.

Every source becomes passage rows with the same shape as PubMed rows, so BM25 and the
answerer need no changes. The citable id stays in `pmid` (a PMID when the source has one,
else an NCT id or a guideline page id); `id` is unique per passage and `cite_label` is what
the answerer shows the model.

Licence rule: full text is kept only under CC BY or CC0 (checked in the article's own
licence statement, not only the search filter). Trials and VA/DoD guidelines are US federal
works. Configuration lives in config/sources.json.
"""

from __future__ import annotations

import hashlib
import io
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from .clean import normalize, text_hash
from .pubmed import EUtils

EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
CT_GOV = "https://clinicaltrials.gov/api/v2/studies"
VA_BASE = "https://www.healthquality.va.gov"
CHUNK_WORDS = 220
SKIP_SECTIONS = re.compile(r"^(references?|acknowledg|funding|author contributions?|conflicts? of interest|"
                           r"competing interests?|data availability|supplementary|abbreviations|ethics|"
                           r"institutional review|informed consent)", re.I)


# Citation markers left empty once <xref> is removed: "[ , ]", "( )", "[–]".
_EMPTY_REF = re.compile(r"\s*[\[(][\s,;–-]*[\])]")


# ---------- licences ----------

def licence_kind(text: str) -> str | None:
    """'cc-by' or 'cc0' when the licence allows reuse with credit; None for NC/ND/SA or unknown.

    Reads licence URLs (creativecommons.org/licenses/by/4.0) and plain words ("CC BY 4.0").
    """
    t = (text or "").lower()
    if "publicdomain/zero" in t or re.search(r"\bcc0\b|cc zero", t):
        return "cc0"
    if re.search(r"licenses/by-(nc|nd|sa)|cc[ -]by[ -](nc|nd|sa)|noncommercial|non-commercial|noderivatives", t):
        return None
    if re.search(r"licenses/by/|\bcc[ -]by\b|creative commons attribution", t):
        return "cc-by"
    return None


# ---------- JATS full text ----------

@dataclass
class Article:
    pmcid: str
    pmid: str | None
    doi: str | None
    title: str
    journal: str
    pub_date: str
    licence_text: str
    sections: list[tuple[str, str]] = field(default_factory=list)  # (heading, text)

    @property
    def licence(self) -> str | None:
        return licence_kind(self.licence_text)


def _text(el: ET.Element | None) -> str:
    return normalize(" ".join(el.itertext())) if el is not None else ""


def _strip(root: ET.Element, tags: tuple[str, ...]) -> None:
    """Remove elements but keep the text that follows them (their tail)."""
    for parent in list(root.iter()):
        prev = None
        for child in list(parent):
            if child.tag in tags:
                if child.tail:
                    if prev is not None:
                        prev.tail = (prev.tail or "") + child.tail
                    else:
                        parent.text = (parent.text or "") + child.tail
                parent.remove(child)
            else:
                prev = child


def _sections(body: ET.Element) -> list[tuple[str, str]]:
    """Flatten <sec> trees into (heading path, paragraph text); tables, figures and formulas dropped."""
    _strip(body, ("table-wrap", "fig", "disp-formula", "xref", "table"))
    out: list[tuple[str, str]] = []

    def walk(sec: ET.Element, path: list[str]) -> None:
        title = _text(sec.find("title"))
        here = path + [title] if title else path
        if title and SKIP_SECTIONS.match(title):
            return
        paras = [_text(p) for p in sec.findall("p")]
        text = "\n".join(p for p in paras if p)
        if text:
            out.append((" > ".join(here), text))
        for sub in sec.findall("sec"):
            walk(sub, here)

    loose = [_text(p) for p in body.findall("p")]
    if any(loose):
        out.append(("", "\n".join(p for p in loose if p)))
    for sec in body.findall("sec"):
        walk(sec, [])
    return out


def parse_jats(xml: bytes) -> list[Article]:
    root = ET.fromstring(xml)
    arts = [root] if root.tag == "article" else list(root.iter("article"))
    out = []
    for art in arts:
        meta = art.find("front/article-meta")
        if meta is None:
            continue
        ids = {a.get("pub-id-type"): (a.text or "").strip() for a in meta.findall("article-id")}
        pmcid = ids.get("pmcid") or ids.get("pmc") or ""
        if pmcid and not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        lic = meta.find("permissions/license")
        lic_text = ""
        if lic is not None:
            href = lic.get("{http://www.w3.org/1999/xlink}href") or ""
            lic_text = f"{href} {lic.get('license-type') or ''} {_text(lic)}"
        year = meta.findtext("pub-date/year") or ""
        sections: list[tuple[str, str]] = []
        abstract = meta.find("abstract")
        if abstract is not None:
            ab = "\n".join(_text(p) for p in abstract.iter("p")) or _text(abstract)
            if ab:
                sections.append(("Abstract", ab))
        body = art.find("body")
        if body is not None:
            sections.extend(_sections(body))
        out.append(Article(
            pmcid=pmcid, pmid=ids.get("pmid") or None, doi=ids.get("doi") or None,
            title=_text(meta.find("title-group/article-title")),
            journal=_text(art.find("front/journal-meta/journal-title-group/journal-title")),
            pub_date=year, licence_text=lic_text, sections=sections))
    return out


def chunk_text(text: str, max_words: int = CHUNK_WORDS) -> list[str]:
    """Split on paragraph then sentence boundaries into passages of at most ~max_words words."""
    pieces: list[str] = []
    for para in text.split("\n"):
        words = para.split()
        if len(words) <= max_words:
            pieces.append(para)
            continue
        cur: list[str] = []
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if cur and len(" ".join(cur).split()) + len(sent.split()) > max_words:
                pieces.append(" ".join(cur))
                cur = []
            cur.append(sent)
        if cur:
            pieces.append(" ".join(cur))
    chunks, cur_words = [], []
    for p in pieces:
        w = p.split()
        if cur_words and len(cur_words) + len(w) > max_words:
            chunks.append(" ".join(cur_words))
            cur_words = []
        cur_words.extend(w)
    if cur_words:
        chunks.append(" ".join(cur_words))
    return [c for c in chunks if len(c.split()) >= 12]


def _row(*, id: str, cite: str, cite_label: str, title: str, text: str, journal: str, pub_date: str,
         topics: list[str], source: str, url: str, terms: str, licence: str, section: str, fetched: str) -> dict:
    return {"id": id, "pmid": cite, "cite_label": cite_label, "title": title, "abstract": text,
            "journal": journal, "pub_date": pub_date, "topics": topics, "source": source,
            "source_id": cite, "url": url, "fetched_at": fetched, "terms": terms, "licence": licence,
            "section": section, "text_hash": text_hash(title, text)}


def article_rows(art: Article, topics: list[str], source: str, fetched: str | None = None) -> list[dict]:
    """Passage rows for one open-access article; [] when its licence is not CC BY or CC0."""
    lic = art.licence
    fetched = fetched or datetime.now(timezone.utc).isoformat(timespec="seconds")
    if lic is None:
        return []
    cite = art.pmid or art.pmcid
    label = f"PMID {art.pmid}" if art.pmid else art.pmcid
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{art.pmcid}/"
    rows, n = [], 0
    for heading, text in art.sections:
        for chunk in chunk_text(_EMPTY_REF.sub("", text)):
            n += 1
            rows.append(_row(id=f"{art.pmcid}#{n}", cite=cite, cite_label=f"{label}, {heading or 'Text'}",
                             title=art.title, text=chunk, journal=art.journal, pub_date=art.pub_date,
                             topics=topics, source=source, url=url, licence=lic, section=heading,
                             terms=f"Open-access full text under {lic.upper()}; reuse with attribution.",
                             fetched=fetched))
    return rows


# ---------- PMC through E-utilities ----------

OA_FILTER = '"open access"[filter] AND ("cc by license"[filter] OR "cc0 license"[filter])'


class PMC:
    """PMC open-access subset via E-utilities (an NCBI-approved route for automated retrieval)."""

    def __init__(self, eutils: EUtils | None = None):
        self.eu = eutils or EUtils()

    def search(self, query: str, days: int | None = None, retmax: int = 50) -> list[str]:
        params = {"db": "pmc", "term": f"{query} AND {OA_FILTER}", "retmax": retmax, "retmode": "json",
                  "sort": "relevance"}
        if days:
            params.update(reldate=days, datetype="edat")
        data = self.eu._get("esearch.fcgi", params).json()
        return [f"PMC{i}" for i in data["esearchresult"].get("idlist", [])]

    def fetch(self, pmcids: list[str], batch: int = 20) -> list[Article]:
        out: list[Article] = []
        for i in range(0, len(pmcids), batch):
            ids = ",".join(p.removeprefix("PMC") for p in pmcids[i:i + batch])
            out.extend(parse_jats(self.eu._get("efetch.fcgi", {"db": "pmc", "id": ids}).content))
        return out


# ---------- Europe PMC ----------

class EuropePMC:
    def __init__(self, session: requests.Session | None = None, min_interval: float = 0.2):
        self.s = session or requests.Session()
        self.min_interval, self._last = min_interval, 0.0

    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            r = self.s.get(url, params=params, timeout=60)
            self._last = time.monotonic()
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            return r
        return r

    def search(self, query: str, n: int = 50, since: str | None = None) -> list[dict]:
        q = f'({query}) AND OPEN_ACCESS:y AND (LICENSE:"cc by" OR LICENSE:"cc0") AND HAS_FT:y'
        if since:
            q += f" AND FIRST_PDATE:[{since} TO 3000-01-01]"
        r = self._get(f"{EUROPE_PMC}/search", {"query": q, "format": "json", "pageSize": min(n, 1000),
                                               "resultType": "lite"})
        r.raise_for_status()
        return [x for x in r.json().get("resultList", {}).get("result", []) if x.get("pmcid")][:n]

    def full_text(self, pmcid: str) -> Article | None:
        r = self._get(f"{EUROPE_PMC}/{pmcid}/fullTextXML")
        if r.status_code != 200 or not r.content.strip():
            return None
        arts = parse_jats(r.content)
        return arts[0] if arts else None


# ---------- ClinicalTrials.gov ----------

def trial_rows(study: dict, topics: list[str], fetched: str | None = None) -> list[dict]:
    """Passages for one study: summary, eligibility (inclusion/exclusion) and primary outcomes."""
    fetched = fetched or datetime.now(timezone.utc).isoformat(timespec="seconds")
    ps = study.get("protocolSection", {})
    nct = ps.get("identificationModule", {}).get("nctId", "")
    title = ps.get("identificationModule", {}).get("briefTitle", "")
    status = ps.get("statusModule", {})
    start = (status.get("startDateStruct") or {}).get("date", "")
    phases = ", ".join(ps.get("designModule", {}).get("phases", []) or [])
    conds = ", ".join(ps.get("conditionsModule", {}).get("conditions", []) or [])
    head = f"Status {status.get('overallStatus', '?')}; phase {phases or 'n/a'}; conditions: {conds}."
    parts = [("Summary", f"{head} {ps.get('descriptionModule', {}).get('briefSummary', '')}"),
             ("Eligibility", ps.get("eligibilityModule", {}).get("eligibilityCriteria", "")),
             ("Primary outcomes", "; ".join(o.get("measure", "") + (f" ({o['timeFrame']})" if o.get("timeFrame") else "")
                                             for o in ps.get("outcomesModule", {}).get("primaryOutcomes", []) or []))]
    rows, n = [], 0
    for sec, text in parts:
        for chunk in chunk_text(normalize(text)):
            n += 1
            rows.append(_row(id=f"{nct}#{n}", cite=nct, cite_label=f"{nct}, {sec}", title=title, text=chunk,
                             journal="ClinicalTrials.gov", pub_date=start, topics=topics, source="clinicaltrials",
                             url=f"https://clinicaltrials.gov/study/{nct}", licence="us-gov",
                             terms="ClinicalTrials.gov record (US government data).", section=sec, fetched=fetched))
    return rows


class ClinicalTrials:
    def __init__(self, session: requests.Session | None = None):
        self.s = session or requests.Session()

    def search(self, condition: str, n: int = 50, statuses: str | None = None) -> list[dict]:
        params = {"query.cond": condition, "pageSize": min(n, 100), "sort": "LastUpdatePostDate:desc"}
        if statuses:
            params["filter.overallStatus"] = statuses
        out, token = [], None
        while len(out) < n:
            if token:
                params["pageToken"] = token
            r = self.s.get(CT_GOV, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            out.extend(data.get("studies", []))
            token = data.get("nextPageToken")
            if not token:
                break
        return out[:n]


# ---------- VA/DoD guidelines ----------

def guideline_rows(pages: list[str], gid: str, title: str, url: str, year: str, topics: list[str],
                   fetched: str | None = None) -> list[dict]:
    """One or more passages per PDF page; the citation names the guideline and page."""
    fetched = fetched or datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for pno, page in enumerate(pages, start=1):
        text = normalize(re.sub(r"[ \t]+", " ", page))
        for j, chunk in enumerate(chunk_text(text.replace("\n", " ")), start=1):
            rows.append(_row(id=f"{gid}-p{pno}-{j}", cite=f"{gid} p.{pno}", cite_label=f"{title}, p.{pno}",
                             title=title, text=chunk, journal="VA/DoD Clinical Practice Guideline", pub_date=year,
                             topics=topics, source="va_dod_guideline", url=f"{url}#page={pno}", licence="us-gov",
                             terms="US federal guideline (VA/DoD), public domain; cite the guideline and page.",
                             section=f"page {pno}", fetched=fetched))
    return rows


def pdf_pages(data: bytes) -> list[str]:
    from pypdf import PdfReader  # optional dependency: pip install -e ".[sources]"
    return [(p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages]


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]
