"""Conservative, offline checks before generation and while releasing answer text.

Scores only rank candidates. They do not establish that a requested fact exists.
This module therefore checks the requested subject, relation, version and exact
values against the selected passages. Ambiguous cases abstain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from services.versioning import VERSION, YEAR, version_metadata


class EvidenceStatus(str, Enum):
    ANSWERABLE = "ANSWERABLE"
    PARTIAL_EVIDENCE = "PARTIAL_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    REQUIRED_VERSION_NOT_FOUND = "REQUIRED_VERSION_NOT_FOUND"
    VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"


@dataclass(frozen=True)
class EvidenceDecision:
    status: EvidenceStatus
    documents: tuple
    reason: str
    missing_parts: tuple[str, ...] = ()
    verified_quotes: tuple[str, ...] = ()
    search_query: str = ""


_STOP = set("a an and are as at be by can could do does for from how in is it its of on or the their this to was what when where which who why with would you your me about explain give tell mean elaborate kare page document source states state again main kya hai ka ki ke ko mein me aur par se ek iska".split())
_RELATIONS = {
    "salary": ("salary", "wage", "pay", "compensation"),
    "warranty": ("warranty", "guarantee"),
    "serial number": ("serial number", "serial no", "serial #"),
    "production": ("produced", "production", "output"),
}
_CROSS_LANGUAGE_TERMS = {
    "fayda": ("advantage", "benefit"),
    "faayda": ("advantage", "benefit"),
    "suraksha": ("safety",),
    "khadan": ("mine", "mining"),
    "khanan": ("mining", "excavation"),
    "niyam": ("rule", "rules"),
    "uddeshya": ("purpose", "function"),
    "matlab": ("meaning", "definition"),
}
_WORD = re.compile(r"[\w]+(?:[-'][\w]+)*", re.UNICODE)
_NUMBER = re.compile(r"(?<![\w])\d+(?:[.,]\d+)*(?:\s*%|\s*[a-zA-Z]+)?")
_MEASURE = re.compile(r"\b\d+(?:[.,]\d+)*(?:\s*)(?:kg|g|tonnes?|tons?|%|metres?|meters?|hours?|days?)\b", re.I)


def _terms(value: str) -> set[str]:
    words = (word.casefold() for word in _WORD.findall(value))
    return {word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word
            for word in words if len(word) > 2 and word not in _STOP}


def _parts(question: str) -> list[str]:
    # Split explicit separate requests; ordinary compound nouns stay together.
    return [p.strip(" ,?;") for p in re.split(
        r",?\s+and\s+(?=(?:what|which|who|when|where|how)\b)", question, flags=re.I
    ) if p.strip(" ,?;")]


def _version_matches(question: str, docs: list) -> bool:
    years = set(YEAR.findall(question))
    versions = set(VERSION.findall(question))
    if not years and not versions:
        return True
    found_years: set[str] = set()
    found_versions: set[str] = set()
    for doc in docs:
        meta = version_metadata(doc)
        text_years = set(YEAR.findall(doc.page_content))
        found_years.update(set(YEAR.findall(meta.get("year", ""))) | text_years)
        if meta.get("version"):
            found_versions.add(meta["version"])
    return years <= found_years and versions <= found_versions


def _supports_part(question: str, docs: list) -> bool:
    terms = _terms(question)
    if not terms:
        return False
    for doc in docs:
        content = doc.page_content.casefold()
        searchable = content + " " + str(doc.metadata.get("source", "")).casefold() + " " + \
            str(doc.metadata.get("year", "")) + " " + str(doc.metadata.get("version", ""))
        present = _terms(searchable)
        matched = sum(term in present or any(alias in present for alias in _CROSS_LANGUAGE_TERMS.get(term, ()))
                      for term in terms)
        if matched < min(2, len(terms)) or matched / len(terms) < 0.5:
            continue
        # A related subject is not evidence for an absent relation.
        if any(cue in question.casefold() and not any(word in content for word in variants)
               for cue, variants in _RELATIONS.items()):
            continue
        if "critical mines" in question.casefold() and "critical mines" not in content:
            continue
        if "how many" in question.casefold() and not re.search(r"\d", content):
            continue
        return True
    return False


def _conflicting_measurements(question: str, docs: list) -> bool:
    if len(docs) < 2 or re.search(r"\b(?:compare|difference|versus|vs)\b", question, re.I):
        return False
    topic = _terms(question) - {"many", "much", "exact", "number", "period", "value"}
    if len(topic) < 2:
        return False
    by_scope: dict[tuple[str, str], set[str]] = {}
    for doc in docs:
        meta = version_metadata(doc)
        scope = (meta.get("year", ""), meta.get("version", ""))
        for sentence in re.split(r"(?<=[.!?])\s+|\n", doc.page_content):
            if len(_terms(sentence) & topic) < 2:
                continue
            measurements = {re.sub(r"\s+", "", hit.group().casefold())
                            for hit in _MEASURE.finditer(sentence)}
            if measurements:
                by_scope.setdefault(scope, set()).update(measurements)
    return any(len(values) > 1 for values in by_scope.values())


def assess_evidence(question: str, documents: list) -> EvidenceDecision:
    docs = list(documents)
    if not docs:
        status = (EvidenceStatus.REQUIRED_VERSION_NOT_FOUND if
                  (YEAR.search(question) and re.search(r"\b(?:edition|version|revision|rules)\b", question, re.I))
                  or VERSION.search(question) else EvidenceStatus.INSUFFICIENT_EVIDENCE)
        return EvidenceDecision(status, (), "no_retrieved_evidence")
    if not _version_matches(question, docs):
        return EvidenceDecision(EvidenceStatus.REQUIRED_VERSION_NOT_FOUND, (), "requested_version_absent")
    parts = _parts(question)
    supported = [_supports_part(part, docs) for part in parts]
    if all(supported) and _conflicting_measurements(question, docs):
        return EvidenceDecision(EvidenceStatus.CONFLICTING_EVIDENCE, tuple(docs),
                                "same_scope_measurements_disagree")
    if all(supported):
        return EvidenceDecision(EvidenceStatus.ANSWERABLE, tuple(docs), "direct_term_and_relation_support")
    if any(supported):
        return EvidenceDecision(EvidenceStatus.PARTIAL_EVIDENCE, tuple(docs),
                                "some_requested_parts_unsupported",
                                tuple(part for part, ok in zip(parts, supported) if not ok))
    return EvidenceDecision(EvidenceStatus.INSUFFICIENT_EVIDENCE, (), "requested_fact_not_supported")


def unsupported_answer_numbers(answer: str, question: str, documents: list) -> set[str]:
    """Catch newly invented numeric claims; years already present in the question are allowed."""
    evidence = " ".join(doc.page_content + " " + str(doc.metadata.get("source", ""))
                        for doc in documents)
    permitted = {m.group().strip().casefold() for m in _NUMBER.finditer(evidence + " " + question)}
    for doc in documents:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            permitted.update((str(page), str(page + 1)))
    # Citation labels refer to the selected passage order, not a factual count.
    permitted.update(str(i) for i in range(1, len(documents) + 1))
    return {m.group().strip().casefold() for m in _NUMBER.finditer(answer)} - permitted
