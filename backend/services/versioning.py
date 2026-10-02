"""Version aware selection for documents with optional metadata."""

import hashlib
import re
from pathlib import Path


YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
VERSION = re.compile(r"\b(?:version|ver|revision|rev|v)[\s._-]*([0-9]+(?:\.[0-9]+)*)\b", re.I)
LATEST = re.compile(r"\b(?:latest|newest|current|most recent|updated|recent)\b", re.I)
HISTORICAL = re.compile(r"\b(?:historical|history|old|older|previous|earlier|archive)\b", re.I)


def version_metadata(doc) -> dict[str, str]:
    metadata = doc.metadata
    result = {}
    for key in ("year", "version", "effective_date", "revision", "source", "catalog", "schema", "table", "dataset", "entity"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            result[key] = str(value).strip()
    source = result.get("source", "")
    if "year" not in result:
        years = YEAR.findall(source)
        if len(set(years)) == 1:
            result["year"] = years[0]
    if "version" not in result:
        match = VERSION.search(Path(source).stem)
        if match:
            result["version"] = match.group(1)
    return result


def document_identity(doc) -> tuple:
    meta = version_metadata(doc)
    return tuple(meta.get(key, "") for key in ("source", "year", "version", "effective_date", "revision", "catalog", "schema", "table", "dataset", "entity")) + (
        doc.metadata.get("page"), doc.metadata.get("start_index"),
        doc.metadata.get("chunk_index"), hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:12],
    )


def _entity(doc) -> str:
    meta = version_metadata(doc)
    explicit = tuple(meta.get(key, "") for key in ("catalog", "schema", "table", "dataset", "entity"))
    if any(explicit):
        return "|".join(explicit).lower()
    source = Path(meta.get("source", "")).stem.lower()
    source = YEAR.sub("", source)
    source = VERSION.sub("", source)
    source = re.sub(r"[\W_]+", " ", source).strip()
    return source or meta.get("source", "").lower()


def _version(doc) -> tuple:
    meta = version_metadata(doc)
    return tuple(meta.get(key, "") for key in ("year", "version", "effective_date", "revision")) or ("",)


def _year_evidence(doc) -> set[str]:
    meta = version_metadata(doc)
    if meta.get("year"):
        return {meta["year"]}
    if meta.get("effective_date"):
        return set(YEAR.findall(meta["effective_date"]))
    # Older indexes have no version fields. Content dates are only used as
    # query evidence; they are not claimed as the document's version.
    return set(YEAR.findall(doc.page_content))


def _entity_for_query(doc, anchors: tuple[str, ...]) -> str:
    meta = version_metadata(doc)
    if any(meta.get(key) for key in ("catalog", "schema", "table", "dataset", "entity")):
        return _entity(doc)
    text = f"{meta.get('source', '')} {doc.page_content}"
    for anchor in anchors:
        if re.search(rf"\b{re.escape(anchor)}\b", text, re.I):
            return f"query:{anchor.lower()}"
    return _entity(doc)


def select_documents(docs: list, question: str, top_k: int, *, candidate_pool: bool = False) -> list:
    years = set(YEAR.findall(question))
    versions = set(VERSION.findall(question))
    anchors = tuple(token for token in re.findall(r"\b[A-Z][A-Z0-9_-]{2,}\b", question) if token not in {"WHAT", "HOW", "THE", "PDF", "AND", "SQL"})
    latest = bool(LATEST.search(question)) and not years and not versions
    historical = bool(HISTORICAL.search(question)) and not years and not versions
    unique = list(dict((document_identity(doc), doc) for doc in docs).values())
    if years or versions:
        unique = [
            doc for doc in unique
            if (not years or bool(_year_evidence(doc) & years))
            and (not versions or version_metadata(doc).get("version") in versions)
        ]
    if historical:
        unique.sort(key=lambda doc: int(version_metadata(doc).get("year", "9999")) if version_metadata(doc).get("year", "").isdigit() else 9999)
    if latest:
        by_entity: dict[str, list] = {}
        for doc in unique:
            by_entity.setdefault(_entity_for_query(doc, anchors), []).append(doc)
        selected = []
        for group in by_entity.values():
            known = [doc for doc in group if any(_version(doc))]
            if not known:
                selected.extend(group)
                continue
            key = lambda doc: (
                int(version_metadata(doc).get("year", "0")) if version_metadata(doc).get("year", "").isdigit() else 0,
                version_metadata(doc).get("effective_date", ""),
                tuple(int(part) for part in version_metadata(doc).get("version", "0").split(".") if part.isdigit()),
            )
            target = max(known, key=key)
            selected.extend(doc for doc in group if _version(doc) == _version(target))
        unique = selected
    if not unique:
        return []
    # Candidate gathering covers sources broadly. Final selection reserves
    # slots only for versions of the leading entity, then follows rank order.
    result, seen = [], set()
    leading_entity = _entity_for_query(unique[0], anchors)
    for doc in unique:
        entity = _entity_for_query(doc, anchors)
        version = _version(doc)
        key = (entity, version, doc.metadata.get("source")) if candidate_pool else version
        if (candidate_pool or (entity == leading_entity and any(version))) and key not in seen:
            result.append(doc)
            seen.add(key)
            if len(result) >= top_k:
                return result
    for doc in unique:
        if doc not in result:
            result.append(doc)
            if len(result) >= top_k:
                break
    return result
