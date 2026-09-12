import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from langchain_core.documents import Document


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "with", "kya", "hai", "hain", "ka", "ki", "ke",
    "ko", "mein", "me", "se", "aur", "batao", "bataiye",
}


def lexical_db_path(vector_db_dir: Path) -> Path:
    return vector_db_dir / "lexical.sqlite3"


def _connect(vector_db_dir: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(lexical_db_path(vector_db_dir), timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            collection_name UNINDEXED,
            chunk_id UNINDEXED,
            document,
            metadata_json UNINDEXED,
            tokenize='unicode61 remove_diacritics 2'
        )
        """
    )
    return connection


def delete_lexical_ids(vector_db_dir: Path, collection_name: str, ids: list[str]) -> None:
    if not ids:
        return
    with closing(_connect(vector_db_dir)) as connection:
        for start in range(0, len(ids), 400):
            batch = ids[start : start + 400]
            placeholders = ",".join("?" for _ in batch)
            connection.execute(
                f"DELETE FROM chunks_fts WHERE collection_name = ? AND chunk_id IN ({placeholders})",
                [collection_name, *batch],
            )
        connection.commit()


def add_lexical_chunks(
    vector_db_dir: Path,
    collection_name: str,
    chunks: list[Document],
    ids: list[str],
) -> None:
    rows = [
        (
            collection_name,
            chunk_id,
            chunk.page_content,
            json.dumps(chunk.metadata, ensure_ascii=False, default=str),
        )
        for chunk, chunk_id in zip(chunks, ids, strict=True)
    ]
    with closing(_connect(vector_db_dir)) as connection:
        connection.executemany(
            "INSERT INTO chunks_fts(collection_name, chunk_id, document, metadata_json) VALUES (?, ?, ?, ?)",
            rows,
        )
        connection.commit()


def reset_lexical_collection(vector_db_dir: Path, collection_name: str) -> None:
    with closing(_connect(vector_db_dir)) as connection:
        connection.execute(
            "DELETE FROM chunks_fts WHERE collection_name = ?",
            (collection_name,),
        )
        connection.commit()


def _match_query(question: str) -> str:
    seen = set()
    terms = []
    for token in TOKEN_RE.findall(question.lower()):
        if len(token) < 2 or token in seen or token in STOP_WORDS:
            continue
        seen.add(token)
        terms.append(token.replace('"', '""'))
        if len(terms) >= 16:
            break
    return " OR ".join(f'"{term}"' for term in terms)


def lexical_search(
    vector_db_dir: Path,
    collection_name: str,
    question: str,
    limit: int,
) -> list[Document]:
    path = lexical_db_path(vector_db_dir)
    query = _match_query(question)
    if not path.exists() or not query:
        return []

    try:
        with closing(_connect(vector_db_dir)) as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, document, metadata_json, bm25(chunks_fts) AS lexical_rank
                FROM chunks_fts
                WHERE chunks_fts MATCH ? AND collection_name = ?
                ORDER BY lexical_rank
                LIMIT ?
                """,
                (query, collection_name, limit),
            ).fetchall()
    except sqlite3.DatabaseError:
        return []

    documents = []
    for chunk_id, content, metadata_json, lexical_rank in rows:
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            metadata = {}
        metadata["chunk_id"] = chunk_id
        metadata["lexical_rank"] = float(lexical_rank)
        documents.append(Document(page_content=content, metadata=metadata))
    return documents
