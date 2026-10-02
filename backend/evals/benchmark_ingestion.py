"""Compare the previous flat DOCX path with structured parsing on one local file."""

import argparse
import json
import statistics
import time
from pathlib import Path

import config
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from train_engine import base_metadata, file_hash, load_docx, load_file, split_documents


def measure(path: Path, raw_root: Path, repeats: int = 3) -> dict:
    old_extract = []
    old_chunk = []
    new_extract = []
    new_chunk = []
    hash_times = []
    old_count = new_count = sections = 0
    old_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        add_start_index=True,
    )
    for _ in range(repeats):
        started = time.perf_counter()
        content = load_docx(path)
        old_extract.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        old_count = len(old_splitter.split_documents([Document(page_content=content, metadata=base_metadata(path, raw_root))]))
        old_chunk.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        docs = load_file(path, raw_root)
        new_extract.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        new_count = len(split_documents(docs))
        new_chunk.append((time.perf_counter() - started) * 1000)
        sections = len(docs)
        started = time.perf_counter()
        file_hash(path)
        hash_times.append((time.perf_counter() - started) * 1000)
    return {
        "file": path.name, "bytes": path.stat().st_size, "repeats": repeats,
        "legacy_extract_ms_p50": round(statistics.median(old_extract), 1),
        "legacy_chunk_ms_p50": round(statistics.median(old_chunk), 1),
        "structured_extract_ms_p50": round(statistics.median(new_extract), 1),
        "structured_chunk_ms_p50": round(statistics.median(new_chunk), 1),
        "unchanged_hash_ms_p50": round(statistics.median(hash_times), 1),
        "legacy_chunks": old_count, "structured_chunks": new_count, "structured_sections": sections,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path(config.RAW_DATA_DIR))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.path.suffix.lower() != ".docx":
        parser.error("This comparison requires a DOCX file used by the legacy extractor.")
    print(json.dumps(measure(args.path.resolve(), args.raw_root.resolve(), args.repeats), indent=2))
