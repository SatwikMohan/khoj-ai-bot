import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import chromadb
import docx2txt
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openpyxl import load_workbook
from pptx import Presentation


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT
RAW_DATA_DIR = REPO_ROOT / "raw_data_files"
VECTOR_DB_DIR = REPO_ROOT / "vector_db"
MANIFEST_FILE_NAME = "ingestion_manifest.json"
INGESTION_VERSION = 4
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".css",
    ".html",
    ".htm",
    ".xml",
    ".yaml",
    ".yml",
}


def load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default.resolve()

    path = Path(value)
    if path.is_absolute():
        return path

    for base_dir in (REPO_ROOT, PROJECT_ROOT):
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate

    return (REPO_ROOT / path).resolve()


def chunk_settings() -> dict:
    return {
        "version": INGESTION_VERSION,
        "embedding_provider": "ollama",
        "embedding_model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "850")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "150")),
        "collection_name": os.getenv("CHROMA_COLLECTION_NAME", "texmin_qa"),
    }


def relative_source(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.relative_to(PROJECT_ROOT).as_posix()


def raw_relative_source(path: Path, raw_data_dir: Path = RAW_DATA_DIR) -> str:
    return path.relative_to(raw_data_dir).as_posix()


def base_metadata(path: Path, raw_data_dir: Path = RAW_DATA_DIR) -> dict:
    raw_relative_path = raw_relative_source(path, raw_data_dir)
    folder_path = Path(raw_relative_path).parent.as_posix()
    if folder_path == ".":
        folder_path = ""

    return {
        "source": relative_source(path),
        "raw_source": raw_relative_path,
        "folder_path": folder_path,
        "folder_depth": 0 if not folder_path else len(Path(folder_path).parts),
        "file_name": path.name,
        "file_path": str(path),
        "extension": path.suffix.lower(),
    }


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def file_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> str:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            rows.append(", ".join(cell.strip() for cell in row if cell is not None))
    return "\n".join(rows)


def load_json(path: Path) -> str:
    data = json.loads(read_text(path))
    return json.dumps(data, indent=2, ensure_ascii=False)


def load_docx(path: Path) -> str:
    return docx2txt.process(str(path)) or ""


def load_xlsx(path: Path) -> str:
    workbook = load_workbook(path, read_only=True, data_only=True)
    parts = []
    for sheet in workbook.worksheets:
        parts.append(f"Sheet: {sheet.title}")
        for row in sheet.iter_rows(values_only=True):
            values = [str(value) for value in row if value is not None]
            if values:
                parts.append("\t".join(values))
    return "\n".join(parts)


def load_pptx(path: Path) -> str:
    presentation = Presentation(str(path))
    parts = []
    for slide_number, slide in enumerate(presentation.slides, start=1):
        parts.append(f"Slide {slide_number}")
        for shape in slide.shapes:
            text = getattr(shape, "text", "")
            if text:
                parts.append(text)
    return "\n".join(parts)


def load_pdf(path: Path, raw_data_dir: Path) -> list[Document]:
    documents = []
    loaded_documents = PyPDFLoader(str(path)).load()
    min_text_chars = int(os.getenv("MIN_EXTRACTED_TEXT_CHARS", "80"))
    for document in loaded_documents:
        if len(" ".join(document.page_content.split())) < min_text_chars:
            continue
        document.metadata.update(base_metadata(path, raw_data_dir))
        documents.append(document)
    return documents


def load_file(path: Path, raw_data_dir: Path) -> list[Document]:
    suffix = path.suffix.lower()
    metadata = base_metadata(path, raw_data_dir)

    if suffix == ".pdf":
        return load_pdf(path, raw_data_dir)
    if suffix == ".csv":
        text = load_csv(path)
    elif suffix == ".json":
        text = load_json(path)
    elif suffix == ".docx":
        text = load_docx(path)
    elif suffix in {".xlsx", ".xlsm"}:
        text = load_xlsx(path)
    elif suffix == ".pptx":
        text = load_pptx(path)
    elif suffix in TEXT_EXTENSIONS:
        text = read_text(path)
    else:
        print(f"Skipping unsupported file type: {relative_source(path)}")
        return []

    if not text.strip():
        print(f"Skipping empty file: {relative_source(path)}")
        return []

    return [Document(page_content=text, metadata=metadata)]


def iter_raw_files(raw_data_dir: Path) -> list[Path]:
    raw_data_dir = raw_data_dir.resolve()
    if not raw_data_dir.exists():
        raise RuntimeError(f"Raw data folder does not exist: {raw_data_dir}")

    return sorted(
        path
        for path in raw_data_dir.rglob("*")
        if path.is_file()
        and path.resolve().is_relative_to(raw_data_dir)
        and not any(part.startswith(".") for part in path.relative_to(raw_data_dir).parts)
    )


def load_documents(raw_data_dir: Path) -> list[Document]:
    raw_data_dir = raw_data_dir.resolve()
    documents = []
    for path in iter_raw_files(raw_data_dir):
        try:
            documents.extend(load_file(path, raw_data_dir))
        except Exception as exc:
            print(f"Could not load {relative_source(path)}: {exc}")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    settings = chunk_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings["chunk_size"],
        chunk_overlap=settings["chunk_overlap"],
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks


def chunk_id(chunk: Document) -> str:
    identity = {
        "source": chunk.metadata.get("source"),
        "page": chunk.metadata.get("page"),
        "start_index": chunk.metadata.get("start_index"),
        "content": chunk.page_content,
    }
    return hashlib.sha1(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def manifest_path(vector_db_dir: Path) -> Path:
    return vector_db_dir / MANIFEST_FILE_NAME


def load_manifest(vector_db_dir: Path) -> dict:
    path = manifest_path(vector_db_dir)
    if not path.exists():
        return {"settings": chunk_settings(), "files": {}}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"settings": {}, "files": {}}


def save_manifest(vector_db_dir: Path, manifest: dict) -> None:
    path = manifest_path(vector_db_dir)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def create_vector_store(reset_collection: bool = False) -> tuple[Chroma, Path, str]:
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "texmin_qa")
    vector_db_dir = resolve_path(os.getenv("VECTOR_DB_DIR"), VECTOR_DB_DIR)

    vector_db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(vector_db_dir))
    if reset_collection:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        keep_alive=int(os.getenv("OLLAMA_KEEP_ALIVE", "1800")),
    )
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vector_store, vector_db_dir, collection_name


def delete_ids(vector_store: Chroma, ids: list[str]) -> None:
    if ids:
        vector_store.delete(ids=ids)


def add_chunks(vector_store: Chroma, chunks: list[Document]) -> list[str]:
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    ids = [chunk_id(chunk) for chunk in chunks]

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        vector_store.add_documents(chunks[start:end], ids=ids[start:end])
        print(f"Embedded chunks {start + 1}-{min(end, len(chunks))} of {len(chunks)}")
    return ids


def train(raw_data_dir: Path, force_rebuild: bool = False) -> None:
    load_environment()
    raw_data_dir = raw_data_dir.resolve()

    raw_files = iter_raw_files(raw_data_dir)
    if not raw_files:
        raise RuntimeError(f"No supported documents found in {raw_data_dir}")

    current_settings = chunk_settings()
    vector_store, vector_db_dir, _ = create_vector_store(reset_collection=force_rebuild)
    manifest = load_manifest(vector_db_dir)
    settings_changed = manifest.get("settings") != current_settings

    if settings_changed and not force_rebuild:
        print("Embedding settings changed. Rebuilding the collection once for consistency.")
        vector_store, vector_db_dir, _ = create_vector_store(reset_collection=True)
        manifest = {"settings": current_settings, "files": {}}
    elif force_rebuild:
        manifest = {"settings": current_settings, "files": {}}

    manifest.setdefault("files", {})
    current_files = {raw_relative_source(path, raw_data_dir): path for path in raw_files}

    stale_sources = sorted(set(manifest["files"]) - set(current_files))
    for source in stale_sources:
        delete_ids(vector_store, manifest["files"][source].get("ids", []))
        del manifest["files"][source]
        print(f"Removed stale embeddings: {source}")

    changed_count = 0
    skipped_count = 0
    document_count = 0
    chunk_count = 0

    for source, path in sorted(current_files.items()):
        current_hash = file_hash(path)
        previous_entry = manifest["files"].get(source)
        if previous_entry and previous_entry.get("hash") == current_hash:
            skipped_count += 1
            continue

        if previous_entry:
            delete_ids(vector_store, previous_entry.get("ids", []))

        documents = load_file(path, raw_data_dir)
        if not documents:
            manifest["files"].pop(source, None)
            continue

        chunks = split_documents(documents)
        if not chunks:
            manifest["files"].pop(source, None)
            continue

        ids = add_chunks(vector_store, chunks)
        manifest["files"][source] = {
            "hash": current_hash,
            "ids": ids,
            "document_count": len(documents),
            "chunk_count": len(chunks),
        }
        changed_count += 1
        document_count += len(documents)
        chunk_count += len(chunks)
        print(f"Updated embeddings: {source} ({len(chunks)} chunks)")

    manifest["settings"] = current_settings
    save_manifest(vector_db_dir, manifest)
    print(
        "Training complete. "
        f"Updated {changed_count} files, skipped {skipped_count} unchanged files, "
        f"removed {len(stale_sources)} stale files, embedded {document_count} documents "
        f"into {chunk_count} chunks."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or refresh the local Chroma vector database from raw_data_files."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=resolve_path(os.getenv("RAW_DATA_DIR"), RAW_DATA_DIR),
        help="Folder containing source files to embed.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete and rebuild the whole Chroma collection.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.raw_data_dir.resolve(), force_rebuild=args.force_rebuild)
