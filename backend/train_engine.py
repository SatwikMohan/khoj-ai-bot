import config
import argparse
import csv
import hashlib
import io
import json
import os
import time
import threading
import shutil
from functools import lru_cache
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import chromadb
import docx2txt
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openpyxl import load_workbook
from pptx import Presentation

from services.embedding_service import PromptedOllamaEmbeddings, embedding_profile
from services.document_parsers import parse as parse_structured_file
from services.language_service import detect_language
from services.lexical_service import add_lexical_chunks, delete_lexical_ids, reset_lexical_collection
from services.versioning import version_metadata


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT
RAW_DATA_DIR = Path(config.RAW_DATA_DIR)
VECTOR_DB_DIR = Path(config.VECTOR_DB_DIR)
MANIFEST_FILE_NAME = "ingestion_manifest.json"
INGESTION_VERSION = config.INGESTION_VERSION
OCR_WARNING_KEYS: set[str] = set()
INGESTION_METRICS = threading.local()
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
IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def load_environment() -> None:
    config.configure_runtime_environment()


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
    profile = embedding_profile()
    return {
        "version": INGESTION_VERSION,
        "embedding_provider": "ollama",
        "embedding_model": profile.model,
        "embedding_profile": profile.as_dict(),
        "embedding_dimensions": int(config.OLLAMA_EMBED_DIMENSIONS),
        "ollama_base_url": config.OLLAMA_BASE_URL,
        "chunk_size": int(config.CHUNK_SIZE),
        "chunk_overlap": int(config.CHUNK_OVERLAP),
        "collection_name": config.CHROMA_COLLECTION_NAME,
        "ocr_enabled": ocr_enabled(),
        "ocr_lang": config.OCR_LANG,
        "ocr_pdf_dpi": int(config.OCR_PDF_DPI),
    }


def relative_source(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.name


def raw_relative_source(path: Path, raw_data_dir: Path = RAW_DATA_DIR) -> str:
    return path.relative_to(raw_data_dir).as_posix()


def base_metadata(path: Path, raw_data_dir: Path = RAW_DATA_DIR) -> dict:
    raw_relative_path = raw_relative_source(path, raw_data_dir)
    folder_path = Path(raw_relative_path).parent.as_posix()
    if folder_path == ".":
        folder_path = ""

    metadata = {
        "source": relative_source(path),
        "raw_source": raw_relative_path,
        "folder_path": folder_path,
        "folder_depth": 0 if not folder_path else len(Path(folder_path).parts),
        "file_name": path.name,
        "file_path": str(path),
        "extension": path.suffix.lower(),
    }
    metadata.update({key: value for key, value in version_metadata(Document(page_content="", metadata=metadata)).items() if key in {"year", "version"}})
    return metadata


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def ocr_enabled() -> bool:
    return config.OCR_ENABLED


@lru_cache(maxsize=1)
def ocr_engine_available() -> bool:
    executable = config.OCR_TESSERACT_EXECUTABLE
    resolved = shutil.which(executable) or (str(Path(executable)) if Path(executable).is_file() else None)
    if not resolved:
        return False
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = resolved
        installed = set(pytesseract.get_languages(config=""))
        required = {language for language in config.OCR_LANG.split("+") if language}
        if not required.issubset(installed):
            return False
    except ImportError:
        return False
    except Exception:
        return False
    return True


def warn_once(key: str, message: str) -> None:
    if key in OCR_WARNING_KEYS:
        return
    OCR_WARNING_KEYS.add(key)
    print(message)


def normalized_text_length(text: str) -> int:
    return len(" ".join((text or "").split()))


def ocr_image(image, source_label: str = "image") -> str:
    started = time.perf_counter()
    try:
        return _ocr_image_impl(image, source_label)
    finally:
        INGESTION_METRICS.ocr_seconds = getattr(INGESTION_METRICS, "ocr_seconds", 0.0) + time.perf_counter() - started


def _ocr_image_impl(image, source_label: str = "image") -> str:
    if not ocr_enabled():
        return ""
    if not ocr_engine_available():
        warn_once("ocr-engine-missing", "OCR unavailable: install local Tesseract with the configured language data before promoting an index.")
        return ""

    try:
        import pytesseract
        from PIL import ImageOps
    except ImportError as exc:
        warn_once(
            "ocr-python-deps",
            f"OCR skipped for {source_label}: install pytesseract and Pillow. {exc}",
        )
        return ""

    prepared_image = ImageOps.grayscale(image)
    INGESTION_METRICS.images = getattr(INGESTION_METRICS, "images", 0) + 1
    configured_lang = config.OCR_LANG.strip() or "eng"
    languages = [configured_lang]
    if configured_lang != "eng":
        languages.append("eng")

    tesseract_config = config.OCR_TESSERACT_CONFIG
    for index, language in enumerate(languages):
        try:
            return pytesseract.image_to_string(prepared_image, lang=language, config=tesseract_config).strip()
        except pytesseract.pytesseract.TesseractNotFoundError as exc:
            warn_once(
                "ocr-tesseract-missing",
                f"OCR skipped for {source_label}: Tesseract is not installed or not on PATH. {exc}",
            )
            return ""
        except pytesseract.pytesseract.TesseractError as exc:
            if index + 1 < len(languages):
                warn_once(
                    f"ocr-lang-{language}",
                    f"OCR language '{language}' failed for {source_label}; falling back to English. {exc}",
                )
                continue
            warn_once(
                f"ocr-error-{language}",
                f"OCR failed for {source_label} with language '{language}'. {exc}",
            )
            return ""

    return ""


def load_image(path: Path, raw_data_dir: Path) -> list[Document]:
    if ocr_enabled() and not ocr_engine_available():
        raise RuntimeError("Image indexing needs the local Tesseract executable and language data.")
    try:
        from PIL import Image
    except ImportError as exc:
        warn_once("ocr-pillow-missing", f"OCR skipped for {relative_source(path)}: install Pillow. {exc}")
        return []

    try:
        with Image.open(path) as image:
            text = ocr_image(image, relative_source(path))
    except Exception as exc:
        print(f"Could not OCR image {relative_source(path)}: {exc}")
        return []

    if not text.strip():
        print(f"Skipping image with no OCR text: {relative_source(path)}")
        return []

    metadata = base_metadata(path, raw_data_dir)
    metadata.update({"ocr": True, "ocr_source": "image", "language": detect_language(text)})
    return [Document(page_content=text, metadata=metadata)]


def load_pdf_ocr_pages(path: Path, raw_data_dir: Path, page_numbers: list[int] | None) -> list[Document]:
    if page_numbers == [] or not ocr_enabled():
        return []
    if not ocr_engine_available():
        warn_once("ocr-engine-missing", "OCR unavailable: install local Tesseract with the configured language data before promoting an index.")
        return []

    try:
        import fitz
        from PIL import Image
    except ImportError as exc:
        warn_once(
            "ocr-pdf-deps",
            f"PDF OCR skipped for {relative_source(path)}: install PyMuPDF and Pillow. {exc}",
        )
        return []

    documents = []
    dpi = int(config.OCR_PDF_DPI)
    try:
        with fitz.open(str(path)) as pdf:
            if page_numbers is None:
                page_numbers = list(range(pdf.page_count))
            for page_number in page_numbers:
                if page_number < 0 or page_number >= pdf.page_count:
                    continue

                page = pdf.load_page(page_number)
                pixmap = page.get_pixmap(dpi=dpi, alpha=False)
                with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
                    text = ocr_image(image, f"{relative_source(path)} page {page_number + 1}")

                if not text.strip():
                    continue

                metadata = base_metadata(path, raw_data_dir)
                metadata.update(
                    {
                        "page": page_number,
                        "ocr": True,
                        "ocr_source": "pdf_page",
                        "language": detect_language(text),
                    }
                )
                documents.append(Document(page_content=text, metadata=metadata))
    except Exception as exc:
        print(f"Could not OCR PDF {relative_source(path)}: {exc}")

    return documents


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
    min_text_chars = int(config.MIN_EXTRACTED_TEXT_CHARS)
    try:
        loaded_documents = PyPDFLoader(str(path)).load()
    except Exception as exc:
        print(f"Could not extract PDF text from {relative_source(path)}; trying OCR. {exc}")
        return load_pdf_ocr_pages(path, raw_data_dir, None)

    low_text_pages = []
    for fallback_page_number, document in enumerate(loaded_documents):
        try:
            page_number = int(document.metadata.get("page", fallback_page_number))
        except (TypeError, ValueError):
            page_number = fallback_page_number
        if normalized_text_length(document.page_content) < min_text_chars:
            low_text_pages.append(page_number)
            continue
        document.metadata.update(base_metadata(path, raw_data_dir))
        document.metadata["language"] = detect_language(document.page_content)
        documents.append(document)

    documents.extend(load_pdf_ocr_pages(path, raw_data_dir, low_text_pages))
    if config.OCR_EMBEDDED_IMAGES and ocr_enabled() and ocr_engine_available():
        try:
            import fitz
            from PIL import Image
            with fitz.open(str(path)) as pdf:
                seen_images = set()
                for page_number, page in enumerate(pdf):
                    if page_number in low_text_pages:
                        continue  # Full-page OCR already covers these images.
                    native = next((doc.page_content for doc in documents if doc.metadata.get("page") == page_number and not doc.metadata.get("ocr")), "")
                    for image_index, image_info in enumerate(page.get_images(full=True)[:int(config.OCR_MAX_IMAGES_PER_PAGE)]):
                        xref = image_info[0]
                        if xref in seen_images:
                            continue
                        seen_images.add(xref)
                        with Image.open(io.BytesIO(pdf.extract_image(xref)["image"])) as image:
                            if image.width * image.height < int(config.OCR_MIN_IMAGE_PIXELS):
                                continue
                            text = ocr_image(image, f"{relative_source(path)} page {page_number + 1} image {image_index + 1}")
                        normalized = " ".join(text.casefold().split())
                        if normalized and normalized not in " ".join(native.casefold().split()):
                            metadata = base_metadata(path, raw_data_dir)
                            metadata.update(page=page_number, image_index=image_index, ocr=True, ocr_source="pdf_embedded")
                            metadata["language"] = detect_language(text)
                            documents.append(Document(page_content=text, metadata=metadata))
        except Exception as exc:
            print(f"Embedded image OCR failed for {relative_source(path)}: {exc}")
    return documents


def load_file(path: Path, raw_data_dir: Path) -> list[Document]:
    suffix = path.suffix.lower()
    metadata = base_metadata(path, raw_data_dir)

    if suffix == ".pdf":
        return load_pdf(path, raw_data_dir)
    if suffix in IMAGE_EXTENSIONS:
        return load_image(path, raw_data_dir)
    return parse_structured_file(path, metadata, ocr_image)


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
    chunks = []
    for document in documents:
        if document.metadata.get("element_type") == "table" and len(document.page_content) > settings["chunk_size"]:
            lines = document.page_content.splitlines()
            header_count = 2 if lines and lines[0].startswith("Sheet:") else 1
            header = lines[:header_count]
            batch = []
            for line in lines[header_count:]:
                if batch and len("\n".join(header + batch + [line])) > settings["chunk_size"]:
                    chunks.extend(splitter.split_documents([Document(page_content="\n".join(header + batch), metadata=document.metadata.copy())]))
                    batch = []
                batch.append(line)
            if batch:
                chunks.extend(splitter.split_documents([Document(page_content="\n".join(header + batch), metadata=document.metadata.copy())]))
        else:
            chunks.extend(splitter.split_documents([document]))
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["language"] = detect_language(chunk.page_content)
    return chunks


def chunk_id(chunk: Document) -> str:
    identity = {
        "source": chunk.metadata.get("source"),
        "year": chunk.metadata.get("year"),
        "version": chunk.metadata.get("version"),
        "effective_date": chunk.metadata.get("effective_date"),
        "revision": chunk.metadata.get("revision"),
        "catalog": chunk.metadata.get("catalog"),
        "schema": chunk.metadata.get("schema"),
        "table": chunk.metadata.get("table"),
        "dataset": chunk.metadata.get("dataset"),
        "entity": chunk.metadata.get("entity"),
        "page": chunk.metadata.get("page"),
        "element_type": chunk.metadata.get("element_type"),
        "heading": chunk.metadata.get("heading"),
        "sheet": chunk.metadata.get("sheet"),
        "slide": chunk.metadata.get("slide"),
        "symbol": chunk.metadata.get("symbol"),
        "row_start": chunk.metadata.get("row_start"),
        "content": chunk.page_content,
    }
    return hashlib.sha1(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def manifest_path(vector_db_dir: Path) -> Path:
    return vector_db_dir / MANIFEST_FILE_NAME


def load_manifest(vector_db_dir: Path) -> dict:
    path = manifest_path(vector_db_dir)
    if not path.exists():
        return {"settings": {}, "files": {}}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"settings": {}, "files": {}}


def save_manifest(vector_db_dir: Path, manifest: dict) -> None:
    path = manifest_path(vector_db_dir)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)


def create_vector_store(
    collection_name: str | None = None,
    reset_collection: bool = False,
) -> tuple[Chroma, Path, str]:
    collection_name = collection_name or config.CHROMA_COLLECTION_NAME
    vector_db_dir = resolve_path(config.VECTOR_DB_DIR, VECTOR_DB_DIR)

    vector_db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(vector_db_dir))
    if reset_collection:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    profile = embedding_profile()
    embeddings = PromptedOllamaEmbeddings(
        profile=profile,
        base_url=config.OLLAMA_BASE_URL,
        keep_alive=int(config.OLLAMA_KEEP_ALIVE),
    )
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata={
            "hnsw:space": "cosine",
            "embedding_model": profile.model,
            "embedding_profile": profile.fingerprint,
            "embedding_dimensions": int(config.OLLAMA_EMBED_DIMENSIONS),
            "ingestion_version": INGESTION_VERSION,
        },
    )
    return vector_store, vector_db_dir, collection_name


def delete_ids(vector_store: Chroma, ids: list[str]) -> None:
    if ids:
        vector_store.delete(ids=ids)


def add_chunks(
    vector_store: Chroma,
    vector_db_dir: Path,
    collection_name: str,
    chunks: list[Document],
) -> list[str]:
    batch_size = int(config.EMBEDDING_BATCH_SIZE)
    ids = [chunk_id(chunk) for chunk in chunks]
    embedding_ms = 0.0
    insertion_ms = 0.0

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        batch_started = time.perf_counter()
        vector_store.add_documents(chunks[start:end], ids=ids[start:end])
        batch_total_ms = (time.perf_counter() - batch_started) * 1000
        batch_embedding_ms = getattr(vector_store._embedding_function, "last_documents_ms", 0.0)
        embedding_ms += batch_embedding_ms
        insertion_ms += max(0.0, batch_total_ms - batch_embedding_ms)
        print(f"Embedded chunks {start + 1}-{min(end, len(chunks))} of {len(chunks)}")
    add_lexical_chunks(vector_db_dir, collection_name, chunks, ids)
    INGESTION_METRICS.last_add = {"embedding_ms": round(embedding_ms, 1), "vector_insertion_ms": round(insertion_ms, 1)}
    return ids


def versioned_collection_name(base_name: str, settings: dict, force_rebuild: bool) -> str:
    settings_fingerprint = hashlib.sha1(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    name = f"{base_name}__{settings_fingerprint}__v{INGESTION_VERSION}"
    if force_rebuild:
        name = f"{name}__{int(time.time())}"
    return name[:63]


def prepare_file(
    path: Path,
    raw_data_dir: Path,
) -> tuple[list[Document], list[Document], str | None, dict]:
    INGESTION_METRICS.ocr_seconds = 0.0
    INGESTION_METRICS.images = 0
    started = time.perf_counter()
    try:
        documents = load_file(path, raw_data_dir)
        parsed_at = time.perf_counter()
        chunks = split_documents(documents) if documents else []
        return documents, chunks, None, {
            "parsing_ms": round((parsed_at - started) * 1000, 1),
            "text_extraction_ms": round(max(0, parsed_at - started - INGESTION_METRICS.ocr_seconds) * 1000, 1),
            "ocr_ms": round(INGESTION_METRICS.ocr_seconds * 1000, 1),
            "chunking_ms": round((time.perf_counter() - parsed_at) * 1000, 1),
            "images_processed": INGESTION_METRICS.images,
            "pages_processed": len({doc.metadata.get("page") for doc in documents if doc.metadata.get("page") is not None}),
        }
    except Exception as exc:
        return [], [], str(exc), {"parsing_ms": round((time.perf_counter() - started) * 1000, 1)}


def iter_prepared_files(work_items: list[tuple], raw_data_dir: Path):
    workers = max(1, min(8, int(config.INGESTION_WORKERS)))
    if workers == 1:
        for item in work_items:
            yield item, prepare_file(item[1], raw_data_dir)
        return

    iterator = iter(work_items)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="texmin-ingest") as executor:
        in_flight = {}
        for _ in range(workers * 2):
            try:
                item = next(iterator)
            except StopIteration:
                break
            in_flight[executor.submit(prepare_file, item[1], raw_data_dir)] = item

        while in_flight:
            completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                item = in_flight.pop(future)
                yield item, future.result()
                try:
                    next_item = next(iterator)
                except StopIteration:
                    continue
                in_flight[executor.submit(prepare_file, next_item[1], raw_data_dir)] = next_item


def train(raw_data_dir: Path, force_rebuild: bool = False) -> None:
    training_started = time.perf_counter()
    load_environment()
    raw_data_dir = raw_data_dir.resolve()

    raw_files = iter_raw_files(raw_data_dir)
    if not raw_files:
        raise RuntimeError(f"No supported documents found in {raw_data_dir}")

    current_settings = chunk_settings()
    vector_db_dir = resolve_path(config.VECTOR_DB_DIR, VECTOR_DB_DIR)
    vector_db_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(vector_db_dir)
    active_manifest = manifest
    settings_changed = manifest.get("settings") != current_settings
    base_collection_name = config.CHROMA_COLLECTION_NAME
    building_new_collection = force_rebuild or settings_changed

    if building_new_collection:
        saved_candidate = manifest.get("candidate") if not force_rebuild else None
        resume_candidate = (saved_candidate and saved_candidate.get("settings") == current_settings
                            and saved_candidate.get("ocr_available_at_build") == ocr_engine_available())
        collection_name = (saved_candidate["active_collection"] if resume_candidate else
                           versioned_collection_name(base_collection_name, current_settings, force_rebuild))
        print(
            "Building a new collection before switching traffic: "
            f"{collection_name}. The current collection remains available."
        )
        vector_store, vector_db_dir, _ = create_vector_store(
            collection_name=collection_name,
            reset_collection=not resume_candidate,
        )
        if not resume_candidate:
            reset_lexical_collection(vector_db_dir, collection_name)
        manifest = saved_candidate if resume_candidate else {
            "schema_version": 2,
            "settings": current_settings,
            "active_collection": collection_name,
            "files": {},
            "ocr_available_at_build": ocr_engine_available(),
        }
    else:
        collection_name = manifest.get("active_collection") or base_collection_name
        vector_store, vector_db_dir, _ = create_vector_store(collection_name=collection_name)

    manifest.setdefault("files", {})
    manifest.setdefault("failures", {})
    current_files = {raw_relative_source(path, raw_data_dir): path for path in raw_files}

    stale_sources = sorted(set(manifest["files"]) - set(current_files))
    for source in stale_sources:
        delete_ids(vector_store, manifest["files"][source].get("ids", []))
        delete_lexical_ids(
            vector_db_dir,
            collection_name,
            manifest["files"][source].get("ids", []),
        )
        del manifest["files"][source]
        print(f"Removed stale embeddings: {source}")

    changed_count = 0
    skipped_count = 0
    document_count = 0
    chunk_count = 0
    failed_count = 0
    embeddings_generated = 0
    duplicates_skipped = 0
    stage_totals_ms = {key: 0.0 for key in ("parsing_ms", "ocr_ms", "text_extraction_ms", "chunking_ms", "embedding_ms", "vector_insertion_ms")}
    work_items = []

    for source, path in sorted(current_files.items()):
        previous_entry = manifest["files"].get(source)
        stat = path.stat()
        if (
            previous_entry
            and config.FAST_FILE_CHECK
            and previous_entry.get("size") == stat.st_size
            and previous_entry.get("mtime_ns") == stat.st_mtime_ns
        ):
            skipped_count += 1
            continue

        current_hash = file_hash(path)
        if previous_entry and previous_entry.get("hash") == current_hash:
            previous_entry["size"] = stat.st_size
            previous_entry["mtime_ns"] = stat.st_mtime_ns
            skipped_count += 1
            continue

        work_items.append((source, path, current_hash, stat.st_size, stat.st_mtime_ns))

    print(
        f"Preparing {len(work_items)} changed files with "
        f"{max(1, min(8, int(config.INGESTION_WORKERS)))} extraction workers."
    )
    for item, prepared in iter_prepared_files(work_items, raw_data_dir):
        source, path, current_hash, file_size, mtime_ns = item
        documents, chunks, preparation_error, metrics = prepared
        previous_entry = manifest["files"].get(source)

        if preparation_error:
            failed_count += 1
            manifest["failures"][source] = preparation_error
            print(f"Could not prepare {source}: {preparation_error}")
            continue
        manifest["failures"].pop(source, None)

        if not documents:
            manifest["files"].pop(source, None)
            continue

        if not chunks:
            manifest["files"].pop(source, None)
            continue

        unique_chunks = {}
        for chunk in chunks:
            unique_chunks.setdefault(chunk_id(chunk), chunk)
        duplicate_count = len(chunks) - len(unique_chunks)
        previous_ids = set(previous_entry.get("ids", [])) if previous_entry else set()
        ids = list(unique_chunks)
        added_ids = [identity for identity in ids if identity not in previous_ids]
        obsolete_ids = sorted(previous_ids - set(ids))
        try:
            if added_ids:
                add_chunks(vector_store, vector_db_dir, collection_name, [unique_chunks[identity] for identity in added_ids])
            else:
                INGESTION_METRICS.last_add = {"embedding_ms": 0.0, "vector_insertion_ms": 0.0}
        except Exception:
            delete_ids(vector_store, added_ids)
            delete_lexical_ids(vector_db_dir, collection_name, added_ids)
            raise
        if obsolete_ids:
            delete_ids(vector_store, obsolete_ids)
            delete_lexical_ids(vector_db_dir, collection_name, obsolete_ids)
        metrics.update(INGESTION_METRICS.last_add)
        embeddings_generated += len(added_ids)
        duplicates_skipped += duplicate_count
        for key in stage_totals_ms:
            stage_totals_ms[key] += float(metrics.get(key, 0.0))
        metrics.update(source=source, documents=len(documents), chunks=len(ids),
                       embeddings_generated=len(added_ids), duplicate_chunks_skipped=duplicate_count,
                       elapsed_ms=round(sum(value for key, value in metrics.items() if key in {"parsing_ms", "chunking_ms", "embedding_ms", "vector_insertion_ms"}), 1))
        print(json.dumps({"event": "index_file_completed", **metrics}, ensure_ascii=False), flush=True)
        manifest["files"][source] = {
            "hash": current_hash,
            "size": file_size,
            "mtime_ns": mtime_ns,
            "ids": ids,
            "document_count": len(documents),
            "chunk_count": len(ids),
        }
        changed_count += 1
        document_count += len(documents)
        chunk_count += len(ids)
        print(f"Updated embeddings: {source} ({len(ids)} chunks, {len(added_ids)} new embeddings)")
        if building_new_collection:
            active_manifest["candidate"] = manifest
            save_manifest(vector_db_dir, active_manifest)
        else:
            manifest["settings"] = current_settings
            manifest["schema_version"] = 2
            manifest["active_collection"] = collection_name
            save_manifest(vector_db_dir, manifest)

    if building_new_collection:
        active_manifest["candidate"] = manifest
        save_manifest(vector_db_dir, active_manifest)
    if building_new_collection and failed_count:
        raise RuntimeError(
            f"The candidate index has {failed_count} failed source files. "
            "Fix the reported extraction errors and rerun ingestion."
        )

    if building_new_collection:
        print(json.dumps({"event": "index_run_completed", "candidate": True,
                          "total_ms": round((time.perf_counter() - training_started) * 1000, 1),
                          "files_updated": changed_count, "files_unchanged": skipped_count,
                          "chunks": chunk_count, "embeddings_generated": embeddings_generated,
                          "duplicate_chunks_skipped": duplicates_skipped,
                          **{key: round(value, 1) for key, value in stage_totals_ms.items()}}, flush=True))
        if not active_manifest.get("active_collection"):
            manifest.pop("candidate", None)
            save_manifest(vector_db_dir, manifest)
            print(f"Initial index {collection_name} activated after complete ingestion.")
            return
        print(f"Candidate collection {collection_name} is ready. The active collection is unchanged. "
              "Run train_engine.py --promote-candidate after checking the evaluation result.")
        return

    manifest["settings"] = current_settings
    manifest["schema_version"] = 2
    manifest["active_collection"] = collection_name
    save_manifest(vector_db_dir, manifest)
    print(
        "Training complete. "
        f"Updated {changed_count} files, skipped {skipped_count} unchanged files, "
        f"failed {failed_count} files, removed {len(stale_sources)} stale files, "
        f"embedded {document_count} documents "
        f"into {chunk_count} chunks."
    )
    print(json.dumps({"event": "index_run_completed", "candidate": False,
                      "total_ms": round((time.perf_counter() - training_started) * 1000, 1),
                      "files_updated": changed_count, "files_unchanged": skipped_count,
                      "chunks": chunk_count, "embeddings_generated": embeddings_generated,
                      "duplicate_chunks_skipped": duplicates_skipped,
                      **{key: round(value, 1) for key, value in stage_totals_ms.items()}}, flush=True))


def _retrieval_recall(vector_store: Chroma, collection_name: str, cases: list[dict], top_k: int = 5) -> float:
    from services.qa_service import _retrieve_context, _source_chunks

    scores = []
    for case in cases:
        docs = _retrieve_context(vector_store, case["question"], top_k,
                                 lexical_collection_name=collection_name)
        sources = [source.source or "" for source in _source_chunks(docs)]
        expected = case["expected_sources"]
        scores.append(sum(any(name.casefold() in source.casefold() for source in sources)
                          for name in expected) / len(expected))
    return sum(scores) / len(scores)


def promote_candidate() -> None:
    """Activate only a complete, dimensionally compatible, non-regressing index."""
    load_environment()
    vector_db_dir = resolve_path(config.VECTOR_DB_DIR, VECTOR_DB_DIR)
    active = load_manifest(vector_db_dir)
    candidate = active.get("candidate")
    if not candidate or candidate.get("settings") != chunk_settings():
        raise RuntimeError("No staged candidate matches the current embedding and ingestion settings.")
    if candidate.get("failures") or not candidate.get("files"):
        raise RuntimeError("The candidate has failed or missing source files; rerun ingestion first.")
    if config.OCR_ENABLED and not candidate.get("ocr_available_at_build"):
        raise RuntimeError("Candidate was built without local OCR. Install Tesseract and rebuild the candidate before promotion.")
    collection_name = candidate["active_collection"]
    collection = chromadb.PersistentClient(path=str(vector_db_dir)).get_collection(collection_name)
    expected_vectors = sum(len(item.get("ids", [])) for item in candidate["files"].values())
    if collection.count() != expected_vectors:
        raise RuntimeError(f"Candidate vector count mismatch: {collection.count()} versus {expected_vectors} in the manifest.")
    peek = collection.peek(1)
    if not len(peek.get("embeddings", [])) or len(peek["embeddings"][0]) != int(config.OLLAMA_EMBED_DIMENSIONS):
        raise RuntimeError("Candidate embedding dimensions do not match the configured model.")
    evaluation_file = Path(config.INDEX_PROMOTION_EVAL_FILE)
    if not evaluation_file.is_file():
        raise RuntimeError(f"Promotion evaluation data is missing: {evaluation_file}")
    from evaluate_retrieval import load_cases
    cases = load_cases(evaluation_file)
    if not cases:
        raise RuntimeError("Promotion evaluation data has no labelled questions.")
    candidate_store, _, _ = create_vector_store(collection_name=collection_name)
    candidate_recall = _retrieval_recall(candidate_store, collection_name, cases)
    minimum = float(config.INDEX_MIN_RECALL_AT_5)
    old_collection = active.get("active_collection")
    old_profile = active.get("settings", {}).get("embedding_profile", {}).get("fingerprint")
    if old_collection and old_profile == candidate["settings"]["embedding_profile"]["fingerprint"]:
        old_store, _, _ = create_vector_store(collection_name=old_collection)
        old_recall = _retrieval_recall(old_store, old_collection, cases)
        print(f"Retrieval recall@5: active={old_recall:.4f}, candidate={candidate_recall:.4f}")
        minimum = max(minimum, old_recall)
    else:
        print(f"Retrieval recall@5: candidate={candidate_recall:.4f}; old embedding space differs or no active index exists.")
    if candidate_recall + 1e-9 < minimum:
        raise RuntimeError(f"Candidate recall@5 {candidate_recall:.4f} is below the promotion threshold {minimum:.4f}.")
    promoted = {key: value for key, value in candidate.items() if key != "candidate"}
    promoted["previous_collection"] = old_collection
    save_manifest(vector_db_dir, promoted)
    print(f"Promoted {collection_name}; previous collection {old_collection} remains on disk.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or refresh the local Chroma vector database from raw_data_files."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=resolve_path(config.RAW_DATA_DIR, RAW_DATA_DIR),
        help="Folder containing source files to embed.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Build a fresh versioned candidate collection without changing the active index.",
    )
    parser.add_argument("--promote-candidate", action="store_true",
                        help="Validate and activate the staged candidate collection.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.promote_candidate:
        promote_candidate()
    else:
        train(args.raw_data_dir.resolve(), force_rebuild=args.force_rebuild)
