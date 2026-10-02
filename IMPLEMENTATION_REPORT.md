# Local voice, indexing, and model implementation report

Measured on the Windows development host on 2 October 2026. Times are from
individual local runs unless a median is stated. This report distinguishes code
coverage from behavior verified on this machine.

For the later DGX Spark deployment target and fresh same-day measurements, see
[`DGX_SPARK_REEVALUATION.md`](DGX_SPARK_REEVALUATION.md). The measurements below
describe the Windows 3B/768-dimensional baseline. The active config now names
the DGX trial models, which still need validation on the actual device.

## 1. Existing architecture and bottlenecks

The Streamlit frontend calls the FastAPI QA and speech services in-process by
default, or over HTTP when configured. The QA service uses Ollama for local
`llama3.2:latest` generation and `nomic-embed-text` embeddings, Chroma for dense
retrieval, and a local lexical index for hybrid retrieval. SSE already emitted
answer tokens, but the frontend waited for the final answer before calling
Piper. A single global response semaphore also serialized unrelated sessions.
The indexer hashed files and used versioned Chroma collections, but flattened
Office files, re-embedded every chunk of a changed file, and lacked a measured
candidate promotion gate.

## 2. Root causes and implementation decisions

The wait for complete text caused voice latency. Speech now starts at a sentence
or bounded text segment while generation continues. A persistent Piper process
per voice avoids paying model startup on each segment. Per-session request IDs
and cancellation events replace global response serialization. Structured parser
outputs and stable chunk IDs preserve source relationships and reuse unchanged
vectors. A candidate collection remains staged until OCR readiness, integrity,
dimension, and labelled recall checks pass.

## 3–5. Files changed, created, and removed

Changed: `Dockerfile`, `backend/Dockerfile`, `backend/config.py`,
`backend/evaluate_retrieval.py`, `backend/helpers/request_models.py`,
`backend/main.py`, `backend/requirements.txt`,
`backend/routes/qa_routes.py`, `backend/services/embedding_service.py`,
`backend/services/language_service.py`, `backend/services/piper_worker.py`,
`backend/services/qa_service.py`, `backend/services/reranker_service.py`,
`backend/services/response_stream.py`, `backend/services/tts_service.py`,
`backend/tests/test_response_delivery.py`, `backend/tests/test_shared_config.py`,
`backend/train_engine.py`, `backend/evals/README.md`, `frontend/app.py`, and
`frontend/voice_query_component/index.html`.

Created: `backend/services/request_lifecycle.py`,
`backend/services/speech_chunker.py`, `backend/services/audio_chunks.py`,
`backend/services/document_parsers.py`,
`backend/tests/test_structured_ingestion.py`,
`backend/evals/golden_local.jsonl`, `backend/evals/benchmark_voice.py`,
`backend/evals/benchmark_ingestion.py`, and this report. Removed: none.

## 6. Streaming voice

The existing RAG prompt and Ollama `.stream()` call are retained. SSE delivers
tokens; Streamlit updates visible text and chat history while an incremental
segmenter sends meaningful segments to a bounded TTS worker pool. Audio clips
are broadcast with monotonically increasing sequence numbers. The browser
plays them in order without overlapping them; a failed clip is skipped. Final
incomplete text is flushed. Language detection selects an installed English or
Hindi Piper voice per segment. Piper synthesizes complete WAV segments, so this
is **chunked synthesis and playback**, not sample-level streaming TTS. The
frontend keeps only the clips needed for current delivery and optional history
replay; it does not wait for the complete answer to play the first clip.

## 7. Request interruption

Each QA request has a session ID, request ID, timestamp, status, and cancellation
event. A new request cancels the old request in that session. The QA stream
checks ownership around retrieval, token delivery, and final output; the worker
closes abandoned Ollama streams. Retrieval futures stop waiting on cancellation,
and queued Piper work is cancelled. An active Piper synthesis subprocess is
terminated when its request becomes stale. The browser stops current audio and
discards its queue on interruption. The microphone component also calls the
session cancellation endpoint. Other sessions have independent registry entries.
The older `/qa/ask`, `/qa/ask/stream`, and `/tts/speech` contracts remain usable;
session and request IDs are optional for older clients.

Cancellation is cooperative around Ollama and Chroma calls: a provider operation
already inside a blocking call may finish before its next cancellation check.
Its result is discarded. Immediate backend cancellation from the browser is
available when FastAPI is reachable; a standalone in-process Streamlit process
without that HTTP endpoint relies on Streamlit rerun and request ownership.

## 8–10. Ingestion, formats, OCR, and languages

The parser registry keeps sections, headings, page/slide numbers, tables, sheet
and column headers, row ranges, hierarchy paths, code symbols and line numbers,
image origin, and source metadata. Prose, tables, and code are chunked along
their logical boundaries before the configured size/overlap splitter is used.
Chunk language metadata is derived locally, preserving Unicode and original
text. Source hashes skip unchanged files. Stable chunk IDs let a changed file
reuse matching vectors and remove obsolete vectors. Two extraction workers and
batched embeddings limit memory use. Per-file logs include parsing, OCR,
extraction, chunking, embedding, insertion, page/image/chunk counts, duplicate
counts, and a run total in new runs.

Supported formats: PDF; DOC, DOCX, RTF, ODT; TXT, Markdown, reStructuredText,
AsciiDoc, HTML, XML; JSON, JSONL, YAML, TOML, INI/CFG, IPYNB; XLS, XLSX, XLSM,
CSV, TSV, ODS; PPT, PPTX, ODP; PNG, JPEG, TIFF, BMP, WEBP, GIF; Python, Java,
JavaScript/JSX, TypeScript/TSX, CSS, SQL, C/C++ headers and sources, Go, Rust,
shell, PowerShell, batch, and LOG. DOC/XLS/PPT conversion needs local
LibreOffice, now included in both Dockerfiles. RTF needs `striprtf`, now in
requirements. Unknown extensions fail with an explicit error. Audio and video
are excluded.

For PDF, pages with enough selectable text retain native text; low-text pages
are rasterized for OCR. Embedded PDF and Office images are OCRed and deduplicated
against native or previously extracted text. Tesseract with `eng+hin` is the
local OCR choice for this CPU/RAM budget; its language packs must be installed
locally. Page and image references are preserved. It does not extract image
bounding boxes or reliably recover complex visual relationships. OCR on the
native Windows host was unavailable; scanned/embedded-image tests use mocked
OCR. The Docker images install Tesseract English/Hindi and LibreOffice.

## 11–13. Model evaluation and selected configuration

Host: Windows 10 build 19045, Intel family 6 model 140 CPU, 4 physical/8
logical cores, 7.7 GiB RAM (about 1.7 GiB free at inspection), no accessible
CUDA GPU/VRAM, and about 387 GiB free on E:. Ollama is the installed runtime.
Only `llama3.2:latest` (3.2B, Q4_K_M, roughly 2 GB weights) and
`nomic-embed-text:latest` (137M, F16, roughly 274 MB, 768 dimensions) were
installed. The active Chroma collection uses cosine distance and has 1,778
vectors across 11 files (8 DOCX, 3 PDFs).

The selected non-reasoning chat model remains Llama 3.2 3B Q4_K_M. Its local
footprint fits this host; the measured response below was grounded in the
source DOCX. Llama 3.1 8B (about 4.9 GB weights) would leave little RAM for
Ollama context, STT, and app services on this machine. Llama 3.3 70B (about
43 GB weights) is impractical here. Official model listings:
[Llama 3.2](https://ollama.com/library/llama3.2),
[Llama 3.1](https://ollama.com/library/llama3.1),
[Llama 3.3](https://ollama.com/library/llama3.3).
The configuration sets 8,192 context tokens, 768 maximum output tokens,
temperature 0.3, disabled thinking, and Ollama keep-alive 1,800 seconds.

The selected embedding model remains the installed 768-dimensional
`nomic-embed-text`; this preserves the active vector space and keeps memory
use low. Its published context is 2,000 tokens
([Ollama listing](https://ollama.com/library/nomic-embed-text)). The active
seven-question local set achieved recall@5 6/7, precision@5 0.1714,
MRR 0.8571, and nDCG@5 0.8571. The Hindi cross-language safety question missed.
This is evidence of a gap, not evidence of strong multilingual retrieval.
The staged structured candidate also reached 6/7 on the same labels.

Multilingual candidates deserve a separate controlled test when installed:
[BGE-M3](https://huggingface.co/BAAI/bge-m3) supports multilingual dense,
sparse, and multi-vector representations with 1,024-dimensional dense vectors;
the current Ollama–Chroma path would use dense vectors only.
[Multilingual E5 large](https://huggingface.co/intfloat/multilingual-e5-large)
also uses 1,024 dimensions. [Nomic Embed Text v2 MoE](https://ollama.com/library/nomic-embed-text-v2-moe)
is a smaller multilingual option listed at 768 dimensions. None was installed
or benchmarked here; no quality ranking between them is claimed. Changing the
embedding model requires a new collection and labelled validation. Configured
name/path, dimensions, batch size, model loading, context, output, temperature,
stream heartbeat, ingestion workers, and speech concurrency live in
`backend/config.py`. The absent reranker is disabled to avoid loading weights
or network access at runtime.

## 14. Measurements

| Measure | Early chunked run | Prewarmed persistent Piper run | Interpretation |
|---|---:|---:|---|
| First playable speech from submission, one live query | 30,549 ms | 17,814 ms | Distinct runs of the new chunked path, not a sequential baseline or controlled paired trial. |
| First LLM token from submission | 13,670 ms | 14,146 ms | Includes retrieval and cold/warm runtime variation. |
| Full LLM completion | 41,277 ms | 36,352 ms | First audio was ready before completion in both experimental chunked runs. |
| Direct Piper short clip, first then warm reuse | 3,938 ms | 109 ms | Same installed English voice; process reuse. |
| One DOCX median extraction (3 runs) | 40.5 ms | 21.7 ms | `UG Mines .docx`, 67,617 bytes. |
| Same DOCX chunking/language tagging median | 3.3 ms | 97.2 ms | Richer preparation is slower. |
| Same DOCX chunks | 180 | 154 | About 14% fewer chunks to embed. |
| Same DOCX unchanged hash check median | — | 0.6 ms | Parsing, OCR, and embeddings skipped on unchanged rerun. |

The original sequential voice path was not timed, so no measured sequential-to-
chunked speedup is claimed. In the prewarmed live voice run, TTS warmup took
10,797 ms before submission;
RAG took 9,041 ms; generation's first token took another 3,412 ms; first
speech segment was submitted at 16,981 ms; first playable WAV was ready at
17,814 ms; the 795-character answer finished at 36,352 ms. The first audible
browser playback time is logged by JavaScript but was not measured headlessly.
The source-grounded BWE answer run took 5,997 ms retrieval and 33,211 ms total.

The candidate index has 1,585 vectors and matched the same 6/7 recall@5 as
the active 1,778-vector index. A repeated active-index retrieval run had p50
latency 205.7 ms, maximum 4,171.5 ms; one earlier cold query took about 30 s
while an absent reranker was being loaded, motivating its disablement. A full
legacy-versus-new indexing wall-time comparison was not captured, so no total
speedup is asserted. New run and per-file logs provide that measurement for
future builds. Chat tokens/second and process RAM/VRAM during generation were
not instrumented, and no alternative chat or embedding model was installed to
support a comparative quality score.

## 15. Verification

`python -m unittest discover -s backend/tests -q` passed 87 tests. Python
`compileall` passed; `git diff --check` found no whitespace errors. Tests cover
segment boundaries, WAV joining, cancellation isolation/stale tokens,
structured Office/table/code/JSON parsing, Hindi/mixed text, unsupported
formats, and mocked scanned PDF OCR. A live Ollama-plus-Piper benchmark showed
first playable audio before complete LLM generation. A real RAG answer was
checked against its BWE DOCX source. Browser microphone and audio playback
were not exercised in an automated browser session on this host.

## 16. Remaining limitations and index state

The staged candidate `texmin_qa__82b087481fe8__v8` was built before OCR
availability metadata was added and on a Windows host without local
Tesseract. Promotion correctly rejected it. The active collection remains
`texmin_qa__eb33bc6823f8__v6`, so retrieval quality in production remains
the old index until a complete OCR-capable candidate passes checks. Candidate
labels identify filenames, not answer spans, and the corpus lacks scanned,
spreadsheet, code, and versioned examples. OCR accuracy, chat faithfulness
across languages, browser first-audio latency, and true provider-side instant
interruption need further real-world measurement.

## 17. Initial offline setup and promotion

Install Python requirements from `backend/requirements.txt`. Provision the
configured chat/embedding models with Ollama and the configured Whisper and
Piper files in a separate connected setup phase using
`backend/scripts/provision_models.py`; runtime is configured offline and must
find all model files locally. The Piper English/Hindi voice files and both
Ollama models were already present on this host. The Whisper model cache was
not present, so offline STT needs provisioning before use.

For native Windows ingestion, install local Tesseract with English and Hindi
trained data and set `OCR_TESSERACT_EXECUTABLE` in `backend/config.py` if it is
not on `PATH`. Install LibreOffice for DOC/XLS/PPT. Docker images include those
tools. Then, from the repository root:

```powershell
$env:PYTHONPATH = 'backend'
.\.venv\Scripts\python.exe backend/train_engine.py --force-rebuild
.\.venv\Scripts\python.exe backend/evaluate_retrieval.py backend/evals/golden_local.jsonl --top-k 5
.\.venv\Scripts\python.exe backend/train_engine.py --promote-candidate
```

The promotion command validates the candidate count, dimension, OCR
availability, and recall gate before switching the active manifest. The old
collection remains on disk for rollback or comparison.
