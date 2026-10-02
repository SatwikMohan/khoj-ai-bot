# Local evaluation and promotion

`golden_local.jsonl` contains seven labelled questions drawn from the current
11-file mining corpus. Labels are source filenames; the Hindi question tests
cross-language retrieval. These labels do not prove chunk-level answer quality.
Add scanned pages, spreadsheets, code, multiple versions, and more Hindi content
as those sources are added to the corpus.

From the repository root, with dependencies and Ollama models installed:

```powershell
$env:PYTHONPATH = 'backend'
.\.venv\Scripts\python.exe backend/evaluate_retrieval.py backend/evals/golden_local.jsonl --top-k 5
.\.venv\Scripts\python.exe backend/evals/benchmark_voice.py
.\.venv\Scripts\python.exe backend/evals/benchmark_chat_runtime.py llama3.2:latest --output backend/evals/chat_runtime.json
.\.venv\Scripts\python.exe backend/evals/benchmark_ingestion.py 'backend/raw_data_files/UG Mines .docx'
```

The ingestion comparison times extraction and chunking on one DOCX. It does not
measure full index construction. The voice benchmark reports first playable
audio bytes, not first audible browser playback. To compare another embedding
model, install it separately, set its name and dimensions in `backend/config.py`,
build a candidate index, and evaluate it before promotion. The configured
promotion command checks index count, dimension, OCR availability, and recall@5
against the active index when their embedding profile matches:

```powershell
.\.venv\Scripts\python.exe backend/train_engine.py --force-rebuild
.\.venv\Scripts\python.exe backend/evaluate_retrieval.py backend/evals/golden_local.jsonl --top-k 5 --candidate
.\.venv\Scripts\python.exe backend/train_engine.py --promote-candidate
```

The active collection remains available until successful promotion. Review the
retrieval output and source coverage before adding labels or changing models.
