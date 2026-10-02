# DGX Spark model and latency re-evaluation

2 October 2026. The Windows machine is a development test host. DGX Spark is
the intended deployment target. The local numbers below are measured on the
Windows CPU; the DGX figures are published Ollama measurements on a different
prompt and are labelled separately.

## Measured on the current Windows test host

The only installed Ollama models were `llama3.2:latest` (3B, Q4_K_M) and
`nomic-embed-text:latest` (768 dimensions). `ollama ps` showed both on 100% CPU.
Docker is not installed on this host, so no container or DGX test was run here.

| Test | Measured result | Conditions |
| --- | ---: | --- |
| Direct chat first token | 21.98 s | One run, Llama 3.2 3B, 786 prompt tokens from `BWE.docx`. |
| Direct chat decode | 5.24 tokens/s | 90 output tokens, Ollama `eval_count/eval_duration`. |
| Direct chat total | 39.19 s | Same run; prompt prefill was 41.11 tokens/s. |
| RAG retrieval | recall@5 6/7; p50 60.7 ms | Seven labelled questions, warmed active index; maximum 2.22 s. |
| Structured candidate retrieval | recall@5 6/7; p50 253.2 ms | Separate run, candidate index; maximum 5.31 s. Caches differ, so latency is not a controlled comparison. |
| Voice query first token | 22.47 s | One BWE question through actual RAG and SSE. |
| First speech segment submitted | 40.61 s | Sentence/length segmentation after token arrival. |
| First playable audio chunk | 43.25 s | Piper WAV generated while LLM was still active; not audible browser start. |
| LLM response finished | 61.51 s | Same 570-character answer; retrieval 5.44 s. |

Piper warmup for that voice run took 9.61 s *before* query submission. The
earlier 17.8 s first-playable measurement used another prompt and warm state;
the two samples demonstrate variation, not a repeatable speedup. The first
speech segment in this run lagged the first token by 18.13 s, so segment
boundaries and output style also affect perceived latency. The true browser
playback start was not measured. Both retrieval runs missed the Hindi query
against an English safety document. The staged index has 1,585 vectors and
remains unpromoted because native Windows OCR is unavailable.

Machine-readable results:
[`recheck_chat_2026-10-02.json`](backend/evals/recheck_chat_2026-10-02.json),
[`recheck_retrieval_2026-10-02.json`](backend/evals/recheck_retrieval_2026-10-02.json),
[`recheck_candidate_2026-10-02.json`](backend/evals/recheck_candidate_2026-10-02.json),
[`recheck_voice_2026-10-02.json`](backend/evals/recheck_voice_2026-10-02.json).

## What changes on DGX Spark

DGX Spark has a 20-core ARM64 CPU, Blackwell GPU, and 128 GB of **unified**
memory. It is not a conventional 128 GB discrete VRAM card. NVIDIA documents
273 GB/s memory bandwidth and Docker GPU support; it also notes that
`nvidia-smi` may show `Memory-Usage: Not Supported` for this integrated GPU.
Use `docker compose exec ollama ollama ps` to check whether the Ollama model is
actually on the GPU. See [NVIDIA hardware](https://docs.nvidia.com/dgx/dgx-spark/hardware.html),
[container runtime](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html),
and [known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).

Ollama published ten-run DGX Spark measurements using firmware 580.95.05 and
Ollama 0.12.6, temperature 0, a book-summary prompt, and 500 output tokens.
Those results are **not** a benchmark of this application or of the Compose
image's Ollama version:

| Published DGX model | Quantization | Prefill | Decode | Weight download |
| --- | --- | ---: | ---: | ---: |
| Llama 3.1 8B | Q4_K_M | 7,614 tokens/s | 38.02 tokens/s | about 4.9 GB |
| Llama 3.1 70B | Q4_K_M | 1,911 tokens/s | 4.423 tokens/s | about 43 GB |

Source: [Ollama DGX Spark performance](https://ollama.com/blog/nvidia-spark-performance)
and [official Llama 3.1 tags](https://ollama.com/library/llama3.1/tags).
At those published decode rates, 150 output tokens alone would take roughly
4 s on 8B versus 34 s on 70B. Retrieval, model loading, prompt prefill,
chunked TTS, browser playback, concurrent sessions, and this application's
prompt can change end-to-end time substantially. No precise DGX first-audio
latency is claimed without running on the Spark.

### Model decision for the DGX trial

**Chat:** `backend/config.py` now selects `llama3.1:8b-instruct-q4_K_M` for the
DGX trial. It is a standard
non-reasoning instruction model, its Q4_K_M tag explicitly selects the weights,
and the published DGX decode rate better fits conversational voice than 70B.
The 3B Windows test model remains a commented fallback. Test 70B only if the 8B answers
fail groundedness or Hindi quality checks and the resulting voice delay is
acceptable. Model size alone does not establish quality on this corpus.

**Embeddings:** `backend/config.py` now selects `bge-m3` for the DGX trial.
The old `nomic-embed-text` index remains on disk until a staged BGE-M3 index
passes validation on representative English/Hindi, scans, tables, code,
and versioned documents. BGE-M3 is a plausible multilingual candidate because
its [model card](https://huggingface.co/BAAI/bge-m3) documents 100+ languages,
8,192-token inputs, and 1,024-dimensional dense vectors. The
[Ollama build](https://ollama.com/library/bge-m3) is about 1.2 GB. The current
Ollama–Chroma integration uses **dense embeddings only**; BGE-M3's optional
sparse/multi-vector modes are not enabled. The current seven labels are too
small to prove a production model choice. Do not mix 1,024-dimensional vectors
with the active 768-dimensional collection.

**Context and speech:** start at the existing 8,192-token runtime context and
measure before raising it, because parallel contexts consume unified memory.
Keep Piper as an offline chunked TTS baseline. If first audio remains slow on
DGX, measure retrieval, first token, first speech segment, Piper synthesis,
and browser playback separately before increasing model size or concurrency.

## Ollama model storage in Docker

`backend/config.py` contains Ollama **model names** and the service URL, not
weight paths. Compose runs a separate `ollama` container and mounts the named
`ollama_models` volume at `/root/.ollama`, where the container stores pulled
models. `ollama-model-init` pulls configured names during the connected setup
phase and skips models already present. The app image contains application
code; rebuilding it preserves the model volume. `OLLAMA_NO_CLOUD=1` disables
Ollama cloud features during local runtime. This follows
[Ollama's Docker volume and GPU pattern](https://docs.ollama.com/docker) and
[model-storage documentation](https://docs.ollama.com/faq).

For an air-gapped **first** deployment, transfer a populated Ollama model store
into the DGX volume and provision `/models` for Whisper/Piper before starting
services. A fresh app image alone cannot supply the weights. The empty
`OLLAMA_CHAT_MODEL_PATH` and `OLLAMA_EMBED_MODEL_PATH` config entries have been
removed because the application never used them. The selected Ollama tag,
not `OLLAMA_CHAT_QUANTIZATION` by itself, controls actual quantization.

## Reproduce the model decision on Spark

After building the images on the ARM64 Spark and provisioning each candidate,
run the same document prompt through installed chat models. The script refuses
missing models and never downloads at runtime:

```bash
docker compose exec ollama ollama list
docker compose exec ollama ollama ps
docker compose run --rm --no-deps app python backend/evals/benchmark_chat_runtime.py llama3.2:latest llama3.1:8b-instruct-q4_K_M --repeats 3 | tee spark_chat.json
docker compose run --rm --no-deps app python backend/evals/benchmark_voice.py | tee spark_voice.log
```

Inspect the answers for groundedness and citations; a fast unsupported answer
should fail selection. Record p50 and tail latency for first token, first
playable audio, full answer, and concurrent users. The benchmark script's
fixed DOCX prompt isolates chat inference; the voice script measures the RAG
and speech pipeline. The browser's `voice_first_playback` console event is
needed for audible-start timing.

The config already sets `OLLAMA_EMBED_MODEL = "bge-m3"` and
`OLLAMA_EMBED_DIMENSIONS = 1024`. Pull the model into Ollama's volume on the
DGX and follow the staged candidate and promotion sequence in
[`DEPLOYMENT.md`](DEPLOYMENT.md). The
`--candidate` evaluation checks the staged collection without changing active
traffic. The old collection remains on disk until promotion. Because the app
bind-mounts `config.py`, schedule an embedding change in a maintenance window:
a process that reloads the edited config before promotion may become unready.

Successful operation on the small Windows host is useful for correctness. The
ARM64 image, GPU placement, model load time, offline assets, and real browser
audio path still require validation on the DGX Spark itself.
