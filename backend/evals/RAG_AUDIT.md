# Offline RAG audit — 24 October 2026

## Execution path and confirmed causes

Code locations below refer to this working tree: discovery and loading `train_engine.py:419-455`; table chunking `train_engine.py:455`; index mutation `train_engine.py:676`; query embedding `services/embedding_service.py:102`; index compatibility `services/qa_service.py:396`; hybrid retrieval `services/qa_service.py:874`; evidence decision `services/evidence_gate.py:128`; generation and stream `services/qa_service.py:1170`; frontend stream rendering `frontend/app.py:414`.

| Stage | Actual implementation | Finding and correction |
| --- | --- | --- |
| Discovery, extraction and metadata | `train_engine.iter_raw_files`, `load_file`, `load_pdf`; `services.document_parsers.parse` | DOCX sections and tables, spreadsheet rows and PDF pages already carry metadata. OCR is attempted when configured, but extraction can return no usable text. `train_engine.train` now records that as a failure and preserves the previous manifest entry rather than silently discarding it. The serving path rejects manifests with unresolved failures. OCR accuracy itself has not been measured. |
| Chunking | `train_engine.split_documents` | Oversized DOCX tables formerly repeated the heading but lost the column header after splitting. Table chunks now repeat heading and column header and keep individual rows intact. Other paragraphs still use the existing 850-character/150-overlap splitter; a corpus-wide chunk-size comparison remains unverified. |
| IDs, embedding and storage | `train_engine.chunk_id`, `create_vector_store`, `train`, `promote_candidate`; `services.embedding_service.PromptedOllamaEmbeddings` | Content-based chunk IDs and candidate collection promotion already exist. Query vectors now receive the same dimension check as document vectors. The service compares manifest dimensions when present and verifies active collection vector count against manifest IDs. The old v6 local manifest lacks explicit dimensions; its embedding fingerprint is still checked. Collection deletion during candidate reset now ignores only a confirmed missing collection. |
| Query and ranking | `services.qa_service._history_aware_query`, `_feedback_retrieval_query`, `_retrieve_context`, `_hybrid_rank`; `services.lexical_service.lexical_search`; `services.reranker_service.rerank_documents` | Dense Chroma cosine search and local SQLite lexical search already use reciprocal-rank fusion. The optional local cross-encoder is disabled; no cached weights or benchmark justifies enabling it. The old 0.25 dense threshold is a candidate filter, not proof of an answer. The v6 Nomic index retrieves an unrelated critical-minerals passage for a Hindi safety question. |
| Context assembly | `services.qa_service._format_context` | Previously cut selected passages at a per-document character budget, possibly dropping a qualification or value. It now keeps whole selected passages and raises a controlled error if they exceed the configured context budget. |
| Evidence and generation | `services.qa_service._select_answer_docs`, `_format_context`, `_prompt_for_query`, `_stream_answer_events`; `services.evidence_gate.assess_evidence` | Previously any nonempty retrieved list could reach the answer model. The new decision checks requested terms, relations and versions before generation; unsupported questions return a fixed indexed-document abstention. It detects direct disagreement between same-scope measured values. A bounded Hinglish term map handles common mixed-language wording without calling another model. The central chat-template system message now explicitly limits factual evidence to selected passages. Numeric claims absent from evidence are rejected before the first release when possible and checked again at completion. This is a conservative heuristic, not a proof of entailment. |
| Rendering | `frontend.app.ask_api_stream` | Existing stream, cancellation and audio APIs were retained. A late numeric validation failure sends a `replace` event, so an earlier streamed phrase may already have been heard; a fully verified speech stream needs sentence-level buffering or a stronger verifier. |

## Reproducible measurement

The version-controlled `rag_cases.jsonl` contains 15 queries from the 11-file local corpus and deliberately absent facts, divided before evaluation into 7 validation and 8 held-out cases. The active Windows index is **Nomic Embed Text, 768 dimensions, 1,778 vectors in 11 files**. The configured DGX profile remains **BGE-M3, 1,024 dimensions**; its weights and compatible index were unavailable here, so no honest model winner or reranker improvement can be claimed.

| Metric | Legacy nonempty-retrieval decision | New evidence decision |
| --- | ---: | ---: |
| Validation decision accuracy | 5/7 (71.4%) | 7/7 (100%) |
| Held-out decision accuracy | 6/8 (75%) | 7/8 (87.5%) |
| False admission on unanswerable cases | 2/6 (33.3%) | 0/6 |
| Source recall@5, all answerable cases | 8/9 (88.9%) | 8/9 (88.9%) |
| Source MRR and nDCG@10 | 0.8889 / 0.8889 | 0.8889 / 0.8889 |

The one held-out false abstention is the Hindi safety question: Nomic ranks critical-minerals PDFs instead of `Safety Modules.docx`. The gate correctly avoids inventing an answer, but this does not meet the desired answerability target. The selected BGE-M3 production model must be evaluated against the same labels on DGX before choosing an embedding configuration. The measured set is small; the zero observed false-admission rate is not a statistical guarantee.

An already-staged Nomic v8 candidate (1,585 vectors, same 850/150 chunk settings) was also evaluated read-only. It retained 8/9 source recall@5 but reduced MRR from 0.8889 to 0.8333 and nDCG@10 from 0.8889 to 0.8479. It was **not promoted**. These are not a controlled chunk-size comparison; the candidate was built before this change with a different ingestion revision.

Retrieval metrics use **source-level labels** and are unchanged because the gate runs after retrieval. Decision metrics are deterministic checks, not final generated-answer correctness or claim-level groundedness. A local answer-generation baseline could only complete three English cases before the Windows host ran out of paging-file capacity; end-to-end answer and audio quality on DGX remain unverified.

## Redacted query traces

| Query | First retrieved source / score / content hash | Decision | Resulting context and reply |
| --- | --- | --- | --- |
| Rotating wheel with buckets | `BWE.docx` / 0.8023 / `3b2c35ac5547` | `ANSWERABLE` | Selected BWE passages are eligible for the model; generated wording not measured here. |
| Number of critical **mines** in 2023 | Critical-minerals PDF / 0.7884 / `8278e3522b41` | `INSUFFICIENT_EVIDENCE` | No passages sent; fixed indexed-document abstention. |
| 2099 edition of mine safety rules | No passages | `REQUIRED_VERSION_NOT_FOUND` | No passages sent; fixed indexed-document abstention. |
| Hindi question about mine safety | Critical-minerals PDF / 0.5649 / `1bf13da366a0` | `INSUFFICIENT_EVIDENCE` | No passages sent; fixed indexed-document abstention; this is the remaining retrieval miss. |

Full score and hash traces are in `rag_baseline_nomic_2026-10-24.json`, `rag_after_nomic_2026-10-24.json`, and `rag_candidate_nomic_2026-10-24.json`. None contains passage text or generated answers.

## Safe DGX validation and migration

From the Compose directory, first back up `backend/vector_db` outside the live path and verify the copy. Do **not** run the default index job against the only copy of the live database if source files or metadata may have changed: incremental indexing can remove entries for deleted source files. With a verified backup, build a staged collection:

```bash
cp -a backend/vector_db backend/vector_db.backup-$(date +%Y%m%d-%H%M%S)
docker compose run --rm index-init python train_engine.py --force-rebuild
docker compose run --rm index-init python evals/evaluate_rag.py --candidate
docker compose run --rm index-init python train_engine.py --promote-candidate
docker compose up -d --build app
```

Review the staged index's ingestion summary, failures, model fingerprint, dimensions, vector count and labeled retrieval results **before** promotion. The `--force-rebuild` path uses a separate candidate collection; promotion retains the prior collection for rollback. On DGX, run the evaluator against the configured BGE-M3 profile without the Nomic override. Do not promote solely because this small local validation set passes.

## Test result and remaining risks

`python -m unittest discover -s tests -p test_*.py -q`: **194 tests passed** after changes; baseline was 182 tests with one failure caused by an outdated hard-coded context-count assertion and the pre-existing setting of three answer passages. New tests cover the evidence decisions, same-scope numeric conflicts, unsupported numeric claims, complete context passages and table header preservation.

The gate uses direct term and relation checks plus a small mixed-language lexicon. It can still admit a plausible sentence whose nonnumeric meaning is unsupported, and it can abstain from genuine cross-language evidence. It does not yet detect general conflicting claims or verify every generated proposition. The current results support safer abstention on the measured examples, not a guarantee that all hallucinations are eliminated. No source files or indexes were deleted or rebuilt during this work.
