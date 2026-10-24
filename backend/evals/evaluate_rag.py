"""Offline labelled retrieval and evidence-decision evaluation.

Example:
  python evals/evaluate_rag.py --embed-model nomic-embed-text --embed-dimensions 768 --baseline
The script never builds, deletes or promotes a collection.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from services.qa_service import _retrieve_context
from evaluate_retrieval import _store_for_evaluation


def _matches(doc, expected):
    source = str(doc.metadata.get("source", "")).casefold()
    return any(name.casefold() in source for name in expected)


def _metrics(rows, k):
    positive = [row for row in rows if row["expected_sources"]]
    if not positive:
        return {}
    recalls, precisions, mrr, ndcg = [], [], [], []
    for row in positive:
        docs = row["retrieved"][:k]
        relevant = [int(any(name.casefold() in item["source"].casefold()
                            for name in row["expected_sources"])) for item in docs]
        found = sum(any(name.casefold() in item["source"].casefold() for item in docs)
                    for name in row["expected_sources"])
        recalls.append(found / len(row["expected_sources"]))
        precisions.append(sum(relevant) / k)
        first = next((i for i, value in enumerate(relevant, 1) if value), None)
        mrr.append(1 / first if first else 0.0)
        ideal = min(k, len(row["expected_sources"]))
        dcg = sum(hit / math.log2(i + 1) for i, hit in enumerate(relevant, 1))
        idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal + 1))
        ndcg.append(min(1.0, dcg / idcg) if idcg else 0.0)
    return {
        f"recall_at_{k}": round(statistics.mean(recalls), 4),
        f"precision_at_{k}": round(statistics.mean(precisions), 4),
        "mrr": round(statistics.mean(mrr), 4),
        f"ndcg_at_{k}": round(statistics.mean(ndcg), 4),
    }


def evaluate(cases, *, baseline=False, candidate=False, semantic=False):
    store, collection_name = _store_for_evaluation(candidate)
    results = []
    for case in cases:
        started = time.perf_counter()
        docs = _retrieve_context(store, case["question"], 10,
                                 lexical_collection_name=collection_name)
        result = None
        if baseline:
            decision = "ANSWERABLE" if docs else "INSUFFICIENT_EVIDENCE"
            reason = "legacy_nonempty_retrieval"
        else:
            if semantic:
                from services.semantic_evidence import assess_semantic_evidence, suggest_local_search_query
                result = assess_semantic_evidence(case["question"], docs[:config.QA_TOP_K])
                if result.status.value == "INSUFFICIENT_EVIDENCE":
                    search_query = result.search_query or (
                        suggest_local_search_query(case["question"]) if not docs else ""
                    )
                    if search_query:
                        retry_docs = _retrieve_context(store, search_query, 10,
                                                       scope_question=case["question"],
                                                       lexical_collection_name=collection_name)
                        if retry_docs:
                            retry_result = assess_semantic_evidence(
                                case["question"], retry_docs[:config.QA_TOP_K])
                            if retry_result.status.value in {
                                "ANSWERABLE", "PARTIAL_EVIDENCE", "CONFLICTING_EVIDENCE"
                            }:
                                docs, result = retry_docs, retry_result
            else:
                from services.evidence_gate import assess_evidence
                result = assess_evidence(case["question"], docs[:config.QA_TOP_K])
            decision, reason = result.status.value, result.reason
        retrieval_ms = round((time.perf_counter() - started) * 1000, 1)
        retrieved = [{
            "source": str(doc.metadata.get("source", "")),
            "chunk_id": str(doc.metadata.get("chunk_id", "")),
            "dense_score": doc.metadata.get("relevance_score"),
            "hybrid_score": doc.metadata.get("hybrid_score"),
            "content_sha256": hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:12],
        } for doc in docs]
        results.append({
            "id": case["id"], "split": case["split"], "category": case["category"],
            "question": case["question"], "expected_decision": case["expected_decision"],
            "actual_decision": decision, "decision_reason": reason,
            "expected_sources": case["expected_sources"], "retrieved": retrieved,
            "retrieval_ms": retrieval_ms,
        })
    summary = {}
    for split in ("validation", "heldout", "all"):
        rows = results if split == "all" else [row for row in results if row["split"] == split]
        negatives = [row for row in rows if row["expected_decision"] in {
            "INSUFFICIENT_EVIDENCE", "REQUIRED_VERSION_NOT_FOUND"}]
        summary[split] = {
            "cases": len(rows),
            **_metrics(rows, 5), **_metrics(rows, 10),
            "decision_accuracy": round(sum(row["actual_decision"] == row["expected_decision"]
                                          for row in rows) / len(rows), 4) if rows else None,
            "unanswerable_false_admission_rate": round(
                sum(row["actual_decision"] == "ANSWERABLE" for row in negatives) /
                len(negatives), 4) if negatives else None,
            "latency_ms_p50": round(statistics.median(row["retrieval_ms"] for row in rows), 1) if rows else None,
        }
    return {"summary": summary, "cases": results, "baseline": baseline,
            "semantic": semantic,
            "collection": collection_name, "candidate": candidate,
            "limitations": "Source-level labels and lexical checks do not prove answer entailment."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("rag_cases.jsonl"))
    parser.add_argument("--embed-model")
    parser.add_argument("--embed-dimensions", type=int)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--candidate", action="store_true", help="Evaluate the staged collection, not the active one.")
    parser.add_argument("--semantic", action="store_true", help="Call the configured offline semantic evidence model.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.embed_model:
        config.OLLAMA_EMBED_MODEL = args.embed_model
    if args.embed_dimensions:
        config.OLLAMA_EMBED_DIMENSIONS = args.embed_dimensions
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = evaluate(cases, baseline=args.baseline, candidate=args.candidate,
                      semantic=args.semantic)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
