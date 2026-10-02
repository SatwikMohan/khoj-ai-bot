import argparse
import json
import statistics
import time
import math
from pathlib import Path

from services.qa_service import _retrieve_context, _source_chunks, _vector_store


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
        if not case.get("question") or not case.get("expected_sources"):
            raise ValueError(f"Case {line_number} needs question and expected_sources.")
        cases.append(case)
    return cases


def source_matches(actual: str, expected: str) -> bool:
    return expected.casefold() in actual.casefold()


def _store_for_evaluation(candidate: bool):
    from services.qa_service import _active_index, _vector_db_dir, _vector_store_for

    if not candidate:
        store = _vector_store()
        return store, _active_index()[0]

    from services.embedding_service import embedding_profile
    from train_engine import load_manifest
    import chromadb

    db_dir = _vector_db_dir()
    staged = load_manifest(db_dir).get("candidate")
    if not staged:
        raise ValueError("No candidate index is staged.")
    profile = embedding_profile()
    indexed_profile = staged.get("settings", {}).get("embedding_profile", {})
    if indexed_profile.get("fingerprint") != profile.fingerprint:
        raise ValueError("Candidate embedding profile does not match backend/config.py.")
    collection_name = staged["active_collection"]
    chromadb.PersistentClient(path=str(db_dir)).get_collection(collection_name)
    return _vector_store_for(collection_name, profile.fingerprint, str(db_dir)), collection_name


def evaluate(path: Path, top_k: int, candidate: bool = False) -> int:
    cases = load_cases(path)
    if not cases:
        raise ValueError("The evaluation file contains no cases.")

    store, collection_name = _store_for_evaluation(candidate)
    recalls = []
    precisions = []
    ndcgs = []
    reciprocal_ranks = []
    latencies = []
    results = []
    for case in cases:
        started_at = time.perf_counter()
        docs = _retrieve_context(store, case["question"], top_k,
                                 lexical_collection_name=collection_name)
        latencies.append((time.perf_counter() - started_at) * 1000)
        actual_sources = list(dict.fromkeys(source.source or "" for source in _source_chunks(docs)))[:top_k]
        expected_sources = [str(source) for source in case["expected_sources"]]
        relevance = [int(any(source_matches(actual, expected) for expected in expected_sources))
                     for actual in actual_sources]
        matched_expected = sum(any(source_matches(actual, expected) for actual in actual_sources)
                               for expected in expected_sources)
        recall = matched_expected / len(expected_sources)
        precision = sum(relevance) / top_k
        matching_rank = next((rank for rank, hit in enumerate(relevance, start=1) if hit), None)
        reciprocal_ranks.append(1 / matching_rank if matching_rank else 0.0)
        dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(relevance, start=1))
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(top_k, len(expected_sources)) + 1))
        recalls.append(recall)
        precisions.append(precision)
        ndcgs.append(min(1.0, dcg / ideal) if ideal else 0.0)
        results.append(
            {
                "question": case["question"],
                "hit": matching_rank is not None,
                "rank": matching_rank,
                "recall": recall,
                "precision": precision,
                "ndcg": ndcgs[-1],
                "actual_sources": actual_sources,
            }
        )

    summary = {
        "collection": collection_name,
        "candidate": candidate,
        "cases": len(cases),
        f"recall_at_{top_k}": round(statistics.mean(recalls), 4),
        f"precision_at_{top_k}": round(statistics.mean(precisions), 4),
        f"ndcg_at_{top_k}": round(statistics.mean(ndcgs), 4),
        "mean_reciprocal_rank": round(statistics.mean(reciprocal_ranks), 4),
        "latency_ms_p50": round(statistics.median(latencies), 1),
        "latency_ms_max": round(max(latencies), 1),
    }
    print(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=True))
    return 0 if all(value == 1.0 for value in recalls) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure retrieval recall and latency.")
    parser.add_argument("cases", type=Path, help="JSONL file containing golden retrieval cases.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate", action="store_true",
                        help="Evaluate the staged candidate with its matching configured embedding model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(evaluate(args.cases, args.top_k, args.candidate))
