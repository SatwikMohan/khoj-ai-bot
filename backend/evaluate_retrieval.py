import argparse
import json
import statistics
import time
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


def evaluate(path: Path, top_k: int) -> int:
    cases = load_cases(path)
    if not cases:
        raise ValueError("The evaluation file contains no cases.")

    store = _vector_store()
    hits = 0
    reciprocal_ranks = []
    latencies = []
    results = []
    for case in cases:
        started_at = time.perf_counter()
        docs = _retrieve_context(store, case["question"], top_k)
        latencies.append((time.perf_counter() - started_at) * 1000)
        actual_sources = [source.source or "" for source in _source_chunks(docs)]
        expected_sources = [str(source) for source in case["expected_sources"]]

        matching_rank = None
        for rank, actual in enumerate(actual_sources, start=1):
            if any(source_matches(actual, expected) for expected in expected_sources):
                matching_rank = rank
                break
        if matching_rank:
            hits += 1
            reciprocal_ranks.append(1 / matching_rank)
        else:
            reciprocal_ranks.append(0.0)
        results.append(
            {
                "question": case["question"],
                "hit": matching_rank is not None,
                "rank": matching_rank,
                "actual_sources": actual_sources,
            }
        )

    summary = {
        "cases": len(cases),
        f"recall_at_{top_k}": round(hits / len(cases), 4),
        "mean_reciprocal_rank": round(statistics.mean(reciprocal_ranks), 4),
        "latency_ms_p50": round(statistics.median(latencies), 1),
        "latency_ms_max": round(max(latencies), 1),
    }
    print(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False))
    return 0 if hits == len(cases) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure retrieval recall and latency.")
    parser.add_argument("cases", type=Path, help="JSONL file containing golden retrieval cases.")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(evaluate(args.cases, args.top_k))
