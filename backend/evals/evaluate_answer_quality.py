"""Run labelled, direct backend answer checks against the current local index.

Example: python backend/evals/evaluate_answer_quality.py --chat-model llama3.2:latest --embed-model nomic-embed-text --embed-dimensions 768
Overrides apply only to this process; backend/config.py remains the DGX profile.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from services.language_service import detect_language


def evaluate(case_file: Path, limit: int = 0) -> dict:
    from services.qa_service import (
        _clean_content, _format_context, _retrieve_context, _select_answer_docs,
        _source_chunks, _vector_store, answer_question,
    )
    from evaluate_retrieval import source_matches

    cases = [json.loads(line) for line in case_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if limit:
        cases = cases[:limit]
    store = _vector_store()
    results = []
    for case in cases:
        question = case["question"]
        docs = _retrieve_context(store, question, config.QA_TOP_K)
        used = _select_answer_docs(question, docs, "document")
        sources = [source.source or "" for source in _source_chunks(used)]
        expected = case["expected_sources"]
        retrieval_ok = any(source_matches(actual, target) for actual in sources for target in expected)
        context = _format_context(used)
        context_ok = retrieval_ok and all(
            term.casefold() in context.casefold() for term in case.get("context_terms", [])
        )
        answer = answer_question(question).answer
        terms_ok = all(term.casefold() in answer.casefold() for term in case.get("answer_terms", []))
        forbidden_ok = not any(term.casefold() in answer.casefold() for term in case.get("forbidden_answer_terms", []))
        language_ok = detect_language(answer) == case["language"]
        grounding_check = all(
            term.casefold() in " ".join(_clean_content(doc.page_content) for doc in used).casefold()
            for term in case.get("answer_terms", [])
        )
        checks = {"retrieval": retrieval_ok, "context": context_ok, "answer_terms": terms_ok,
                  "forbidden_terms": forbidden_ok, "language": language_ok,
                  "labelled_facts_in_evidence": grounding_check}
        results.append({"id": case["id"], "checks": checks, "passed": all(checks.values()),
                        "sources": sources, "answer": answer})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    return {"cases": len(results), "passed": sum(item["passed"] for item in results),
            "results": results,
            "limitation": "Term and language checks do not prove full factual entailment or natural speech quality."}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("answer_quality_cases.jsonl"))
    parser.add_argument("--chat-model")
    parser.add_argument("--embed-model")
    parser.add_argument("--embed-dimensions", type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.chat_model:
        config.OLLAMA_CHAT_MODEL = args.chat_model
    if args.embed_model:
        config.OLLAMA_EMBED_MODEL = args.embed_model
    if args.embed_dimensions:
        config.OLLAMA_EMBED_DIMENSIONS = args.embed_dimensions
    result = evaluate(args.cases, args.limit)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}), flush=True)
    return 0 if result["passed"] == result["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
