"""Measure installed Ollama chat models on one fixed, local document prompt.

This isolates model prefill/decode from retrieval and speech. It never pulls a
model. Run the same command inside the DGX app container after provisioning.
"""

import argparse
import json
import platform
import statistics
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import config
from services.embedding_service import ollama_model_names, model_is_available


WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def document_context(path: Path, max_chars: int = 3500) -> str:
    if path.suffix.lower() != ".docx":
        raise ValueError("The fixed comparison source must be a DOCX file.")
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs = ["".join(node.text or "" for node in paragraph.iter(f"{WORD_NS}t")).strip()
                  for paragraph in root.iter(f"{WORD_NS}p")]
    context = "\n".join(part for part in paragraphs if part)
    if not context:
        raise ValueError(f"No text found in {path}")
    return context[:max_chars]


def sample(model: str, context: str, max_tokens: int) -> dict:
    question = "What is a bucket wheel excavator, and where is it used? Answer in three concise sentences."
    payload = {
        "model": model,
        "stream": True,
        "think": False,
        "keep_alive": int(config.OLLAMA_KEEP_ALIVE),
        "options": {"temperature": 0, "num_ctx": int(config.OLLAMA_NUM_CTX),
                    "num_predict": max_tokens},
        "messages": [
            {"role": "system", "content": "Answer only from the supplied document. Do not include private reasoning."},
            {"role": "user", "content": f"Document: BWE.docx\n{context}\n\nQuestion: {question}"},
        ],
    }
    request = urllib.request.Request(
        config.OLLAMA_BASE_URL.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_ms = None
    answer = []
    final = {}
    with urllib.request.urlopen(request, timeout=float(config.OLLAMA_REQUEST_TIMEOUT_SECONDS)) as response:
        for line in response:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            token = event.get("message", {}).get("content", "")
            if token:
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - started) * 1000, 1)
                answer.append(token)
            if event.get("done"):
                final = event
                break
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    decode_ns = int(final.get("eval_duration") or 0)
    prefill_ns = int(final.get("prompt_eval_duration") or 0)
    return {
        "model": model,
        "first_token_ms": first_token_ms,
        "total_ms": elapsed_ms,
        "prompt_tokens": final.get("prompt_eval_count"),
        "prefill_tokens_per_second": round(int(final.get("prompt_eval_count") or 0) * 1e9 / prefill_ns, 2) if prefill_ns else None,
        "output_tokens": final.get("eval_count"),
        "decode_tokens_per_second": round(int(final.get("eval_count") or 0) * 1e9 / decode_ns, 2) if decode_ns else None,
        "answer": "".join(answer).strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=[config.OLLAMA_CHAT_MODEL])
    parser.add_argument("--source", type=Path,
                        default=Path(config.RAW_DATA_DIR) / "Q&As" / "BWE.docx")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1:
        parser.error("repeats and max-tokens must be positive")
    context = document_context(args.source)
    installed = ollama_model_names(config.OLLAMA_BASE_URL)
    samples = []
    missing = []
    for model in args.models:
        if not model_is_available(model, installed):
            missing.append(model)
            continue
        for _ in range(args.repeats):
            samples.append(sample(model, context, args.max_tokens))
    summaries = []
    for model in args.models:
        selected = [item for item in samples if item["model"] == model]
        if selected:
            summaries.append({"model": model, "runs": len(selected),
                              "first_token_ms_p50": statistics.median(item["first_token_ms"] for item in selected),
                              "total_ms_p50": statistics.median(item["total_ms"] for item in selected),
                              "decode_tokens_per_second_p50": statistics.median(item["decode_tokens_per_second"] for item in selected)})
    report = {"host": platform.platform(), "source": str(args.source),
              "context_chars": len(context), "models_missing": missing,
              "summary": summaries, "samples": samples}
    rendered = json.dumps(report, ensure_ascii=True, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
