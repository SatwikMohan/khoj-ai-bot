import os
import threading
from functools import lru_cache


RERANK_LOCK = threading.Lock()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _components():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
    offline = _env_bool("OFFLINE_MODE", True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        local_files_only=offline,
    )
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        local_files_only=offline,
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device


def _format_pair(question: str, document: str) -> str:
    instruction = "Retrieve passages from mining documents that answer the user's question"
    return f"<Instruct>: {instruction}\n<Query>: {question}\n<Document>: {document}"


def rerank_documents(question: str, documents: list, top_k: int):
    if not _env_bool("RERANK_ENABLED", True) or len(documents) <= 1:
        return documents[:top_k]

    try:
        import torch

        tokenizer, model, device = _components()
        prefix = (
            "<|im_start|>system\nDecide whether the document satisfies the query and instruction. "
            "The answer must be yes or no.<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        max_length = int(os.getenv("RERANK_MAX_LENGTH", "2048"))
        pairs = [
            _format_pair(question, document.page_content[:6000])
            for document in documents
        ]
        encoded = tokenizer(
            pairs,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )["input_ids"]
        input_ids = [prefix_tokens + item + suffix_tokens for item in encoded]
        batch = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        ).to(device)
        yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id = tokenizer.encode("no", add_special_tokens=False)[0]
        with RERANK_LOCK, torch.inference_mode():
            logits = model(**batch).logits[:, -1, :]
            binary_logits = torch.stack((logits[:, no_id], logits[:, yes_id]), dim=1)
            scores = torch.softmax(binary_logits, dim=1)[:, 1].float().cpu().tolist()

        ranked = []
        for document, score in zip(documents, scores, strict=True):
            document.metadata["rerank_score"] = round(float(score), 5)
            ranked.append(document)
        ranked.sort(key=lambda document: document.metadata["rerank_score"], reverse=True)
        return ranked[:top_k]
    except Exception as exc:
        print(f"Reranker unavailable; using fused retrieval order: {exc}")
        return documents[:top_k]


def reranker_runtime_status() -> dict:
    if not _env_bool("RERANK_ENABLED", True):
        return {"status": "disabled"}
    try:
        _components()
        return {
            "status": "ready",
            "model": os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B"),
        }
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}
