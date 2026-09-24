import config
import os
import threading
from functools import lru_cache


RERANK_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _components():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config.RERANK_MODEL
    offline = config.OFFLINE_MODE
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
    if not config.RERANK_ENABLED or len(documents) <= 1:
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
        max_length = int(config.RERANK_MAX_LENGTH)
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
            # Qwen3 only needs yes/no logits at the last position. Computing a
            # full vocabulary tensor for every passage token wastes GBs of RAM.
            logits = model(**batch, logits_to_keep=1).logits[:, -1, :]
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
    if not config.RERANK_ENABLED:
        return {"status": "disabled"}
    try:
        _components()
        return {
            "status": "ready",
            "model": config.RERANK_MODEL,
        }
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}
