import config
import threading
from functools import lru_cache


RERANK_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _components():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = config.RERANK_MODEL
    offline = config.OFFLINE_MODE
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        local_files_only=offline,
    )
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=dtype,
        local_files_only=offline,
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device


def rerank_documents(question: str, documents: list, top_k: int):
    if not config.RERANK_ENABLED or len(documents) <= 1:
        return documents[:top_k]

    try:
        import torch

        tokenizer, model, device = _components()
        pairs = [(question, document.page_content[:6000]) for document in documents]
        batch = tokenizer(pairs, padding=True, truncation=True,
                          max_length=min(int(config.RERANK_MAX_LENGTH), 512),
                          return_tensors="pt").to(device)
        with RERANK_LOCK, torch.inference_mode():
            scores = model(**batch).logits.view(-1).float().cpu().tolist()

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
