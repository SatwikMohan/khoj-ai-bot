import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from helpers.request_models import ChatMessage, QAResponse, SourceChunk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
DEFAULT_VECTOR_DB_DIR = REPO_ROOT / "vector_db"
DEFAULT_RELEVANCE_THRESHOLD = 0.15
DEFAULT_CONTEXT_MAX_CHARS = 24000
DEFAULT_RETRIEVAL_FETCH_K = 60
DEFAULT_CHAT_HISTORY_MAX_TURNS = 8
DEFAULT_CHAT_HISTORY_MAX_CHARS = 2400
DEFAULT_CHAT_HISTORY_MESSAGE_CHARS = 520
DEFAULT_MIN_RETRIEVED_TEXT_CHARS = 40
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

HINGLISH_MARKERS = {
    "aap",
    "ap",
    "hai",
    "hain",
    "kya",
    "kaise",
    "kaisa",
    "kaisi",
    "mujhe",
    "muje",
    "batao",
    "bataye",
    "bataiye",
    "kripya",
    "dhanyavad",
    "shukriya",
    "namaste",
    "haan",
    "nahi",
    "ka",
    "ki",
    "ke",
    "mein",
    "me",
}

GENERAL_EXACT_MESSAGES = {
    "hi",
    "hii",
    "hello",
    "hey",
    "heyy",
    "hola",
    "namaste",
    "namaskar",
    "नमस्ते",
    "नमस्कार",
    "thanks",
    "thank you",
    "thankyou",
    "dhanyavad",
    "shukriya",
    "ok",
    "okay",
    "fine",
    "great",
    "cool",
    "bye",
    "goodbye",
}

GENERAL_PHRASES = {
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "how r u",
    "who are you",
    "what are you",
    "what can you do",
    "what is your name",
    "tell me about yourself",
    "can you help me",
    "help me",
    "kaise ho",
    "aap kaise ho",
    "tum kaise ho",
    "aap kaun ho",
    "tum kaun ho",
    "kya kar sakte ho",
    "madad kar sakte ho",
}

SUMMARY_TERMS = {
    "summarize",
    "summarise",
    "summary",
    "overview",
    "brief",
    "recap",
    "gist",
    "highlights",
    "key points",
    "short note",
}


class QAEngineError(RuntimeError):
    pass


def _load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default.resolve()

    path = Path(value)
    if path.is_absolute():
        return path

    for base_dir in (REPO_ROOT, PROJECT_ROOT):
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate

    return (REPO_ROOT / path).resolve()


def _normalized_query(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower()).strip(" .!?।")


def _detect_language_style(question: str) -> str:
    normalized = _normalized_query(question)
    if DEVANAGARI_RE.search(question):
        return "hindi"

    words = set(re.findall(r"[a-zA-Z]+", normalized))
    marker_count = len(words & HINGLISH_MARKERS)
    if marker_count >= 2 or normalized in {"namaste", "dhanyavad", "shukriya"}:
        return "hinglish"

    return "english"


def _language_instruction(language_style: str) -> str:
    if language_style == "hindi":
        return (
            "Respond in natural Hindi using Devanagari script. Keep it warm and spoken, "
            "with clear Markdown formatting where it helps."
        )
    if language_style == "hinglish":
        return (
            "Respond in natural Hinglish using simple Roman Hindi-English phrasing. Keep it "
            "warm and spoken, with clear Markdown formatting where it helps."
        )
    return "Respond in natural English with clear Markdown formatting where it helps."


def _is_general_query(question: str) -> bool:
    normalized = _normalized_query(question)
    if not normalized:
        return True

    words = normalized.split()
    if normalized in GENERAL_EXACT_MESSAGES:
        return True

    if len(words) <= 6 and any(phrase == normalized for phrase in GENERAL_PHRASES):
        return True

    if len(words) <= 4 and any(
        normalized.startswith(f"{greeting} ")
        for greeting in ("hi", "hello", "hey", "namaste", "नमस्ते", "नमस्कार")
    ):
        return True

    return False


def _is_summary_query(question: str) -> bool:
    normalized = _normalized_query(question)
    return any(term in normalized for term in SUMMARY_TERMS)


def _target_filter(question: str) -> dict | None:
    return None


def _general_response(question: str) -> QAResponse:
    language_style = _detect_language_style(question)
    normalized = _normalized_query(question)

    if normalized in {"thanks", "thank you", "thankyou", "dhanyavad", "shukriya"}:
        if language_style == "hindi":
            answer = "बहुत खुशी हुई। जब भी आप तैयार हों, अपने दस्तावेजों से जुड़ा सवाल पूछिए।"
        elif language_style == "hinglish":
            answer = "Khushi hui. Jab bhi aap ready ho, documents se related sawaal pooch lijiye."
        else:
            answer = "You are very welcome. Whenever you are ready, ask me anything from your documents."
    elif normalized in {"bye", "goodbye"}:
        if language_style == "hindi":
            answer = "ठीक है, फिर मिलते हैं। जब जरूरत हो, मैं यहीं हूँ।"
        elif language_style == "hinglish":
            answer = "Theek hai, phir milte hain. Jab zarurat ho, main yahin hoon."
        else:
            answer = "Alright, see you soon. I will be here when you need me."
    elif language_style == "hindi":
        answer = (
            "नमस्ते! मैं तैयार हूँ। आप चाहें तो सामान्य बात कर सकते हैं, या अपने दस्तावेजों "
            "से जुड़ा कोई सवाल पूछ सकते हैं।"
        )
    elif language_style == "hinglish":
        answer = (
            "Hi! Main ready hoon. Aap normal chat kar sakte ho, ya documents se related "
            "koi specific sawaal pooch sakte ho."
        )
    else:
        answer = (
            "Hi! I am ready. You can chat with me normally, or ask a specific question "
            "from your indexed documents."
        )

    return QAResponse(answer=answer, sources=[], query_type="general")


def _collection_name() -> str:
    return os.getenv("CHROMA_COLLECTION_NAME", "texmin_qa")


def _vector_db_dir() -> Path:
    return _resolve_path(os.getenv("VECTOR_DB_DIR"), DEFAULT_VECTOR_DB_DIR)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _ollama_keep_alive() -> int:
    return _env_int("OLLAMA_KEEP_ALIVE", 1800)


@lru_cache(maxsize=1)
def _embeddings() -> OllamaEmbeddings:
    _load_environment()
    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma"),
        base_url=_ollama_base_url(),
        keep_alive=_ollama_keep_alive(),
    )


@lru_cache(maxsize=1)
def _vector_store() -> Chroma:
    _load_environment()
    db_dir = _vector_db_dir()
    if not db_dir.exists():
        raise QAEngineError(
            f"Vector database not found at {db_dir}. Run `python train_engine.py` first."
        )

    return Chroma(
        collection_name=_collection_name(),
        embedding_function=_embeddings(),
        persist_directory=str(db_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


@lru_cache(maxsize=8)
def _chat_llm(model: str, base_url: str, temperature: float) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        keep_alive=_ollama_keep_alive(),
        num_predict=_env_int("OLLAMA_NUM_PREDICT", 900),
        num_ctx=_env_int("OLLAMA_NUM_CTX", 8192),
    )


def _llm(temperature: float) -> ChatOllama:
    _load_environment()
    return _chat_llm(
        os.getenv("OLLAMA_CHAT_MODEL", "qwen3:30b"),
        _ollama_base_url(),
        round(float(temperature), 2),
    )


def warm_up_qa_engine() -> None:
    _load_environment()
    if not _env_bool("QA_WARMUP_ON_STARTUP", True):
        return

    _embeddings().embed_query("warm up document retrieval")
    _llm(0.1).invoke("Reply with: ready")


def _doc_key(doc) -> tuple:
    return (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.metadata.get("start_index"),
        doc.metadata.get("chunk_index"),
    )


def _clean_content(content: str) -> str:
    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(("Document source:", "Folder path:", "File name:")):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _doc_years(doc) -> tuple[str, ...]:
    text = " ".join(
        [
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("folder_path", "")),
            str(doc.metadata.get("file_name", "")),
            _clean_content(doc.page_content)[:2500],
        ]
    )
    return tuple(sorted(set(YEAR_RE.findall(text))))


def _has_meaningful_content(doc) -> bool:
    min_chars = _env_int("MIN_RETRIEVED_TEXT_CHARS", DEFAULT_MIN_RETRIEVED_TEXT_CHARS)
    return len(" ".join(_clean_content(doc.page_content).split())) >= min_chars


def _format_context(docs) -> str:
    max_chars = int(os.getenv("CONTEXT_MAX_CHARS", str(DEFAULT_CONTEXT_MAX_CHARS)))
    used_chars = 0
    context_blocks = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        folder_path = doc.metadata.get("folder_path")
        folder_label = f", folder {folder_path}" if folder_path else ""
        page = doc.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        years = _doc_years(doc)
        year_label = f", years {', '.join(years)}" if years else ""
        content = _clean_content(doc.page_content)
        remaining_chars = max_chars - used_chars
        if remaining_chars <= 0:
            break
        if len(content) > remaining_chars:
            content = content[:remaining_chars].rsplit(" ", 1)[0].strip()
        block = f"[Context {index}: {source}{folder_label}{page_label}{year_label}]\n{content}"
        context_blocks.append(block)
        used_chars += len(block)
    return "\n\n".join(context_blocks)


def _source_chunks(docs) -> list[SourceChunk]:
    chunks = []
    for doc in docs:
        preview = " ".join(_clean_content(doc.page_content).split())[:280]
        chunks.append(
            SourceChunk(
                source=doc.metadata.get("source"),
                folder_path=doc.metadata.get("folder_path"),
                page=doc.metadata.get("page"),
                chunk_index=doc.metadata.get("chunk_index"),
                relevance_score=doc.metadata.get("relevance_score"),
                preview=preview,
            )
        )
    return chunks


def _source_dicts(sources: list[SourceChunk]) -> list[dict]:
    return [
        source.model_dump() if hasattr(source, "model_dump") else source.dict()
        for source in sources
    ]


def _message_content(message) -> str:
    if isinstance(message, ChatMessage):
        return message.content
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _message_role(message) -> str:
    if isinstance(message, ChatMessage):
        return message.role
    if isinstance(message, dict):
        return str(message.get("role", ""))
    return str(getattr(message, "role", ""))


def _trim_text(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rsplit(" ", 1)[0].strip() + "..."


def _format_chat_history(chat_history: list[ChatMessage] | None) -> str:
    max_turns = _env_int("CHAT_HISTORY_MAX_TURNS", DEFAULT_CHAT_HISTORY_MAX_TURNS)
    max_chars = _env_int("CHAT_HISTORY_MAX_CHARS", DEFAULT_CHAT_HISTORY_MAX_CHARS)
    max_message_chars = _env_int("CHAT_HISTORY_MESSAGE_CHARS", DEFAULT_CHAT_HISTORY_MESSAGE_CHARS)

    lines = []
    used_chars = 0
    for message in reversed((chat_history or [])[-max_turns:]):
        role = _message_role(message).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _trim_text(_message_content(message), max_message_chars)
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        line = f"{label}: {content}"
        if used_chars + len(line) > max_chars:
            break
        lines.append(line)
        used_chars += len(line)

    if not lines:
        return "No prior conversation."
    return "\n".join(reversed(lines))


def _history_aware_query(question: str, chat_history: list[ChatMessage] | None) -> str:
    recent_history = []
    for message in (chat_history or [])[-6:]:
        role = _message_role(message).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _trim_text(_message_content(message), 240)
        if content:
            recent_history.append(f"{role}: {content}")

    if not recent_history:
        return question

    history_text = " ".join(recent_history)
    return _trim_text(f"Current question: {question}. Recent conversation: {history_text}", 1600)


def _diversity_key(doc) -> tuple[str, str]:
    years = _doc_years(doc)
    source = str(doc.metadata.get("source") or doc.metadata.get("file_name") or "")
    return source, ",".join(years)


def _diversify_docs(docs, top_k: int):
    selected = []
    selected_keys = set()
    selected_doc_keys = set()

    for doc in docs:
        key = _diversity_key(doc)
        if key in selected_keys:
            continue
        selected.append(doc)
        selected_keys.add(key)
        selected_doc_keys.add(_doc_key(doc))
        if len(selected) >= top_k:
            return selected

    for doc in docs:
        key = _doc_key(doc)
        if key in selected_doc_keys:
            continue
        selected.append(doc)
        selected_doc_keys.add(key)
        if len(selected) >= top_k:
            break

    return selected


def _retrieve_context(vector_store: Chroma, question: str, top_k: int):
    fetch_k = max(top_k, _env_int("RETRIEVAL_FETCH_K", DEFAULT_RETRIEVAL_FETCH_K))
    threshold = float(os.getenv("RELEVANCE_SCORE_THRESHOLD", str(DEFAULT_RELEVANCE_THRESHOLD)))

    scored_docs = vector_store.similarity_search_with_relevance_scores(question, k=fetch_k)
    fallback_by_key = {}
    relevant_by_key = {}
    for doc, score in scored_docs:
        if not _has_meaningful_content(doc):
            continue
        doc.metadata["relevance_score"] = round(float(score), 4)
        fallback_by_key[_doc_key(doc)] = doc
        if score >= threshold:
            relevant_by_key[_doc_key(doc)] = doc

    if not relevant_by_key:
        return _diversify_docs(list(fallback_by_key.values()), top_k)

    if not _env_bool("RETRIEVAL_MMR_ENABLED", True):
        return _diversify_docs(list(relevant_by_key.values()), top_k)

    lambda_mult = float(os.getenv("MMR_LAMBDA_MULT", "0.25"))
    mmr_docs = vector_store.max_marginal_relevance_search(
        question,
        k=top_k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )

    selected_docs = []
    selected_keys = set()
    for doc in mmr_docs:
        key = _doc_key(doc)
        if key in relevant_by_key:
            selected_docs.append(relevant_by_key[key])
            selected_keys.add(key)

    if len(selected_docs) < top_k:
        for key, doc in relevant_by_key.items():
            if key not in selected_keys:
                selected_docs.append(doc)
                if len(selected_docs) >= top_k:
                    break

    return _diversify_docs(selected_docs, top_k)


def _retrieve_summary_context(vector_store: Chroma, question: str, top_k: int):
    summary_k = max(top_k, int(os.getenv("SUMMARY_CONTEXT_CHUNKS", "14")))
    fetch_k = max(summary_k, int(os.getenv("SUMMARY_FETCH_K", "80")))
    lambda_mult = float(os.getenv("SUMMARY_MMR_LAMBDA_MULT", "0.2"))
    filter_kwargs = _target_filter(question)
    expanded_query = (
        f"{question}. annual report overview performance operations financial results "
        "production sustainability safety risks strategy management discussion highlights"
    )

    docs = vector_store.max_marginal_relevance_search(
        expanded_query,
        k=summary_k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        filter=filter_kwargs,
    )

    selected_docs = []
    seen = set()
    for doc in docs:
        if not _has_meaningful_content(doc):
            continue
        key = _doc_key(doc)
        if key in seen:
            continue
        doc.metadata["relevance_score"] = None
        selected_docs.append(doc)
        seen.add(key)

    return selected_docs[:summary_k]


def _not_enough_context_response(question: str) -> QAResponse:
    language_style = _detect_language_style(question)
    if language_style == "hindi":
        answer = (
            "मैंने दस्तावेजों में देखा, लेकिन इस सवाल का भरोसे से जवाब देने लायक पर्याप्त "
            "जानकारी नहीं मिली। मैं अनुमान लगाने के बजाय साफ कहना चाहूँगा। थोड़ा और विवरण "
            "दें, या संबंधित दस्तावेज जोड़कर ट्रेनिंग स्क्रिप्ट फिर चलाएँ।"
        )
    elif language_style == "hinglish":
        answer = (
            "Maine documents mein check kiya, lekin is sawaal ka confidently jawab dene ke "
            "liye enough reliable context nahi mila. Guess karne se better hai main honestly "
            "bata doon. Thoda aur detail dijiye, ya relevant documents add karke training "
            "script dobara run kar lijiye."
        )
    else:
        answer = (
            "I checked the documents, but I do not have enough reliable information to answer "
            "that confidently. I would rather be honest than guess. Try adding a little more "
            "detail, or add the relevant documents and run the training script again."
        )

    return QAResponse(answer=answer, sources=[], query_type="document")


def _prompt_for_query(query_type: str) -> ChatPromptTemplate:
    shared_style = (
        "Write like a thoughtful person speaking to the user, not like a citation engine. "
        "Do not start with phrases like 'based on the context', 'from the provided context', "
        "'the snippets say', or 'according to the documents'. Start directly with the useful "
        "answer. Use Markdown naturally: short paragraphs, headings, bullets, bold emphasis, "
        "and tables when comparison or figures are clearer in a table. Keep it grounded in the "
        "retrieved context and never invent unsupported facts. Use the chat history only to "
        "understand what the user's follow-up refers to; do not treat chat history as evidence. "
        "If multiple context blocks discuss the same subject with different dates, years, figures, "
        "names, locations, rules, thresholds, statuses, assumptions, sources, pages, or exceptions, "
        "include each relevant version instead of using only the first one. When the blocks disagree "
        "or cover different conditions, say so clearly and compare the differences."
    )

    if query_type == "summary":
        system_message = (
            "You are Texmin AI's warm, expressive document analyst. Summarize only from the "
            "supplied context. For broad summary requests, synthesize the major themes across "
            "the retrieved excerpts. If coverage is partial, mention that briefly near the end "
            "as a caveat, not as an opening refusal. Make the answer feel alive, clear, and "
            f"presentation-ready. {shared_style} {{language_instruction}}"
        )
    else:
        system_message = (
            "You are Texmin AI's warm, expressive QA assistant. Answer only from the supplied "
            "context. Speak naturally, as if a thoughtful human is explaining it out loud. "
            "Let sentences feel clear, complete, conversational, and suitable for future lip "
            f"sync. {shared_style} First identify the exact entity, topic, and constraints the "
            "user is asking about, using chat history only for pronouns or follow-up context. Then "
            "check whether the retrieved context contains multiple relevant versions or conflicting "
            "details; if it does, present the comparison before drawing a conclusion. If the context "
            "does not support the answer, say that simply and ask for the missing detail. "
            "{language_instruction}"
        )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            (
                "human",
                "Conversation so far, summarized to recent turns:\n{chat_history}\n\n"
                "Retrieved document context:\n{context}\n\n"
                "Current question:\n{question}\n\nAnswer:",
            ),
        ]
    )


def answer_question(
    question: str,
    top_k: int = 8,
    temperature: float = 0.55,
    chat_history: list[ChatMessage] | None = None,
) -> QAResponse:
    _load_environment()
    if _is_general_query(question):
        return _general_response(question)

    vector_store = _vector_store()
    query_type = "summary" if _is_summary_query(question) else "document"
    retrieval_query = _history_aware_query(question, chat_history)

    if query_type == "summary":
        docs = _retrieve_summary_context(vector_store, retrieval_query, top_k)
    else:
        docs = _retrieve_context(vector_store, retrieval_query, top_k)

    if not docs:
        return _not_enough_context_response(question)

    language_style = _detect_language_style(question)
    chain = _prompt_for_query(query_type) | _llm(temperature) | StrOutputParser()
    answer = chain.invoke(
        {
            "context": _format_context(docs),
            "chat_history": _format_chat_history(chat_history),
            "question": question,
            "language_instruction": _language_instruction(language_style),
        }
    )

    return QAResponse(answer=answer.strip(), sources=_source_chunks(docs), query_type=query_type)


def stream_answer_events(
    question: str,
    top_k: int = 8,
    temperature: float = 0.55,
    chat_history: list[ChatMessage] | None = None,
) -> Iterator[dict]:
    _load_environment()
    if _is_general_query(question):
        response = _general_response(question)
        yield {
            "type": "done",
            "answer": response.answer,
            "sources": _source_dicts(response.sources),
            "query_type": response.query_type,
        }
        return

    yield {"type": "status", "message": "Searching your document library"}
    vector_store = _vector_store()
    query_type = "summary" if _is_summary_query(question) else "document"
    retrieval_query = _history_aware_query(question, chat_history)

    if query_type == "summary":
        docs = _retrieve_summary_context(vector_store, retrieval_query, top_k)
    else:
        docs = _retrieve_context(vector_store, retrieval_query, top_k)

    if not docs:
        response = _not_enough_context_response(question)
        yield {
            "type": "done",
            "answer": response.answer,
            "sources": _source_dicts(response.sources),
            "query_type": response.query_type,
        }
        return

    language_style = _detect_language_style(question)
    chain = _prompt_for_query(query_type) | _llm(temperature) | StrOutputParser()
    inputs = {
        "context": _format_context(docs),
        "chat_history": _format_chat_history(chat_history),
        "question": question,
        "language_instruction": _language_instruction(language_style),
    }

    answer_parts = []
    yield {"type": "status", "message": "Writing the answer"}
    for token in chain.stream(inputs):
        if not token:
            continue
        answer_parts.append(token)
        yield {"type": "token", "text": token}

    yield {
        "type": "done",
        "answer": "".join(answer_parts).strip(),
        "sources": _source_dicts(_source_chunks(docs)),
        "query_type": query_type,
    }
