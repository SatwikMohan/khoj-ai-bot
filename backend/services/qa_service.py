import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from helpers.request_models import ChatMessage, QAResponse, SourceChunk
from services.embedding_service import (
    PromptedOllamaEmbeddings,
    embedding_profile,
    model_is_available,
    ollama_model_names,
)
from services.lexical_service import lexical_search
from services.reranker_service import rerank_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
DEFAULT_VECTOR_DB_DIR = REPO_ROOT / "vector_db"
DEFAULT_RELEVANCE_THRESHOLD = 0.15
DEFAULT_CONTEXT_MAX_CHARS = 16000
DEFAULT_RETRIEVAL_FETCH_K = 40
DEFAULT_CHAT_HISTORY_MAX_TURNS = 4
DEFAULT_CHAT_HISTORY_MAX_CHARS = 1200
DEFAULT_CHAT_HISTORY_MESSAGE_CHARS = 320
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
    "आप कैसे हैं",
    "तुम कैसे हो",
    "आप कौन हैं",
    "तुम कौन हो",
    "आप क्या कर सकते हैं",
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
INTERNAL_ANALYSIS_PREFIXES = (
    "okay, let me unpack",
    "okay, let's unpack",
    "okay, let me break this down",
    "okay, let's break this down",
    "the user asked",
    "the user is asking",
    "looking at the context",
    "looking at the working notes",
    "the working notes",
    "working notes mention",
    "hmm",
    "we need to",
    "i need to",
    "let's craft",
    "the question asks",
)
INTERNAL_ANALYSIS_MARKERS = (
    "the user just said",
    "the user asked",
    "the user seems to",
    "from the working notes",
    "looking at the context",
    "looking at the working notes",
    "context snippets",
    "conversation history",
    "i should give",
    "i'll keep it",
    "the key points from the notes",
)
INTERNAL_ANALYSIS_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\b(?:the|this) user\b",
        r"\buser (?:asked|said|wants?|needs?|seems?|requested)\b",
        r"\bthey (?:asked|said|want|need|seem|requested)\b",
        r"\b(?:he|she) (?:asked|said|wants?|needs?|seems?|requested)\b",
        r"\b(?:the person|the questioner) (?:asked|said|wants?|needs?|seems?|requested)\b",
        r"\b(?:based on|from|looking at|according to) (?:the )?(?:private |provided |retrieved |supplied )?(?:context|notes|snippets|documents?|facts|evidence|reference material)\b",
        r"\b(?:working notes|context snippets|retrieved context|provided context|supplied material|conversation history|conversation memory|chat history)\b",
        r"\b(?:the )?(?:key points|information) (?:from|in) (?:the )?(?:notes|context|documents?)\b",
        r"\bi (?:see|can see|notice) (?:that )?(?:the )?(?:context|notes|documents?|snippets?)\b",
        r"\blooking at (?:the )?(?:references?|mcqs?|results?|material)\b",
        r"\breference \d+ (?:mentions?|states?|says?|contains?|shows?)\b",
        r"\b(?:wait,? )?the question (?:is|asks?)\b",
        r"\b(?:i need to|i should|i will|i'll|we need to|let me|let's) (?:answer|explain|respond|give|craft|keep|break|unpack|analy[sz]e|check|look|identify|review|find)\b",
        r"\b(?:my|the) (?:reasoning|analysis|thought process)\b",
    )
)


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
    configured_style = os.getenv("RESPONSE_LANGUAGE", "auto").strip().lower()
    if configured_style in {"english", "hindi", "hinglish"}:
        return configured_style

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
        if os.getenv("HINGLISH_SCRIPT", "mixed").strip().lower() == "mixed":
            return (
                "Respond in natural spoken Hinglish. Write Hindi words in Devanagari and "
                "keep English words in Latin script so the offline Indic voice pronounces "
                "both clearly. Keep it warm and use light Markdown only where it helps."
            )
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
    if any(term in normalized for term in SUMMARY_TERMS):
        return True
    words = normalized.split()
    return len(words) <= 10 and normalized.startswith(
        ("explain ", "describe ", "walk me through ", "lets talk about ", "let's talk about ")
    )


def _looks_like_internal_analysis(text: str) -> bool:
    probe = re.sub(r"\s+", " ", text or "").strip().lower()
    return (
        "<think>" in probe
        or probe.startswith(INTERNAL_ANALYSIS_PREFIXES)
        or any(marker in probe[:1200] for marker in INTERNAL_ANALYSIS_MARKERS)
        or any(pattern.search(probe[:1600]) for pattern in INTERNAL_ANALYSIS_PATTERNS)
    )


def _clean_model_answer(raw_answer: str) -> str:
    text = raw_answer or ""
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    tagged_answers = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if tagged_answers:
        text = tagged_answers[-1]
    else:
        text = re.sub(r"</?answer>", "", text, flags=re.IGNORECASE)

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    text = "\n\n".join(paragraphs)
    text = re.sub(r"^(?:final\s+)?answer\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    return text.strip()


def _direct_answer_content(raw_answer: str) -> str:
    answer = _clean_model_answer(raw_answer)
    if not answer:
        return ""

    direct_parts = []
    for part in re.split(r"(?<=[.!?।])(?:\s+|$)|\n+", answer):
        candidate = part.strip()
        if not candidate:
            continue
        candidate = re.sub(
            r"^(?:(?:based on|according to|from|looking at) "
            r"(?:the )?(?:private |provided |retrieved |supplied )?(?:context|notes|snippets|documents?|facts|evidence|reference material)"
            r"|(?:the )?(?:context|notes|documents?) (?:show|shows|state|states|say|says|indicate|indicates) that)"
            r"\s*[,;:]?\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        if candidate and not _looks_like_internal_analysis(candidate):
            direct_parts.append(candidate)
    return " ".join(direct_parts).strip()


def _safe_generated_answer(raw_answer: str, question: str) -> str:
    answer = _direct_answer_content(raw_answer)
    if answer:
        return answer
    language_style = _detect_language_style(question)
    if language_style == "hindi":
        return "इसका भरोसेमंद जवाब देने के लिए मुझे थोड़ा और स्पष्ट विवरण चाहिए। आप किस हिस्से के बारे में जानना चाहते हैं?"
    if language_style == "hinglish":
        return "Iska reliable jawab dene ke liye mujhe thodi aur clear detail chahiye. Aap kis part ke baare mein jaanna chahte hain?"
    return "I need a little more specific detail to answer reliably. Which part would you like to focus on?"


def _repair_generated_answer(
    raw_answer: str,
    question: str,
    context: str,
    language_style: str,
) -> str:
    repair_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Return only a direct final reply to the person speaking. Do not mention a user, "
                "context, notes, documents, instructions, reasoning, or the process of answering. "
                "Do not narrate what you are doing. Address the person as 'you' when needed. "
                "Use only supported facts. {language_instruction}",
            ),
            (
                "human",
                "Private reference material:\n{context}\n\nQuestion:\n{question}\n\n"
                "Write the direct reply now. Start immediately with the useful information:",
            ),
        ]
    )
    # This is already a recovery pass. Reasoning here is slower and can
    # reproduce the empty-content failure that caused the recovery.
    chain = repair_prompt | _llm(0.1, reasoning=False) | StrOutputParser()
    repaired = chain.invoke(
        {
            "context": context,
            "question": question,
            "language_instruction": _language_instruction(language_style),
        }
    )
    direct_answer = _direct_answer_content(repaired)
    return direct_answer or _safe_generated_answer(raw_answer, question)


def _target_filter(question: str) -> dict | None:
    return None


def _general_response(question: str) -> QAResponse:
    language_style = _detect_language_style(question)
    normalized = _normalized_query(question)
    assistant_name = os.getenv("ASSISTANT_NAME", "Khoj").strip() or "Khoj"
    is_wellbeing = normalized in {
        "how are you",
        "how r u",
        "kaise ho",
        "aap kaise ho",
        "tum kaise ho",
        "आप कैसे हैं",
        "तुम कैसे हो",
    }
    is_identity = normalized in {
        "who are you",
        "what are you",
        "what is your name",
        "tell me about yourself",
        "aap kaun ho",
        "tum kaun ho",
        "आप कौन हैं",
        "तुम कौन हो",
    }
    is_capability = normalized in {
        "what can you do",
        "kya kar sakte ho",
        "आप क्या कर सकते हैं",
    }
    is_help = normalized in {"can you help me", "help me", "madad kar sakte ho"}

    if normalized in {"thanks", "thank you", "thankyou", "dhanyavad", "shukriya"}:
        if language_style == "hindi":
            answer = "ज़रूर। जब चाहें, अगला सवाल पूछिए।"
        elif language_style == "hinglish":
            answer = "Bilkul—jab chahein, agla sawaal pooch lijiye."
        else:
            answer = "Anytime. What would you like to look at next?"
    elif normalized in {"bye", "goodbye"}:
        if language_style == "hindi":
            answer = "ठीक है, फिर मिलते हैं। अपना ख्याल रखिए।"
        elif language_style == "hinglish":
            answer = "Theek hai, phir milte hain. Apna khayal rakhiye."
        else:
            answer = "See you soon. Take care."
    elif language_style == "hindi":
        if is_wellbeing:
            answer = "मैं बढ़िया हूँ—आपसे बात करके अच्छा लगा। बताइए, आज क्या देखना है?"
        elif is_identity:
            answer = f"मैं {assistant_name} हूँ। जटिल दस्तावेज़ों को समझने, तुलना करने और सीधा जवाब देने में आपकी मदद करता हूँ।"
        elif is_capability:
            answer = "मैं आपके दस्तावेज़ खोज सकता हूँ, नियमों और आँकड़ों की तुलना कर सकता हूँ, और बात को आसान भाषा में समझा सकता हूँ।"
        elif is_help:
            answer = "बिल्कुल। बताइए कहाँ अटके हैं, हम वहीं से शुरू करते हैं।"
        else:
            answer = "नमस्ते—आपसे बात करके अच्छा लगा। बताइए, आज क्या जानना है?"
    elif language_style == "hinglish":
        if is_wellbeing:
            answer = "Main badhiya hoon—aur aapse baat karke achha laga. Bataiye, aaj kya dekhna hai?"
        elif is_identity:
            answer = f"Main {assistant_name} hoon. Complex documents samajhne, compare karne aur seedha jawab nikalne mein aapki help karta hoon."
        elif is_capability:
            answer = "Main aapke documents search kar sakta hoon, rules aur figures compare kar sakta hoon, aur simple language mein samjha sakta hoon."
        elif is_help:
            answer = "Bilkul. Bataiye kahan atke hain—wahin se shuru karte hain."
        else:
            answer = "Namaste—achha laga aapse baat karke. Bataiye, aaj kya dekhna hai?"
    elif is_wellbeing:
        answer = "I’m doing well—glad you’re here. What are we looking into today?"
    elif is_identity:
        answer = f"I’m {assistant_name}. I help you make sense of complex documents, compare details, and get to a clear answer quickly."
    elif is_capability:
        answer = "I can search your documents, compare rules and figures, summarize long material, and explain the answer in plain language."
    elif is_help:
        answer = "Of course. Tell me where you’re stuck, and we’ll start there."
    else:
        answer = "Hey—good to have you here. What would you like to dig into?"

    return QAResponse(answer=answer, sources=[], query_type="general")


def _topic_opener_response(question: str) -> QAResponse | None:
    match = re.fullmatch(
        r"\s*(?:(?:let(?:'|’)s|lets|let us|can we|could we)\s+talk\s+about|i (?:want|would like) to talk about)\s+(.+?)\s*[.!?]*\s*",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    topic = re.sub(r"\s+", " ", match.group(1)).strip()
    if _normalized_query(topic) == "dgms":
        answer = (
            "Sure. DGMS—the Directorate General of Mines Safety—is India’s mine-safety "
            "regulator. We can talk through its rules, circulars, inspections, accident reporting, "
            "or the duties of mine owners and managers. Where would you like to start?"
        )
    else:
        answer = f"Sure—what would you like to explore about {topic}?"
    return QAResponse(answer=answer, sources=[], query_type="general")


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


def _active_index() -> tuple[str, str]:
    db_dir = _vector_db_dir()
    manifest_path = db_dir / "ingestion_manifest.json"
    if not manifest_path.exists():
        raise QAEngineError(
            f"Index manifest not found at {manifest_path}. Run `python train_engine.py` first."
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QAEngineError(f"Index manifest is unreadable: {exc}") from exc

    collection_name = manifest.get("active_collection")
    indexed_profile = manifest.get("settings", {}).get("embedding_profile", {})
    configured_profile = embedding_profile()
    if not collection_name or indexed_profile.get("fingerprint") != configured_profile.fingerprint:
        indexed_model = manifest.get("settings", {}).get("embedding_model", "unknown")
        raise QAEngineError(
            "The vector index is incompatible with the configured embedding model. "
            f"Index={indexed_model}, configured={configured_profile.model}. "
            "Build a new blue-green index with `python train_engine.py`."
        )
    return str(collection_name), configured_profile.fingerprint


@lru_cache(maxsize=4)
def _embeddings(profile_fingerprint: str) -> PromptedOllamaEmbeddings:
    profile = embedding_profile()
    if profile.fingerprint != profile_fingerprint:
        raise QAEngineError("Embedding configuration changed while the service was running.")
    return PromptedOllamaEmbeddings(
        profile=profile,
        base_url=_ollama_base_url(),
        keep_alive=_ollama_keep_alive(),
    )


@lru_cache(maxsize=4)
def _vector_store_for(collection_name: str, profile_fingerprint: str, db_dir: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=_embeddings(profile_fingerprint),
        persist_directory=db_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _vector_store() -> Chroma:
    _load_environment()
    db_dir = _vector_db_dir()
    if not db_dir.exists():
        raise QAEngineError(
            f"Vector database not found at {db_dir}. Run `python train_engine.py` first."
        )
    collection_name, profile_fingerprint = _active_index()
    return _vector_store_for(collection_name, profile_fingerprint, str(db_dir))


@lru_cache(maxsize=12)
def _chat_llm(
    model: str,
    base_url: str,
    temperature: float,
    reasoning: bool,
) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=base_url,
        reasoning=reasoning,
        temperature=temperature,
        keep_alive=_ollama_keep_alive(),
        num_predict=_env_int("OLLAMA_NUM_PREDICT", 384),
        num_ctx=_env_int("OLLAMA_NUM_CTX", 8192),
        sync_client_kwargs={
            "timeout": _env_int("OLLAMA_REQUEST_TIMEOUT_SECONDS", 180)
        },
    )


def _llm(temperature: float, reasoning: bool | None = None) -> ChatOllama:
    _load_environment()
    if reasoning is None:
        reasoning = _env_bool("QA_REASONING_ENABLED", False)
    return _chat_llm(
        os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:35b"),
        _ollama_base_url(),
        round(float(temperature), 2),
        reasoning,
    )


def warm_up_qa_engine() -> None:
    _load_environment()
    if not _env_bool("QA_WARMUP_ON_STARTUP", True):
        return

    installed_models = ollama_model_names(_ollama_base_url())
    profile = embedding_profile()
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:35b")
    missing = [
        model
        for model in (profile.model, chat_model)
        if not model_is_available(model, installed_models)
    ]
    if missing:
        raise QAEngineError(
            "Required Ollama models are not installed: "
            f"{', '.join(missing)}. Pull them before starting the backend."
        )

    _vector_store()
    _, profile_fingerprint = _active_index()
    _embeddings(profile_fingerprint).embed_query("warm up document retrieval")
    _llm(0.1, reasoning=False).invoke("Reply with: ready")


def runtime_status() -> dict:
    _load_environment()
    profile = embedding_profile()
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:35b")
    status = {
        "status": "ok",
        "ollama": "unavailable",
        "chat_model": chat_model,
        "embedding_model": profile.model,
        "index": "unavailable",
    }
    try:
        installed = ollama_model_names(_ollama_base_url())
        missing = [
            model
            for model in (profile.model, chat_model)
            if not model_is_available(model, installed)
        ]
        status["ollama"] = "ready" if not missing else "missing_models"
        status["missing_models"] = missing
    except RuntimeError as exc:
        status["status"] = "degraded"
        status["ollama_error"] = str(exc)

    try:
        collection_name, _ = _active_index()
        status["index"] = "ready"
        status["collection"] = collection_name
    except QAEngineError as exc:
        status["status"] = "degraded"
        status["index_error"] = str(exc)
    if status["ollama"] != "ready" or status["index"] != "ready":
        status["status"] = "degraded"
    return status


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
        block = f"[Reference {index}: {source}{folder_label}{page_label}{year_label}]\n{content}"
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
        if role == "assistant" and _looks_like_internal_analysis(content):
            continue
        if not content:
            continue
        label = "You said" if role == "user" else "I replied"
        line = f"{label}: {content}"
        if used_chars + len(line) > max_chars:
            break
        lines.append(line)
        used_chars += len(line)

    if not lines:
        return "No earlier conversation."
    return "\n".join(reversed(lines))


def _history_aware_query(question: str, chat_history: list[ChatMessage] | None) -> str:
    normalized = _normalized_query(question)
    follow_up_markers = {
        "it",
        "its",
        "that",
        "this",
        "they",
        "them",
        "those",
        "these",
        "he",
        "she",
        "there",
        "same",
        "previous",
        "above",
        "unka",
        "uska",
        "iske",
        "uske",
    }
    words = set(re.findall(r"\w+", normalized, flags=re.UNICODE))
    explicit_follow_up_phrases = (
        "i want to know about the ",
        "i want to know about those ",
        "i want to know about these ",
        "tell me about the ",
        "tell me about those ",
        "tell me about these ",
        "what about ",
        "how about ",
        "and what about ",
    )
    is_follow_up = len(words) <= 12 and (
        bool(words & follow_up_markers)
        or normalized.startswith(explicit_follow_up_phrases)
    )
    if not is_follow_up:
        return question

    recent_user_messages = []
    clean_assistant_messages = []
    for message in (chat_history or [])[-6:]:
        role = _message_role(message).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _trim_text(_message_content(message), 180)
        if not content:
            continue
        if role == "user" and not _is_general_query(content):
            recent_user_messages.append(content)
        elif role == "assistant" and not _looks_like_internal_analysis(content):
            clean_assistant_messages.append(content)

    topic_messages = recent_user_messages[-2:] or clean_assistant_messages[-1:]
    if not topic_messages:
        return question

    history_text = " ".join(topic_messages)
    return _trim_text(
        f"Current question: {question}. Recent conversation topic: {history_text}", 900
    )


def _hybrid_rank(dense_docs: list, lexical_docs: list, limit: int):
    rrf_constant = _env_int("HYBRID_RRF_K", 60)
    scores: dict[tuple, float] = {}
    documents = {}

    for rank, doc in enumerate(dense_docs, start=1):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_constant + rank)
        documents[key] = doc

    for rank, doc in enumerate(lexical_docs, start=1):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_constant + rank)
        documents.setdefault(key, doc)

    ranked = sorted(documents.values(), key=lambda doc: scores[_doc_key(doc)], reverse=True)
    for doc in ranked:
        doc.metadata["hybrid_score"] = round(scores[_doc_key(doc)], 6)
    return ranked[:limit]


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
    relevant_by_key = {}
    dense_docs = []
    for doc, score in scored_docs:
        if not _has_meaningful_content(doc):
            continue
        doc.metadata["relevance_score"] = round(float(score), 4)
        dense_docs.append(doc)
        if score >= threshold:
            relevant_by_key[_doc_key(doc)] = doc

    if _env_bool("HYBRID_SEARCH_ENABLED", True):
        collection_name, _ = _active_index()
        lexical_docs = lexical_search(
            _vector_db_dir(),
            collection_name,
            question,
            fetch_k,
        )
        lexical_docs = [doc for doc in lexical_docs if _has_meaningful_content(doc)]
        if dense_docs or lexical_docs:
            candidate_count = max(top_k, _env_int("RERANK_CANDIDATES", 24))
            candidates = _hybrid_rank(dense_docs, lexical_docs, candidate_count)
            selected = _diversify_docs(
                rerank_documents(question, candidates, top_k), top_k
            )
            print(
                json.dumps(
                    {
                        "event": "retrieval_completed",
                        "dense_candidates": len(dense_docs),
                        "above_threshold": len(relevant_by_key),
                        "lexical_candidates": len(lexical_docs),
                        "selected": len(selected),
                        "best_dense_score": (
                            dense_docs[0].metadata.get("relevance_score")
                            if dense_docs
                            else None
                        ),
                    }
                )
            )
            return selected

    if not relevant_by_key:
        if _env_bool("ALLOW_LOW_RELEVANCE_FALLBACK", True):
            fallback_docs = dense_docs
            return _diversify_docs(fallback_docs, top_k)
        return []

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
    summary_k = max(top_k, int(os.getenv("SUMMARY_CONTEXT_CHUNKS", "8")))
    fetch_k = max(summary_k, int(os.getenv("SUMMARY_FETCH_K", "32")))
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
    assistant_name = os.getenv("ASSISTANT_NAME", "Khoj").strip() or "Khoj"
    persona = os.getenv(
        "ASSISTANT_PERSONA",
        "a calm, perceptive colleague who explains difficult material in plain language",
    ).strip()
    shared_style = (
        f"You are {assistant_name}, {persona}. Reply directly to the person speaking with you. "
        "Output only the words you want them to read or hear. Never describe the request, your process, "
        "the conversation history, the supplied material, or how you plan to answer. Never refer to "
        "the person as 'the user' or as 'they'; address them naturally as 'you'. Start with the useful "
        "answer, not with 'okay, let me break this down'. Use a compact conversational paragraph for "
        "simple questions and bullets only when they genuinely help. Use only facts supported by the "
        "document material. Match every number to the exact noun and unit in the question; never "
        "substitute a related figure. For example, mineral blocks are not mines. If the material "
        "supports only a related metric, state that distinction instead of guessing. If information "
        "is missing, say exactly what is missing and ask one short question. Do not output analysis, "
        "planning, or thinking. Return the final reply immediately."
    )

    if query_type == "summary":
        system_message = (
            "Give a natural overview of the topic and invite the person to choose a specific area when "
            "their request is broad. Bring related facts together instead of listing excerpts. "
            f"{shared_style} {{language_instruction}}"
        )
    else:
        system_message = (
            f"{shared_style} Understand the exact entity and constraint being discussed before "
            "answering. If the notes don't support a reliable answer, say what is missing in one "
            "natural sentence and ask one focused follow-up question. "
            "{language_instruction}"
        )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            (
                "human",
                "Private conversation memory:\n{chat_history}\n\n"
                "Private reference material:\n{context}\n\nMessage to answer:\n{question}\n\n"
                "Reply directly now:",
            ),
        ]
    )


def answer_question(
    question: str,
    top_k: int = 5,
    temperature: float = 0.5,
    chat_history: list[ChatMessage] | None = None,
) -> QAResponse:
    started_at = time.perf_counter()
    _load_environment()
    if _is_general_query(question):
        response = _general_response(question)
        response.timings_ms = {"total": round((time.perf_counter() - started_at) * 1000, 1)}
        return response
    topic_opener = _topic_opener_response(question)
    if topic_opener:
        topic_opener.timings_ms = {"total": round((time.perf_counter() - started_at) * 1000, 1)}
        return topic_opener

    vector_store = _vector_store()
    query_type = "summary" if _is_summary_query(question) else "document"
    retrieval_query = _history_aware_query(question, chat_history)

    retrieval_started_at = time.perf_counter()
    if query_type == "summary":
        docs = _retrieve_summary_context(vector_store, retrieval_query, top_k)
    else:
        docs = _retrieve_context(vector_store, retrieval_query, top_k)

    if not docs:
        response = _not_enough_context_response(question)
        response.timings_ms = {
            "retrieval": round((time.perf_counter() - retrieval_started_at) * 1000, 1),
            "total": round((time.perf_counter() - started_at) * 1000, 1),
        }
        return response

    language_style = _detect_language_style(question)
    chain = _prompt_for_query(query_type) | _llm(temperature) | StrOutputParser()
    generation_started_at = time.perf_counter()
    formatted_context = _format_context(docs)
    raw_answer = chain.invoke(
        {
            "context": formatted_context,
            "chat_history": _format_chat_history(chat_history),
            "question": question,
            "language_instruction": _language_instruction(language_style),
        }
    )
    answer = _direct_answer_content(raw_answer)
    if not answer:
        answer = _repair_generated_answer(
            raw_answer, question, formatted_context, language_style
        )

    timings = {
        "retrieval": round((generation_started_at - retrieval_started_at) * 1000, 1),
        "generation": round((time.perf_counter() - generation_started_at) * 1000, 1),
        "total": round((time.perf_counter() - started_at) * 1000, 1),
    }
    print(json.dumps({"event": "qa_completed", "query_type": query_type, **timings}))
    return QAResponse(
        answer=answer,
        sources=_source_chunks(docs),
        query_type=query_type,
        timings_ms=timings,
    )


def stream_answer_events(
    question: str,
    top_k: int = 5,
    temperature: float = 0.5,
    chat_history: list[ChatMessage] | None = None,
) -> Iterator[dict]:
    started_at = time.perf_counter()
    _load_environment()
    if _is_general_query(question):
        response = _general_response(question)
        yield {
            "type": "done",
            "answer": response.answer,
            "sources": _source_dicts(response.sources),
            "query_type": response.query_type,
            "timings_ms": {"total": round((time.perf_counter() - started_at) * 1000, 1)},
        }
        return
    topic_opener = _topic_opener_response(question)
    if topic_opener:
        yield {
            "type": "done",
            "answer": topic_opener.answer,
            "sources": [],
            "query_type": topic_opener.query_type,
            "timings_ms": {"total": round((time.perf_counter() - started_at) * 1000, 1)},
        }
        return

    yield {"type": "status", "message": "Finding the most relevant information…"}
    vector_store = _vector_store()
    query_type = "summary" if _is_summary_query(question) else "document"
    retrieval_query = _history_aware_query(question, chat_history)

    retrieval_started_at = time.perf_counter()
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
            "timings_ms": {
                "retrieval": round((time.perf_counter() - retrieval_started_at) * 1000, 1),
                "total": round((time.perf_counter() - started_at) * 1000, 1),
            },
        }
        return

    language_style = _detect_language_style(question)
    chain = _prompt_for_query(query_type) | _llm(temperature) | StrOutputParser()
    formatted_context = _format_context(docs)
    inputs = {
        "context": formatted_context,
        "chat_history": _format_chat_history(chat_history),
        "question": question,
        "language_instruction": _language_instruction(language_style),
    }

    raw_answer_parts = []
    visible_answer_parts = []
    pending_visible_text = ""
    rejected_internal_analysis = False
    generation_started_at = time.perf_counter()
    first_token_ms = None
    yield {"type": "status", "message": "Thinking…"}
    for token in chain.stream(inputs):
        if not token:
            continue
        if first_token_ms is None:
            first_token_ms = round((time.perf_counter() - generation_started_at) * 1000, 1)
        raw_answer_parts.append(token)
        pending_visible_text += token
        has_sentence_boundary = bool(
            re.search(r"[.!?।](?:\s|$)|\n", pending_visible_text)
        )
        if not has_sentence_boundary and len(pending_visible_text) < 96:
            continue
        cleaned_segment = re.sub(
            r"</?(?:answer|think)>", "", pending_visible_text, flags=re.IGNORECASE
        )
        if _looks_like_internal_analysis(pending_visible_text):
            salvaged_segment = _direct_answer_content(pending_visible_text)
            if salvaged_segment:
                visible_answer_parts.append(salvaged_segment)
                yield {"type": "token", "text": salvaged_segment}
                pending_visible_text = ""
                continue
            rejected_internal_analysis = True
            pending_visible_text = ""
            continue
        if cleaned_segment:
            visible_answer_parts.append(cleaned_segment)
            yield {"type": "token", "text": cleaned_segment}
        pending_visible_text = ""

    raw_answer = "".join(raw_answer_parts)
    if rejected_internal_analysis:
        # The model can begin with private reasoning and still append a valid
        # final answer. Inspect the complete response instead of discarding
        # everything after the first rejected segment.
        answer = "".join(visible_answer_parts).strip() or _direct_answer_content(raw_answer)
        if not answer:
            answer = _repair_generated_answer(
                raw_answer, question, formatted_context, language_style
            )
            if answer:
                yield {"type": "token", "text": answer}
    else:
        cleaned_remainder = re.sub(
            r"</?(?:answer|think)>", "", pending_visible_text, flags=re.IGNORECASE
        )
        if cleaned_remainder:
            visible_answer_parts.append(cleaned_remainder)
            yield {"type": "token", "text": cleaned_remainder}
        answer = "".join(visible_answer_parts).strip()
        if not answer:
            answer = _repair_generated_answer(
                raw_answer, question, formatted_context, language_style
            )
            if not answer:
                raise QAEngineError(
                    "The answer model returned no usable text after one recovery attempt."
                )
            yield {"type": "token", "text": answer}

    if not answer.strip():
        raise QAEngineError(
            "The chat model completed without returning an answer. Retry once; if this "
            "continues, inspect the app and Ollama logs."
        )

    timings = {
        "retrieval": round((generation_started_at - retrieval_started_at) * 1000, 1),
        "first_token": first_token_ms or 0.0,
        "generation": round((time.perf_counter() - generation_started_at) * 1000, 1),
        "total": round((time.perf_counter() - started_at) * 1000, 1),
    }
    print(json.dumps({"event": "qa_stream_completed", "query_type": query_type, **timings}))
    yield {
        "type": "done",
        "answer": answer,
        "sources": _source_dicts(_source_chunks(docs)),
        "query_type": query_type,
        "timings_ms": timings,
    }
