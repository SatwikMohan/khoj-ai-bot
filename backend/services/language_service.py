"""Offline query language identification shared by answers and speech."""

import re
from dataclasses import dataclass
from functools import lru_cache

import config


SCRIPT_LANGUAGES = (
    ("\u0980", "\u09ff", "bn"), ("\u0a00", "\u0a7f", "pa"),
    ("\u0a80", "\u0aff", "gu"), ("\u0b00", "\u0b7f", "or"),
    ("\u0b80", "\u0bff", "ta"), ("\u0c00", "\u0c7f", "te"),
    ("\u0c80", "\u0cff", "kn"), ("\u0d00", "\u0d7f", "ml"),
    ("\u0600", "\u06ff", "ar"), ("\u0400", "\u04ff", "ru"),
    ("\u0370", "\u03ff", "el"), ("\u3040", "\u30ff", "ja"),
    ("\uac00", "\ud7af", "ko"), ("\u4e00", "\u9fff", "zh"),
)
ROMAN_HINDI = {
    "aap", "hai", "hain", "kya", "kaise", "kaisa", "kaisi", "mujhe",
    "batao", "bataye", "bataiye", "kripya", "dhanyavad", "shukriya",
    "namaste", "nahi", "mera", "meri", "hoga", "karo", "samjhao",
    "chahiye", "mein", "se", "ka", "ki", "ke", "yaar", "kal",
    "jaana", "jana", "karna", "aaj", "aur", "ko", "yaad", "dilao",
    "yeh", "ye", "woh", "voh", "iska", "iski", "iske", "isme",
    "unhe", "unko", "agar", "lekin", "isliye", "kyunki", "liye",
    "sawaal", "jawab", "sahi", "galat", "zaroori", "zaruri",
    "hota", "hoti", "hote", "hogi", "honge", "karta", "karti",
    "rakha", "rakhi", "rakhe", "gaya", "gayi", "gaye", "diya",
    "dikhata", "dikhati", "batata", "batati", "samajh", "sakte",
}
ROMAN_HINDI_STRONG = ROMAN_HINDI - {"ka", "ki", "ke", "se", "ko", "mein", "aur", "kal"}
ENGLISH_CLUES = {"the", "and", "is", "are", "what", "which", "with", "from", "this", "that", "their", "into", "before", "after"}
SHORT_LANGUAGE_WORDS = {
    "hola": "es", "gracias": "es", "bonjour": "fr", "merci": "fr",
    "ciao": "it", "hallo": "de", "namaste": "hinglish",
    "shukriya": "hinglish", "dhanyavad": "hinglish",
}
LATIN_CLUES = {
    "es": {"explique", "estructura", "catálogo", "catalogo", "cuál", "cómo", "quiero", "dame", "años"},
    "pt": {"qual", "estrutura", "catálogo", "catalogo", "quais", "pode", "diga", "tabela"},
    "fr": {"quelle", "quelles", "catalogue", "expliquez", "donnez", "sont", "avec"},
    "de": {"wie", "struktur", "katalogs", "welche", "bitte", "sind"},
    "it": {"struttura", "spiega", "quale", "sono", "tabella"},
}


@lru_cache(maxsize=1)
def _classifier():
    try:
        from langid.langid import LanguageIdentifier, model
    except ImportError:
        return None
    return LanguageIdentifier.from_modelstring(model, norm_probs=True)


def detect_language(text: str, fallback: str = "en") -> str:
    """Use the query, with conservative fallback for short technical questions."""
    text = text.strip()
    if not text:
        return fallback
    if re.search(r"[\u0900-\u097f]", text):
        return "hi"
    for start, end, code in SCRIPT_LANGUAGES:
        if any(start <= char <= end for char in text):
            return code
    words = re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower())
    if len(words) == 1 and words[0] in SHORT_LANGUAGE_WORDS:
        return SHORT_LANGUAGE_WORDS[words[0]]
    hindi_words = sum(word in ROMAN_HINDI for word in words)
    if hindi_words >= 2 or (hindi_words and len(words) <= 2):
        return "hinglish"
    # Abbreviations and identifiers dominate very short questions. Avoid a
    # language guess based on one product name or schema identifier.
    if len(words) < 4:
        return fallback
    if len(set(words) & ENGLISH_CLUES) >= 3:
        return "en"
    clue_scores = {code: len(set(words) & clues) for code, clues in LATIN_CLUES.items()}
    best = max(clue_scores, key=clue_scores.get)
    if clue_scores[best] >= 2 and sum(score == clue_scores[best] for score in clue_scores.values()) == 1:
        return best
    classifier = _classifier()
    if classifier is None:
        return fallback
    code, confidence = classifier.classify(text)
    return code if confidence >= 0.7 else fallback


def response_language(question: str) -> str:
    return resolve_language_context(question).target_language


def language_instruction(code: str) -> str:
    if code == "hinglish":
        return (
            "Reply in natural Roman-script Hinglish: use conversational Hindi sentence structure "
            "written in Latin letters, with English technical terms where useful. For example, "
            "'Yeh rule safety ke liye hai' or 'Iska matlab daily inspection zaroori hai.' "
            "A fully English explanation is not Hinglish. Do not switch to Devanagari unless "
            "the message does. Keep product names, code, acronyms and identifiers exactly as written."
        )
    if code == "hi":
        return (
            "Reply in grammatical Hindi in Devanagari with correct spelling and matras. "
            "Keep English technical terms, product names, code, acronyms and identifiers unchanged."
        )
    if code == "en":
        return "Reply in grammatical English with standard spelling. Keep technical identifiers unchanged."
    return f"Reply in the user's language ({code}). Keep technical identifiers unchanged."


@dataclass(frozen=True)
class LanguageContext:
    original_input: str
    transcript: str | None
    detected_language: str
    target_language: str
    script: str
    selected_language: str | None = None
    confidence: float | None = None
    retrieved_languages: tuple[str, ...] = ()
    tts_language: str = ""
    tts_engine: str = ""
    tts_voice: str = ""


def resolve_language_context(
    question: str, *, transcript: str | None = None, selected_language: str | None = None,
) -> LanguageContext:
    """Resolve the current query once; retrieved text and history cannot change it."""
    current = (transcript if transcript is not None else question).strip()
    detected = detect_language(current)
    configured = config.RESPONSE_LANGUAGE.strip().lower()
    selected = (selected_language or "").strip().lower() or None
    target = selected or (detected if configured == "auto" else configured)
    target = {"english": "en", "hindi": "hi"}.get(target, target)
    script = "Devanagari" if re.search(r"[\u0900-\u097f]", current) else "Latin"
    return LanguageContext(
        question, transcript, detected, target, script, selected,
        tts_language=target, tts_engine=config.TTS_ENGINE, tts_voice=config.TTS_VOICE,
    )


def answer_language_matches(text: str, expected: str) -> bool:
    """Conservative guard against obvious script/language switches.

    Ambiguous short text and technical identifiers are not treated as evidence
    of a mismatch. This is a guard, not a claim of full language identification.
    """
    expected = {"english": "en", "hindi": "hi"}.get(expected.lower(), expected.lower())
    body = re.sub(r"`[^`]*`|https?://\S+", " ", text)
    letters = [char for char in body if char.isalpha()]
    if len(letters) < 10:
        return True
    devanagari = sum("\u0900" <= char <= "\u097f" for char in letters)
    if expected == "hi":
        return devanagari / len(letters) >= 0.18
    if expected == "en":
        if devanagari / len(letters) >= 0.18:
            return False
        words = re.findall(r"[a-zA-Z\u00c0-\u00ff]+", body.lower())
        if len(words) >= 8:
            classifier = _classifier()
            if classifier:
                detected, confidence = classifier.classify(body)
                if detected != "en" and confidence >= 0.95:
                    return False
        return True
    if expected == "hinglish":
        words = re.findall(r"[a-zA-Z]+", body.lower())
        hindi = sum(word in ROMAN_HINDI for word in words)
        strong_hindi = sum(word in ROMAN_HINDI_STRONG for word in words)
        return devanagari / len(letters) < 0.18 and (hindi >= 2 or strong_hindi >= 1)
    if len(letters) >= 35:
        classifier = _classifier()
        if classifier:
            detected, confidence = classifier.classify(body)
            if confidence >= 0.95 and detected != expected:
                return False
    return True
