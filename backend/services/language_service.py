"""Offline query language identification shared by answers and speech."""

import re
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
}
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
    configured = config.RESPONSE_LANGUAGE.strip().lower()
    return detect_language(question) if configured == "auto" else configured


def language_instruction(code: str) -> str:
    if code == "hinglish":
        return (
            "Reply in natural Roman-script Hinglish with consistent Hindi spellings and standard "
            "English technical terms. Do not switch to Devanagari unless the message does. "
            "Keep product names, code, acronyms and identifiers exactly as written."
        )
    if code == "hi":
        return (
            "Reply in grammatical Hindi in Devanagari with correct spelling and matras. "
            "Keep English technical terms, product names, code, acronyms and identifiers unchanged."
        )
    if code == "en":
        return "Reply in grammatical English with standard spelling. Keep technical identifiers unchanged."
    return f"Reply in the user's language ({code}). Keep technical identifiers unchanged."
