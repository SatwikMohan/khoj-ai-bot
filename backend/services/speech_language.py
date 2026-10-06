"""Prepare speakable text while keeping one local speaker per answer."""

import re

from services.language_service import detect_language


DEVANAGARI = re.compile(r"[\u0900-\u097f]")


def speech_phrases(text: str, language: str | None = None) -> list[tuple[str, str]]:
    """Return one synthesis job for one speaker, including mixed-language text.

    Piper's Hindi and English checkpoints are different people. Switching at
    each word made one answer alternate between male and female recordings.
    Keep the reply language fixed for every streamed chunk of that answer.
    """
    text = text.strip()
    if not text:
        return []
    language = (language or detect_language(text)).lower()
    if language in {"hi", "hindi"}:
        voice = "hi"
    elif language in {"en", "english", "hinglish"}:
        voice = "en"
    else:
        voice = language
    return [(voice, text)]


ISO_DATE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
RUPEES = re.compile(r"₹\s*(\d+(?:[.,]\d+)*)")
DOLLARS = re.compile(r"\$\s*(\d+(?:[.,]\d+)*)")
ABBREVIATION = re.compile(r"\b(?:GPU|CPU|API|PDF|URL|RAM|LLM|TTS|STT)\b")
CLOCK_SUFFIX = re.compile(r"\b(\d{1,2})\s*(AM|PM)\b", re.I)
MONTHS_EN = ("January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December")
MONTHS_HI = ("जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
             "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर")


def normalize_speech_text(text: str, language: str | None = None) -> str:
    """Expand only unambiguous speech forms; leave displayed chat untouched."""
    language = (language or detect_language(text)).lower()
    if language not in {"hi", "hindi", "hinglish", "en", "english"}:
        return text
    hindi = language in {"hi", "hindi", "hinglish"} or bool(DEVANAGARI.search(text))
    months = MONTHS_HI if hindi else MONTHS_EN
    text = ISO_DATE.sub(
        lambda m: f"{int(m.group(3))} {months[int(m.group(2)) - 1]} {m.group(1)}",
        text,
    )
    text = RUPEES.sub(lambda m: f"{m.group(1)} {'रुपये' if hindi else 'rupees'}", text)
    text = DOLLARS.sub(lambda m: f"{m.group(1)} {'डॉलर' if hindi else 'dollars'}", text)
    text = ABBREVIATION.sub(lambda m: " ".join(m.group()), text)
    text = CLOCK_SUFFIX.sub(lambda m: f"{m.group(1)} {' '.join(m.group(2).upper())}", text)
    return text

