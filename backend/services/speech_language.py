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
# Spell initialisms, including mining terms, instead of phonemizing them as words.
ABBREVIATION = re.compile(r"\b(?:DGMS|MMDR|MCDR|CMR|GSI|IBM|GPU|CPU|API|PDF|URL|RAM|LLM|TTS|STT|AI)\b")
HINDI_LETTER_NAMES = {
    "A": "\u090f", "B": "\u092c\u0940", "C": "\u0938\u0940", "D": "\u0921\u0940",
    "E": "\u0908", "F": "\u090f\u092b", "G": "\u091c\u0940", "H": "\u090f\u091a",
    "I": "\u0906\u0908", "J": "\u091c\u0947", "K": "\u0915\u0947", "L": "\u090f\u0932",
    "M": "\u090f\u092e", "N": "\u090f\u0928", "O": "\u0913", "P": "\u092a\u0940",
    "Q": "\u0915\u094d\u092f\u0942", "R": "\u0906\u0930", "S": "\u090f\u0938",
    "T": "\u091f\u0940", "U": "\u092f\u0942", "V": "\u0935\u0940", "W": "\u0921\u092c\u094d\u0932\u094d\u092f\u0942",
    "X": "\u090f\u0915\u094d\u0938", "Y": "\u0935\u093e\u0908", "Z": "\u095b\u0947\u0921",
}
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
    def letters(word: str) -> str:
        return " ".join(HINDI_LETTER_NAMES[letter] if hindi else letter
                        for letter in word)
    text = ABBREVIATION.sub(lambda m: letters(m.group()), text)
    text = CLOCK_SUFFIX.sub(lambda m: f"{m.group(1)} {letters(m.group(2).upper())}", text)
    return text

