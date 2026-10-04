"""Speech-only Hindi/English phrase routing for monolingual local TTS voices."""

import re

from services.language_service import detect_language

# Deliberately curated: an unknown Latin word stays Latin so product names and
# English words never get blindly transliterated into Devanagari.
ROMAN_HINDI_SPEECH = {
    "aaj": "आज", "aap": "आप", "aapka": "आपका", "aapki": "आपकी",
    "achha": "अच्छा", "accha": "अच्छा", "aur": "और",
    "bahut": "बहुत", "batao": "बताओ", "bataye": "बताइए",
    "chahiye": "चाहिए", "chalo": "चलो", "dhanyavad": "धन्यवाद",
    "hai": "है", "hain": "हैं", "ham": "हम", "hum": "हम",
    "ho": "हो", "hoga": "होगा", "kal": "कल", "ka": "का",
    "kaise": "कैसे", "kaisi": "कैसी", "kaisa": "कैसा",
    "kar": "कर", "kare": "करें", "karen": "करें", "karna": "करना",
    "karo": "करो", "ke": "के", "ki": "की", "ko": "को",
    "kripya": "कृपया", "kya": "क्या", "kyun": "क्यों",
    "mai": "मैं", "mujhe": "मुझे", "mera": "मेरा",
    "meri": "मेरी", "mein": "में", "nahi": "नहीं",
    "namaste": "नमस्ते", "par": "पर", "raha": "रहा",
    "rahi": "रही", "rahe": "रहे", "samjhao": "समझाओ",
    "se": "से", "shukriya": "शुक्रिया", "tha": "था",
    "thi": "थी", "tum": "तुम",
    "yaar": "यार", "jaana": "जाना", "jana": "जाना",
    "yaad": "याद", "dilao": "दिलाओ", "sakta": "सकता",
    "sakti": "सकती", "sakte": "सकते", "hu": "हूँ",
    "hun": "हूँ", "hoon": "हूँ", "yeh": "यह", "ye": "ये",
}
TOKEN = re.compile(r"\s+|[\u0900-\u097f]+|[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.:/]\d+)*|[^\s]")
DEVANAGARI = re.compile(r"[\u0900-\u097f]")
LATIN = re.compile(r"[A-Za-z]")


def speech_phrases(text: str, language: str | None = None) -> list[tuple[str, str]]:
    """Return (voice language, speech text) without changing displayed text."""
    language = (language or detect_language(text)).lower()
    if language not in {"hi", "hindi", "en", "english", "hinglish"}:
        return [(language, text.strip())] if text.strip() else []
    roman_hindi = language == "hinglish" or detect_language(text) == "hinglish" or bool(DEVANAGARI.search(text))
    parts: list[tuple[str, str]] = []
    pending = ""
    for match in TOKEN.finditer(text):
        token = match.group()
        if token.isspace():
            pending += token
            continue
        if DEVANAGARI.search(token):
            voice, spoken = "hi", token
        elif LATIN.search(token):
            transliterated = ROMAN_HINDI_SPEECH.get(token.lower()) if roman_hindi else None
            voice, spoken = ("hi", transliterated) if transliterated else ("en", token)
        else:
            # Keep punctuation and numbers with the preceding phrase.
            if parts:
                old_voice, old_text = parts[-1]
                parts[-1] = old_voice, old_text + pending + token
                pending = ""
                continue
            pending += token
            continue
        if parts and parts[-1][0] == voice:
            old_voice, old_text = parts[-1]
            parts[-1] = old_voice, old_text + pending + spoken
        else:
            parts.append((voice, (pending + spoken).strip()))
        pending = ""
    if parts and pending:
        voice, spoken = parts[-1]
        parts[-1] = voice, spoken + pending
    return [(voice, spoken.strip()) for voice, spoken in parts if spoken.strip()]



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

