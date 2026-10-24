"""Conservative local word conversions for low-latency Hinglish speech.

Only complete, known phrases use this fast path. Unknown lowercase words
stay on the existing model-assisted conversion path.
"""

HINDI_WORDS = {
    "aap": "आप", "aapka": "आपका", "aapke": "आपके", "aapki": "आपकी",
    "aapko": "आपको", "aaj": "आज", "agar": "अगर", "aur": "और",
    "baare": "बारे", "bare": "बारे", "batao": "बताओ", "bataye": "बताएं",
    "bataiye": "बताइए", "bataein": "बताएं", "bataen": "बताएं",
    "banata": "बनाता", "banati": "बनाती",
    "batata": "बताता", "batati": "बताती", "bhi": "भी", "bilkul": "बिल्कुल",
    "chahiye": "चाहिए", "de": "दे", "den": "दें", "dena": "देना",
    "dekhta": "देखता", "dekhti": "देखती", "dekhna": "देखना",
    "desh": "देश", "dhanyavad": "धन्यवाद", "dhyan": "ध्यान",
    "diya": "दिया", "dikhata": "दिखाता",
    "dikhati": "दिखाती", "dilao": "दिलाओ", "do": "दो", "ek": "एक",
    "gaya": "गया", "gayi": "गई", "gaye": "गए", "galat": "गलत",
    "hai": "है", "hain": "हैं", "hoga": "होगा", "hogi": "होगी",
    "honge": "होंगे", "hota": "होता", "hoti": "होती", "hote": "होते",
    "hum": "हम", "iska": "इसका", "iski": "इसकी", "iske": "इसके",
    "isko": "इसको", "isme": "इसमें", "isliye": "इसलिए",
    "jaana": "जाना", "jana": "जाना", "jaankari": "जानकारी",
    "jagah": "जगह",
    "jankari": "जानकारी", "jawab": "जवाब", "jo": "जो",
    "ka": "का", "kare": "करें", "karen": "करें", "karein": "करें",
    "karna": "करना", "karo": "करो", "karta": "करता", "karti": "करती",
    "kal": "कल", "kaise": "कैसे", "kaisa": "कैसा", "kaisi": "कैसी",
    "ke": "के", "ki": "की", "khoj": "खोज", "kisi": "किसी",
    "ko": "को", "kripya": "कृपया", "kya": "क्या",
    "kyunki": "क्योंकि", "lekin": "लेकिन", "liye": "लिए",
    "main": "मैं", "matlab": "मतलब", "mein": "में", "me": "में",
    "mera": "मेरा", "meri": "मेरी", "mujhe": "मुझे",
    "nahi": "नहीं", "naam": "नाम", "namaste": "नमस्ते",
    "par": "पर", "pura": "पूरा", "puri": "पूरी", "poora": "पूरा",
    "prabhag": "प्रभाग", "rakha": "रखा", "rakhi": "रखी",
    "rakhe": "रखे", "rakhta": "रखता",
    "sahi": "सही", "samajh": "समझ", "samjhao": "समझाओ",
    "samjhaiye": "समझाइए", "samjhaye": "समझाएं", "sakte": "सकते",
    "sawaal": "सवाल", "se": "से", "shukriya": "शुक्रिया",
    "tak": "तक", "unhe": "उन्हें", "unko": "उनको",
    "vikas": "विकास", "voh": "वह", "woh": "वह", "yaad": "याद",
    "yaar": "यार", "ye": "ये", "yeh": "यह",
    "zaroori": "ज़रूरी", "zaruri": "ज़रूरी",
}

AMBIGUOUS_HINDI_WORDS = frozenset({"bare", "de", "do", "main", "me", "par"})

# Common English glue and domain terms retain their Latin spelling for the
# bilingual voice. This is intentionally narrow: unknown words use the model.
ENGLISH_WORDS = frozenset({
    "a", "an", "and", "are", "at", "about", "as", "be", "by", "can",
    "for", "from", "has", "have", "in", "into", "is", "it", "its",
    "of", "on", "or", "our", "the", "their", "this", "that", "to",
    "was", "were", "what", "which", "who", "with", "you", "your",
    "act", "acts", "answer", "answers", "critical", "department",
    "details", "dgms", "document", "documents", "equipment",
    "form", "full", "general", "india", "information", "inspection",
    "inspections", "limit", "limits", "material", "minerals",
    "mine", "mines", "mining", "model", "rules", "rule", "safety",
    "system", "systems", "regulation", "regulations", "structure",
    "technical", "terms", "version", "versions",
})