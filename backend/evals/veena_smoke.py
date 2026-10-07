"""Generate four listening clips with the configured DGX Veena speaker."""

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

import config
from helpers.request_models import TTSRequest
from services.tts_service import synthesize_speech


SAMPLES = (
    ("english", "en", "Hello, I am Khoj. I can explain the report in clear, natural English."),
    ("hindi", "hi", "\u0928\u092e\u0938\u094d\u0924\u0947, \u092e\u0948\u0902 \u0916\u094b\u091c \u0939\u0942\u0901\u0964 \u092e\u0948\u0902 \u0906\u092a\u0915\u094b \u092f\u0939 \u0930\u093f\u092a\u094b\u0930\u094d\u091f \u0906\u0938\u093e\u0928 \u092d\u093e\u0937\u093e \u092e\u0947\u0902 \u0938\u092e\u091d\u093e\u090a\u0901\u0917\u0940\u0964"),
    ("hinglish_roman", "hinglish", "Yaar, mujhe yeh report samajhne mein madad chahiye. Let's look at the main points."),
    ("hinglish_mixed_script", "hinglish", "\u092e\u0941\u091d\u0947 \u092f\u0939 report \u0938\u092e\u091d\u0928\u0947 \u092e\u0947\u0902 \u092e\u0926\u0926 \u091a\u093e\u0939\u093f\u090f\u0964 Let's look at the main points."),
)


def main() -> None:
    if config.TTS_ENGINE != "veena":
        raise SystemExit("This listening check requires TTS_ENGINE=veena on the DGX.")
    output_dir = BACKEND_ROOT / "evals" / "voice_samples" / "veena_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, language, text in SAMPLES:
        audio, media_type = synthesize_speech(TTSRequest(
            text=text, language=language, response_format="wav",
        ))
        if media_type != "audio/wav":
            raise RuntimeError(f"Unexpected format: {media_type}")
        path = output_dir / f"{name}.wav"
        path.write_bytes(audio)
        manifest.append({"file": path.name, "language": language, "text": text,
                         "bytes": len(audio), "speaker": config.VEENA_SPEAKER})
        print(path, flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
