"""One bounded CPU speech job. Invoked by tts_service in a subprocess."""

import contextlib
import io
import json
import sys
import wave


def main():
    request = json.load(sys.stdin)
    # Keep stdout exclusively for WAV bytes, including during library imports.
    with contextlib.redirect_stdout(sys.stderr):
        from piper import PiperVoice, SynthesisConfig

        voice = PiperVoice.load(request["model"], use_cuda=False)
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            voice.synthesize_wav(
                request["text"], wav_file,
                syn_config=SynthesisConfig(length_scale=request["length_scale"]),
            )
    sys.stdout.buffer.write(output.getvalue())


if __name__ == "__main__":
    main()
