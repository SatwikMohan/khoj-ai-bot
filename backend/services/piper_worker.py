"""Offline Piper worker, with an optional framed persistent protocol."""

import contextlib
import io
import json
import sys
import wave
import struct


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


def persistent_main(model_path: str):
    with contextlib.redirect_stdout(sys.stderr):
        from piper import PiperVoice, SynthesisConfig
        voice = PiperVoice.load(model_path, use_cuda=False)
    for line in sys.stdin.buffer:
        try:
            request = json.loads(line)
            output = io.BytesIO()
            with wave.open(output, "wb") as wav_file:
                with contextlib.redirect_stdout(sys.stderr):
                    voice.synthesize_wav(
                        request["text"], wav_file,
                        syn_config=SynthesisConfig(length_scale=request["length_scale"]),
                    )
            payload = b"\x01" + output.getvalue()
        except Exception as exc:
            payload = b"\x00" + str(exc).encode("utf-8", errors="replace")
        sys.stdout.buffer.write(struct.pack("<Q", len(payload)))
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--persistent":
        persistent_main(sys.argv[2])
    else:
        main()
