"""Persistent offline Veena/SNAC worker for one built-in Indian female speaker.

Protocol matches piper_worker.py: JSON lines in, length-prefixed WAV frames out.
The model is loaded only from files provisioned under /models.
"""

import contextlib
import io
import json
import struct
import sys


START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
AUDIO_OFFSET = 128266
CODEBOOK_SIZE = 4096
SAMPLE_RATE = 24000


def split_audio_tokens(tokens: list[int]) -> list[list[int]]:
    """Unpack Veena's seven interleaved SNAC codebooks, rejecting bad frames."""
    start = tokens.index(START_OF_SPEECH) + 1 if START_OF_SPEECH in tokens else 0
    end = tokens.index(END_OF_SPEECH, start) if END_OF_SPEECH in tokens[start:] else len(tokens)
    audio = [token for token in tokens[start:end]
             if AUDIO_OFFSET <= token < AUDIO_OFFSET + 7 * CODEBOOK_SIZE]
    if len(audio) < 7:
        raise ValueError("Veena generated no complete audio frame.")
    audio = audio[: len(audio) // 7 * 7]
    layers = [[], [], []]
    for frame in range(0, len(audio), 7):
        codes = []
        for index, token in enumerate(audio[frame:frame + 7]):
            code = token - AUDIO_OFFSET - index * CODEBOOK_SIZE
            if not 0 <= code < CODEBOOK_SIZE:
                raise ValueError("Veena generated an invalid audio token.")
            codes.append(code)
        layers[0].append(codes[0])
        layers[1].extend((codes[1], codes[4]))
        layers[2].extend((codes[2], codes[3], codes[5], codes[6]))
    return layers


def load_voice(model_dir: str, codec_dir: str):
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from snac import SNAC

        if not torch.cuda.is_available():
            raise RuntimeError("Veena requires a CUDA GPU. Use Piper on the Windows development machine.")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True,
            torch_dtype=torch.bfloat16, device_map="auto",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        codec = SNAC.from_pretrained(codec_dir, local_files_only=True).eval()
    return torch, model, tokenizer, codec


def synthesize(text: str, speaker: str, runtime) -> bytes:
    torch, model, tokenizer, codec = runtime
    if speaker != "kavya":
        raise ValueError("Only the configured Kavya speaker is supported.")
    prompt = tokenizer.encode(f"<spk_{speaker}> {text}", add_special_tokens=False)
    input_ids = torch.tensor([[
        START_OF_HUMAN, *prompt, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH,
    ]], device=model.device)
    max_tokens = min(4096, max(700, len(text) * 14))
    with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
        output = model.generate(
            input_ids, max_new_tokens=max_tokens, do_sample=True,
            temperature=0.5, top_p=0.9, repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=[END_OF_SPEECH, END_OF_AI], use_cache=True,
        )
        token_ids = output[0, input_ids.shape[-1]:].tolist()
        layers = split_audio_tokens(token_ids)
        codec_device = next(codec.parameters()).device
        codes = [torch.tensor(layer, dtype=torch.int32, device=codec_device).unsqueeze(0)
                 for layer in layers]
        samples = codec.decode(codes).squeeze().clamp(-1, 1).cpu().numpy()
    import soundfile as sf
    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def main(model_dir: str, codec_dir: str, speaker: str) -> None:
    runtime = load_voice(model_dir, codec_dir)
    for line in sys.stdin.buffer:
        try:
            request = json.loads(line)
            audio = synthesize(request["text"], speaker, runtime)
            payload = b"\x01" + audio
        except Exception as exc:
            payload = b"\x00" + str(exc).encode("utf-8", errors="replace")
        sys.stdout.buffer.write(struct.pack("<Q", len(payload)))
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--persistent":
        raise SystemExit("Usage: veena_worker.py --persistent MODEL_DIR CODEC_DIR SPEAKER")
    main(sys.argv[2], sys.argv[3], sys.argv[4])
