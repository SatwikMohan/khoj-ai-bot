"""Deterministic IndicF5 loading without Hugging Face remote model code."""

from dataclasses import dataclass
from pathlib import Path


class IndicF5RuntimeError(RuntimeError):
    pass


@dataclass
class IndicF5Runtime:
    model: object
    vocoder: object
    device: str

    def synthesize(self, text: str, reference_audio: str, reference_text: str):
        try:
            from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

            prepared_audio, prepared_text = preprocess_ref_audio_text(
                reference_audio,
                reference_text,
                device=self.device,
            )
            audio, sample_rate, _spectrogram = infer_process(
                prepared_audio,
                prepared_text,
                text,
                self.model,
                self.vocoder,
                mel_spec_type="vocos",
                device=self.device,
            )
        except Exception as exc:
            raise IndicF5RuntimeError(f"IndicF5 inference failed: {exc}") from exc
        if audio is None:
            raise IndicF5RuntimeError("IndicF5 returned no audio.")
        return audio, int(sample_rate)


def load_indicf5_runtime(
    model_location: str = "ai4bharat/IndicF5",
    *,
    local_files_only: bool = True,
) -> IndicF5Runtime:
    """Build IndicF5 locally and apply its checkpoint without AutoModel remote code."""
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from f5_tts.model import DiT
        from f5_tts.infer.utils_infer import load_model, load_vocoder
    except ImportError as exc:
        raise IndicF5RuntimeError(f"IndicF5 dependency is missing: {exc}") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        vocab_path = hf_hub_download(
            model_location,
            filename="checkpoints/vocab.txt",
            local_files_only=local_files_only,
        )
        checkpoint_path = hf_hub_download(
            model_location,
            filename="model.safetensors",
            local_files_only=local_files_only,
        )
        vocos_config = hf_hub_download(
            "charactr/vocos-mel-24khz",
            filename="config.yaml",
            local_files_only=local_files_only,
        )
        hf_hub_download(
            "charactr/vocos-mel-24khz",
            filename="pytorch_model.bin",
            local_files_only=local_files_only,
        )

        vocoder = load_vocoder(
            vocoder_name="vocos",
            is_local=True,
            local_path=str(Path(vocos_config).parent),
            device=device,
        )
        model = load_model(
            DiT,
            dict(
                dim=1024,
                depth=22,
                heads=16,
                ff_mult=2,
                text_dim=512,
                conv_layers=4,
            ),
            mel_spec_type="vocos",
            vocab_file=vocab_path,
            device=device,
        )

        checkpoint = load_file(checkpoint_path, device=device)
        model_state = {}
        for key, value in checkpoint.items():
            if key.startswith("ema_model._orig_mod."):
                model_state[key.removeprefix("ema_model._orig_mod.")] = value
            elif key.startswith("ema_model."):
                model_state[key.removeprefix("ema_model.")] = value
        if not model_state:
            raise IndicF5RuntimeError("IndicF5 checkpoint contains no ema_model weights.")

        incompatible = model.load_state_dict(model_state, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.endswith("num_batches_tracked")]
        if missing:
            raise IndicF5RuntimeError(
                "IndicF5 checkpoint is incompatible; missing weights: " + ", ".join(missing[:8])
            )
        model.eval()
        return IndicF5Runtime(model=model, vocoder=vocoder, device=device)
    except IndicF5RuntimeError:
        raise
    except Exception as exc:
        mode = "offline cache" if local_files_only else "Hugging Face"
        raise IndicF5RuntimeError(
            f"Could not initialize IndicF5 '{model_location}' from {mode}: {exc}"
        ) from exc
