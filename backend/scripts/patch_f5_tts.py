"""Apply the IndicF5 meta-tensor fix to the installed F5-TTS package."""

from importlib.util import find_spec
from pathlib import Path


spec = find_spec("f5_tts.infer.utils_infer")
if spec is None or spec.origin is None:
    raise RuntimeError("Could not locate f5_tts.infer.utils_infer")

path = Path(spec.origin)
source = path.read_text(encoding="utf-8")
marker = "if any(parameter.is_meta for parameter in vocoder.parameters()):"
if marker in source:
    print(f"IndicF5 meta-tensor patch already present in {path}")
    raise SystemExit(0)

needle = """        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location=\"cpu\", weights_only=True)
"""
replacement = """        vocoder = Vocos.from_hparams(config_path)
        # Transformers may construct custom models under a meta-device context.
        # Materialize Vocos before loading its real checkpoint tensors.
        if any(parameter.is_meta for parameter in vocoder.parameters()):
            vocoder = vocoder.to_empty(device=\"cpu\")
        state_dict = torch.load(model_path, map_location=\"cpu\", weights_only=True)
"""
if needle not in source:
    raise RuntimeError(
        f"F5-TTS layout changed; refusing to apply an unsafe patch to {path}"
    )

path.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
print(f"Applied IndicF5 meta-tensor patch to {path}")
