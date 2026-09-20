"""Apply the IndicF5 meta-tensor fix to the installed F5-TTS package."""

from importlib.util import find_spec
from pathlib import Path


spec = find_spec("f5_tts.infer.utils_infer")
if spec is None or spec.origin is None:
    raise RuntimeError("Could not locate f5_tts.infer.utils_infer")

path = Path(spec.origin)
source = path.read_text(encoding="utf-8")
changed = False

vocoder_marker = "if any(parameter.is_meta for parameter in vocoder.parameters()):"
vocoder_needle = """        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location=\"cpu\", weights_only=True)
"""
vocoder_replacement = """        vocoder = Vocos.from_hparams(config_path)
        # Transformers may construct custom models under a meta-device context.
        # Materialize Vocos before loading its real checkpoint tensors.
        if any(parameter.is_meta for parameter in vocoder.parameters()):
            vocoder = vocoder.to_empty(device=\"cpu\")
        state_dict = torch.load(model_path, map_location=\"cpu\", weights_only=True)
"""
if vocoder_marker not in source:
    if vocoder_needle not in source:
        raise RuntimeError(
            f"F5-TTS Vocos layout changed; refusing to patch {path}"
        )
    source = source.replace(vocoder_needle, vocoder_replacement, 1)
    changed = True

model_marker = "if any(parameter.is_meta for parameter in model.parameters()):"
model_needle = """        vocab_char_map=vocab_char_map,
    ).to(device)
"""
model_replacement = """        vocab_char_map=vocab_char_map,
    )
    # The outer Transformers loader will apply the real IndicF5 checkpoint.
    # Allocate storage first when construction occurred on the meta device.
    if any(parameter.is_meta for parameter in model.parameters()):
        model = model.to_empty(device=device)
    else:
        model = model.to(device)
"""
if model_marker not in source:
    if model_needle not in source:
        raise RuntimeError(
            f"F5-TTS CFM layout changed; refusing to patch {path}"
        )
    source = source.replace(model_needle, model_replacement, 1)
    changed = True

if changed:
    path.write_text(source, encoding="utf-8")
    print(f"Applied IndicF5 meta-tensor patches to {path}")
else:
    print(f"IndicF5 meta-tensor patches already present in {path}")
