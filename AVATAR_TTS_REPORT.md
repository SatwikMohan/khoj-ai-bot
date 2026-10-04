# Avatar and offline speech implementation

## Avatar

The frontend rewrite in commit `4cdf931` removed the original GLTF renderer and replaced it with the CSS `core` and waveform in `frontend/visualization.py`. The model, binary, and textures were still present in `frontend/assets/`; the configured `AVATAR_MODEL_FILE` was no longer read.

The left column now renders that original `scene.gltf` through Three.js and GLTFLoader. These JavaScript modules are vendored under `frontend/assets/` so the browser needs no CDN. The GLTF's binary and texture references are embedded for the existing Streamlit iframe. The model is framed from its actual bounds, scales with the iframe, and retains the existing two-column dark theme. Headless Edge screenshots on Windows are saved locally at `frontend/tests/avatar_smoke.png` and `frontend/tests/avatar_mobile_smoke.png`.

The asset contains a skin and jaw bone but **zero animation clips and zero mouth morph targets**. Idle, listening, retrieving, and generating use restrained movement of the existing root/head rig. Speaking rotates its jaw from measured playback amplitude. When audio analysis is unavailable, the jaw remains neutral; no guessed phoneme animation runs. Interrupted and error states stop speech motion. Playback order and cancellation still use the existing queue.

A missing model or vendored module now shows an error in the avatar panel instead of leaving its loading label indefinitely.

The model is `male04 face rigged` by photon under CC BY 4.0; the UI includes linked attribution and `frontend/assets/license.txt` retains the full credit. Three.js 0.160.1 is MIT licensed; its text is in `frontend/assets/THREE_LICENSE.txt`.

## Speech pipeline

LLM tokens are buffered into meaningful speech chunks by `IncrementalSpeechSegments`. The frontend sends each chunk, detected response language, and request ID to `/tts/speech`. The backend removes Markdown only for speech, expands unambiguous dates/currencies/abbreviations, and routes phrase spans through local Piper voices. Romanized Hindi uses a curated speech-only lexicon; unknown Latin words stay English. Hindi and English WAV phrases are normalized to 22.05 kHz mono PCM and joined with a short crossfade. The browser plays finished chunks in sequence and preserves interruption. Displayed chat text is never normalized or transliterated. The former corrupted question-mark suffix on truncated Hindi speech has been replaced with a readable Hindi ending.

The selected voices are `hi_IN-pratham-medium` and `en_IN-spicor-english`. The old code selected one voice for a whole chunk: any Devanagari forced every embedded English word through the Hindi phonemizer, while Roman Hinglish went through a US English voice. Phrase routing now uses the appropriate phonemizer for each span. The Indian English checkpoint was selected after the user listened to local English and Hinglish comparisons. The two languages still use different speaker recordings.

## Windows verification

`python -m unittest backend.tests.test_language_aware_tts backend.tests.test_versions_and_languages.LanguageAndVoiceTests frontend.tests.test_interface` passed. The complete backend suite passed (106 tests) and the frontend interface suite passed (3 tests) after voice selection. Headless Edge loaded the vendored modules and rendered the original model without cropping.

`python backend/evals/evaluate_language_aware_tts.py --output-dir backend/evals/voice_samples/selected` saved five clips with the selected English voice and machine results in `backend/evals/voice_samples/selected/`. The earlier US English comparison clips remain in `backend/evals/voice_samples/`:

| Sample | Format | Duration | Generation time |
| --- | --- | ---: | ---: |
| English | 22.05 kHz WAV | 5.48 s | 6136 ms |
| Hindi | 22.05 kHz WAV | 4.84 s | 4485 ms |
| Roman Hinglish | 22.05 kHz WAV | 5.20 s | 7969 ms |
| Mixed script | 22.05 kHz WAV | 4.83 s | 394 ms |
| Technical terms | 22.05 kHz WAV | 8.95 s | 740 ms |

These are full-utterance generation times from a cold-to-warm sequential run, **not first-audio latency**. Valid WAV output and script/voice routing were verified. The user chose the Indian English voice after listening; formal pronunciation, transition, and naturalness scores are still unavailable. The code does not claim those subjective properties are solved.

## Model decision

| Candidate | Fit for this task | Offline and deployment considerations |
| --- | --- | --- |
| Selected Piper pair | Hindi uses `hi_IN-pratham-medium`; English uses the NavGurukul `en_IN-spicor-english` checkpoint selected by the user from local comparisons. Phrase routing improves Hindi/English phonemizer selection. Voices differ and the English model JSON still specifies the en-us phonemizer. | CPU works locally. The Hindi voice dataset card identifies CC BY-NC-SA 4.0; the Indian English repository labels its checkpoint AGPL-3.0. Review intended deployment terms. |
| Former US English voice | `en_US-lessac-medium` remains installed for A/B review and is no longer selected by default. | The original English/Hinglish listening clips are saved locally. |
| XTTS-v2 | Supports Hindi and English with cross-language speaker cloning; a plausible single-speaker option. | Requires a reference voice and a larger GPU runtime. Coqui Public Model License restricts commercial use. No local listening or latency test was made. |
| IndicF5 | Strong Hindi candidate, reference-conditioned; model card lists 11 Indic languages. | English is not among those listed, so mixed English needs validation or segmentation. Files are gated; no model was downloaded or benchmarked. |
| IndicF5 Hindi-English code-switch fine-tune | Claims embedded English support. | New third-party model; no local quality, latency, or ARM64 test yet. Its custom-code loading needs separate review. |

Piper remains the offline engine. After listening to the comparison clips, the user selected the Indian English checkpoint. `PIPER_ENGLISH_VOICE` now points to it. The selected model is under the ignored `backend/models/piper` directory locally; `scripts/provision_models.py` downloads and verifies the ONNX and JSON against pinned SHA256 hashes for a new DGX installation. The old US English weights and clips remain locally for A/B review.

## Remaining limits and DGX Spark

- The installed Streamlit version warns that `components.html` is deprecated. The current version passed browser and interface tests; a future Streamlit upgrade will need an iframe hosting path for this self-contained WebGL page.
- The original GLTF has no natural skeletal idle/listening/speaking animation; only small procedural rig motion is possible without an authored animation. Jaw amplitude follows audio energy, not phoneme-level visemes.
- Unknown Romanized Hindi words remain in Latin script and may be pronounced as English. Short isolated English insertions require a voice switch. Crossfading reduces clicks within each synthesized chunk; the existing browser player can still make a small gap between separately streamed chunks.
- The user selected the Indian English voice by listening to English and Hinglish comparisons. The Hindi and English models do not share a speaker identity. Pronunciation of particular technical words and naturalness across voice switches still need evaluation with real queries. Review the Indian English checkpoint's AGPL-3.0 terms for the intended deployment.
- `PIPER_DEVICE` in `backend/config.py` defaults to CPU. CUDA needs an ARM64-compatible ONNX Runtime GPU package on DGX Spark; the Windows packages cannot be copied over. Build and smoke-test the backend container on the Spark, copy or provision the selected ONNX and adjacent JSON into /models/piper, verify the voice licenses, then test actual first-audio latency and memory. An offline `--only-tts --offline` run checks file hashes and synthesizes both voices. The frontend stays independent of inference and its WebGL assets work offline.

Sources: [Indian English Piper candidate](https://huggingface.co/navgurukul-ai-labs/text-to-speech-en-IN-piper), [Piper Hindi model card](https://huggingface.co/rhasspy/piper-voices/blob/main/hi/hi_IN/pratham/medium/MODEL_CARD), [Piper English model card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/lessac/medium/MODEL_CARD), [XTTS-v2 model card](https://huggingface.co/coqui/XTTS-v2), [IndicF5 model card](https://huggingface.co/ai4bharat/IndicF5), [IndicF5 code-switch fine-tune](https://huggingface.co/Tharshan/indicf5_hindi-english_code_switch), [NVIDIA DGX Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/compilation.html).

