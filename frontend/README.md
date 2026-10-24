# AI interface

Prepare dependencies and the models selected in `backend/config.py` before
offline deployment. See [deployment preparation](../DEPLOYMENT.md).
Runtime startup does not download models.

Preserve an existing compatible index. When changing embedding models, use
`python train_engine.py` to build a candidate, evaluate it, then explicitly
promote it with `python train_engine.py --promote-candidate`. Do not force-rebuild
or delete the active index as a routine setup step.

Run Streamlit directly, with the local Ollama service and provisioned models available:

```powershell
cd frontend
streamlit run app.py
```

The interface uses two columns: the offline GLTF avatar and a scrollable
conversation. Text and microphone input run through the shared backend Python
services in the Streamlit process. The microphone sends audio through the
Streamlit component bridge; it does not call FastAPI. Start conversation once
to grant microphone permission. The browser meter confirms speech and ends
the utterance after the configured silence interval; offline STT then submits
the finalized question automatically. The microphone is released while the
answer plays and starts listening again after the ordered audio queue finishes.
End conversation stops capture and playback. Typed chat remains available.
This is a half-duplex flow: speech during answer playback is not transcribed.
The retained FastAPI application in backend/main.py is optional for external
clients.

All application settings live in `backend/config.py`. For example:

```python
OLLAMA_KEEP_ALIVE = 1800
QA_WARMUP_ON_STARTUP = True
RETRIEVAL_FETCH_K = 40
RETRIEVAL_MMR_ENABLED = True
MMR_LAMBDA_MULT = 0.35
CONTEXT_MAX_CHARS = 16000
CHAT_HISTORY_MAX_TURNS = 4
CHAT_HISTORY_MAX_CHARS = 1200
CHAT_HISTORY_MESSAGE_CHARS = 320
```

The broader retrieval defaults help the bot compare repeated topics across different
facts, years, rules, figures, and sources instead of answering from the first matching chunk only.
Recent chat history is included only to resolve follow-up questions and is trimmed before prompting.

Text-to-speech uses one offline Goonj speaker pack for English, Hindi, and converted Hinglish. The displayed answer is unchanged. The model and speaker files are provisioned with `python scripts/provision_models.py --only-tts` from `backend` on a connected machine, then verified with `--offline`. See the deployment guide for transfer to an air-gapped DGX. Piper and Veena are optional voice engines.
See [deployment and model configuration](../DEPLOYMENT.md) for local/DGX settings,
offline setup and switching models. The pull commands above are examples; use
the names selected in `backend/config.py`. Restart the process after edits; no
`.env` is loaded. Both Dockerfiles include the complete backend service layer and start Streamlit directly.

The Streamlit process now starts each configured model warm-up once in a
background thread: TTS when enabled and preloading is enabled, STT when
`STT_WARMUP_ON_STARTUP` is true, and QA when `QA_WARMUP_ON_STARTUP` is true.
A warm-up failure does not prevent the remaining components from warming.
QA warm-up remains disabled by default; change it only after verifying local
model/index compatibility and the combined memory budget.

Recordings retain their complete media container. During silence, capture starts
a fresh container at `RECORDING_NO_SPEECH_TIMEOUT_SECONDS` without submitting a
question. Total capture is capped by the smaller recording/Whisper limit, with
one second reserved for recorder finalization.

Normal responses show only the conversational answer, without document names,
page numbers, or source panels. Source metadata remains internal; explicitly
asking for sources or page locations enables supported locations in the answer.
Conversation history still lasts only for the current Streamlit session.

Slow speech synthesis does not block text streaming. Active synthesis jobs and
deferred speech segments have separate bounded queues. The UI marks text as
"Answer complete" as soon as generation ends, shows "preparing speech" while
synthesis remains pending, and uses actual playback events for speaking state.
A full deferred queue reports omitted speech rather than blocking text indefinitely.
TTS timeouts discard deferred work and keep the text answer visible.

Veena's persistent SNAC decoder now runs on CUDA, retaining its original precision.
Worker logs include `veena_runtime_ready` (load time and device placement) and
`veena_synthesis_stages` (token generation and decoding times). These diagnostics
go to stderr so they cannot corrupt the binary audio protocol on stdout.
The DGX sample supplied on 2026-10-11 took 3.22 seconds for QA and 24.88 seconds
for synthesis of 6.45 seconds of speech before this change. Re-measure on the DGX;
local mocked tests do not establish a GPU speedup.

The default offline speech engine is Goonj-1-82M, using one Indian `en_priya` speaker pack for English, Hindi, and Hinglish. Piper and Veena remain optional. The mic keeps one media stream during a voice conversation, resets silent recording buffers without asking for permission again, and allows 1.6 seconds of pause before submitting an utterance. The controls above the message input have been removed; speech follows `TTS_ENABLED` in `backend/config.py`.

## Hinglish speech conversion

TTS_HINGLISH_TRANSLITERATION=True supersedes the earlier English-answer fallback.
The written answer remains Roman Hinglish. Before Goonj synthesis, the existing
local Ollama chat model converts Hindi word spellings to Devanagari; the resulting
speech-only text uses Goonj's Hindi phonemizer with the same speaker pack. English words and technical terms are
instructed to remain unchanged. Numbers/punctuation are reconstructed from the
original text; common units, acronyms and numeric identifiers are protected.

This is script conversion, not answer regeneration or a Hindi translation.
The model can still misspell or misidentify words; listen to representative mining
terminology on the DGX. Structural validation is not a semantic correctness proof.
No cloud service, new model download or external transliteration dependency is used.
English and Hindi answers bypass this conversion.

Each speech segment adds one local model request, bounded by a token cap and the
12-second TTS_TRANSLITERATION_TIMEOUT_SECONDS socket timeout. No automatic retry
is added. This can increase first-audio latency and contend with answer generation.
The hinglish_speech_conversion log reports its elapsed time. On failure, the text
answer stays visible and speech reports an error. Disabling the setting restores
the automatic English-answer fallback when Piper and TTS are enabled.

Copy the updated code and rebuild/recreate app; no voice download is needed if the
Hindi Piper model was already provisioned. The configured Llama 3.1 model must be
installed locally in Ollama. Local tests use mocked conversion responses; target
conversion accuracy and latency remain unverified.
