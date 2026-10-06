# Voice continuity and quality follow-up

Test host: Windows development machine, 6 October 2026. Deployment target: NVIDIA DGX Spark.

## Reproduced cause

`speech_phrases()` previously split one Hinglish sentence at individual Hindi words. For example, "The system works hai and here is the result" became English, Hindi, English synthesis jobs. Piper uses different people for its English and Hindi checkpoints, so this deliberately produced the reported female/male/female alternation and phrase joins.

## Change

The Piper plan now selects one model for each answer chunk: English for English or Roman Hinglish replies, Hindi for Hindi replies. The frontend fixes the reply language from the question for every streamed chunk in that request. The TTS service rejects multiple Piper voice jobs in a chunk and logs the actual selected ONNX model with request ID. Text and queue ownership remain unchanged. No new dependency or network TTS was introduced into the app.

## Verification and limits

All 113 backend tests and 7 frontend tests passed. Five local WAV samples were generated with one model per clip under `backend/evals/voice_samples/single_voice/`; the Hinglish sample uses the English model throughout. These files are local listening artifacts and are ignored by Git. No human listening score, full live browser listening test, or DGX test was completed here.

Piper's current tone presets change speed; they do not add controlled emotion or humanlike expression. Pure Hindi still uses the configured Hindi checkpoint, whose timbre differs from the English checkpoint across separate replies. Roman Hinglish uses one English speaker but Hindi word pronunciation needs listening review. A truly consistent, expressive female Hindi/English speaker requires a different, validated offline model.

An isolated trial of the [IndicVoice-82M model](https://huggingface.co/Bindkushal/IndicVoice-82M) loaded its weights, but the current trial package needed an undeclared spaCy dependency, its bundled eSpeak setup needed a data path fix on Windows, and its English phonemizer then failed because `us_gold.json` was missing from the package. It also attempted a runtime spaCy model download. It was not added to the app or DGX configuration because this would break offline deployment.

## DGX review

Listen to the generated samples, then evaluate a packaged and pinned offline multilingual model with the same female speaker identity in English, Hindi, and Hinglish. Measure first-audio latency and memory alongside the LLM/STT on DGX Spark; keep Piper until the new model passes those tests. The Windows trial does not establish DGX quality or compatibility.
