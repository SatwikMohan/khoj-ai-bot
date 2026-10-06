# Production fixes: avatar, answer quality, and audio ownership

Test host: Windows development machine, 6 October 2026. Target: NVIDIA DGX Spark. The DGX is not available from this workspace, so target hardware results are pending.

## 1. Avatar root cause and asset path

The current tracked scene.gltf loads, but its skinned meshes were not updated before their bounds were measured. A headless Edge reproduction loaded the model and issued 4 draw calls while producing zero visible WebGL pixels. Updating world and skin bind matrices before framing rendered the actual model. The renderer now checks for visible pixels and reports WebGL load, context, and blank render failures in the panel and browser console. The old UI credit named a different model; it now credits the current GLTF asset, Invincible - Debbie Grayson by ASideOfChidori.

No repository asset or path omission was found: frontend/assets/scene.gltf, scene.bin, the referenced textures, and vendored Three.js modules are tracked, are real files rather than Git LFS pointers, and are copied by the root Dockerfile. The configured path is resolved from frontend/avatar_renderer.py rather than the working directory. The new validator checks exact filename case, resource presence, empty files and LFS pointers. Both application Dockerfiles now fail their build if these assets are missing. The observed remote deployment failure cannot be conclusively attributed without DGX browser and container logs.

## 2. Answer quality root cause and diagnostics

The direct backend path uses ChatPromptTemplate system and human messages with ChatOllama. Retrieved context reached the model in a direct test, and streaming and nonstreaming share the same answer path. The local test model answered the bucket wheel question incorrectly as a Continuous Miner when five retrieved passages included unrelated equipment. After routine questions were limited to two answer passages, it answered Bucket Wheel Excavator (BWE) correctly. Summary, comparison, and version questions still receive all selected passages. Context passages are separated and internal labels are filtered from user output. QA_DIAGNOSTICS_ENABLED in backend/config.py logs selected documents, scores/metadata, final context, formatted messages, model settings, and answer to server logs when explicitly enabled.

The Windows host has llama3.2:latest, nomic-embed-text, and an active version 6 nomic index; it does not have the configured DGX llama3.1:8b-instruct-q4_K_M or bge-m3 models. The staged local version 8 index does not replace the active version 6 index. The direct six-case Windows benchmark passed all four English cases and failed Hindi safety retrieval and a Hinglish BWE answer. Those failures remain unresolved on the Windows profile. A local-LLM translation attempt was rejected after it broadened or failed to translate queries. The DGX multilingual model and its new index require a separate evaluation.

## 3. Duplicate voice root cause and architecture

The prior frontend had independent avatar and plain-audio player implementations plus a separate full-answer synthesis path. No browser-session audio owner prevented an old iframe from playing while a new query started. The observed two-voice event was not recorded on this test host, so this is a code-path root cause rather than an acoustic trace of that specific incident.

The frontend now uses one shared AudioManager per conversation session. It alone starts playback, owns one HTML audio element per active request, orders chunks by sequence, rejects mismatched request IDs and queue IDs, drops duplicates, and clears old clips on replacement or interruption. The separate full-answer playback and historical Streamlit audio player were removed. TTS generation may run on two workers for latency, but generated chunks enter one serialized browser queue. Server logs identify TTS request, sequence, language, voice and completion; browser logs identify queued, started, ended, stopped and completed playback. An in-flight Piper synthesis call may finish after cancellation, but its output is discarded and cannot play.

## 4. Changes, dependencies, and tests

Files changed: AVATAR_TTS_REPORT.md; Dockerfile; frontend/Dockerfile; backend/config.py; backend/services/qa_service.py; backend/services/response_stream.py; backend/tests/test_response_delivery.py; backend/evals/README.md; backend/evals/answer_quality_cases.jsonl; backend/evals/evaluate_answer_quality.py; backend/evals/local_answer_quality_2026-10-06.json; frontend/app.py; frontend/audio_manager.py; frontend/assets/audio_manager.js; frontend/avatar_renderer.py; frontend/visualization.py; frontend/requirements.txt; frontend/tests/test_interface.py.

Dependency change: Streamlit is pinned to the locally tested 1.62.0 release. No cloud service or new package was introduced. Streamlit warns that components.html is deprecated; a later migration to st.iframe still needs integration testing.

Tests executed on Windows: 113 backend unit tests passed; 7 frontend interface tests passed; Python compilation and git diff checks passed; local avatar asset validator found scene.gltf and three referenced resources; headless Edge displayed the current 3D avatar; a browser smoke test delivered audio chunks 1 then 0 and confirmed playback order 0 then 1 while a stale request and duplicate chunk were ignored; a separate browser owner test confirmed replacement invalidated the old owner. The six-case direct answer benchmark passed 4/6 under local model overrides, as described above. Browser playback was simulated in the queue smoke test; no human listening test was performed.

## 5. DGX Spark deployment checks still required

Build the app image on the DGX to execute the new asset check. Provision the exact config.py Ollama, Whisper and Piper models into local storage before offline startup; build and promote a bge-m3 index because the Windows nomic index is incompatible. Run the direct answer benchmark without local overrides, inspect model name, quantization, context and GPU residency in Ollama, and test English, Hindi and Hinglish with real questions. Open the app through the production Caddy URL in a WebGL-capable browser, check the avatar and console, then interrupt speech across consecutive text and microphone queries while listening. Measure GPU/CPU memory and first-audio latency on the actual ARM64/CUDA stack. None of these DGX checks have been claimed as passed here.
