# Offline speech speed update ? 2026-10-12

Changed the DGX default from Veena to existing Piper CPU workers. No new runtime
dependency, cloud service, model download at startup, or index change was added.
Veena remains configurable. The change favors latency over single-speaker
bilingual naturalness; Roman-script Hinglish is unsupported.

Measured locally on Windows AMD64 with installed models and OFFLINE_MODE=True.
The final repeat run completed 20 syntheses (10 English, 10 Hindi), including
one cold call per voice. English warm median: 507.0 ms. Hindi warm median: 361.9 ms.
These are not DGX results or a network-denial certification.

Earlier trials had intermittent Hindi subprocess/empty-audio failures. Isolated
Hindi synthesis passed. Added watchdog joining before worker reuse to close a
cancellation race; this does not prove every earlier failure had that cause.
The final 20-call run passed. Run the same benchmark on the DGX and listen to
the generated WAVs before treating the voice quality and performance as validated.

Keep TTS_PRELOAD_VOICES enabled to move cold loading out of ordinary turns.
Use backend/evals/benchmark_tts_only.py --runs 10 for reproducible timings.
Results and audio are under backend/evals/voice_samples/tts_speed (ignored artifacts).
