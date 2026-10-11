"""Deterministic stage clocks and speech buffer lifecycle tests."""
import time
import io
import wave
import unittest
from unittest.mock import patch
from contextlib import closing

from services.response_stream import StreamProgress, bounded_events
from services.speech_chunker import IncrementalSpeechSegments
from services.request_lifecycle import registry
from services.tts_service import TTSEngineError, synthesize_speech
from helpers.request_models import TTSRequest


def limits(**overrides):
    values = {
        "admission": 0.5, "retrieval": 0.5, "model_init": 0.5,
        "first_token": 0.5, "finalizing": 0.5,
        "inter_token": 0.5, "total_generation": 0.5,
    }
    values.update(overrides)
    return values


class StageClockTests(unittest.TestCase):
    def collect(self, factory, progress, **overrides):
        return list(bounded_events(
            factory, 0.03, 0.005, stage_timeouts=limits(**overrides), progress=progress
        ))

    def test_retrieval_timeout_has_accurate_category(self):
        progress = StreamProgress()

        def source():
            progress.mark("retrieval")
            time.sleep(0.08)
            yield {"type": "done", "answer": "late"}

        events = self.collect(source, progress, retrieval=0.035)
        self.assertEqual(events[-1]["category"], "retrieval_timeout")

    def test_model_initialization_timeout(self):
        progress = StreamProgress()

        def source():
            progress.mark("model_init")
            time.sleep(0.08)
            yield {"type": "done", "answer": "late"}

        self.assertEqual(self.collect(source, progress, model_init=0.035)[-1]["category"],
                         "model_init_timeout")

    def test_first_token_timeout(self):
        progress = StreamProgress()

        def source():
            progress.mark("first_token")
            time.sleep(0.08)
            progress.advance()
            yield {"type": "token", "text": "late"}
            yield {"type": "done", "answer": "late"}

        self.assertEqual(self.collect(source, progress, first_token=0.035)[-1]["category"],
                         "first_token_timeout")

    def test_steady_generation_outlives_inter_token_limit(self):
        progress = StreamProgress()

        def source():
            progress.mark("first_token")
            for index in range(8):
                time.sleep(0.017)
                progress.advance()
                yield {"type": "token", "text": str(index)}
            yield {"type": "done", "answer": "01234567"}

        events = self.collect(source, progress, inter_token=0.05, total_generation=0.4)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "token"), "01234567")

    def test_generation_stall_preserves_partial_status(self):
        progress = StreamProgress()

        def source():
            progress.mark("first_token")
            progress.advance()
            yield {"type": "token", "text": "partial"}
            time.sleep(0.09)
            yield {"type": "done", "answer": "late"}

        events = self.collect(source, progress, inter_token=0.035)
        self.assertEqual(events[-1]["category"], "generation_stalled")
        self.assertTrue(events[-1]["partial"])

    def test_total_generation_deadline_even_with_progress(self):
        progress = StreamProgress()

        def source():
            progress.mark("first_token")
            for index in range(12):
                time.sleep(0.016)
                progress.advance()
                yield {"type": "token", "text": str(index)}
            yield {"type": "done", "answer": "late"}

        events = self.collect(source, progress, inter_token=0.06, total_generation=0.07)
        self.assertEqual(events[-1]["category"], "generation_deadline")
        self.assertTrue(events[-1]["partial"])

    def test_dense_progressing_stream_still_honors_total_deadline(self):
        progress = StreamProgress()

        def source():
            progress.mark("first_token")
            for index in range(30):
                time.sleep(0.003)
                progress.advance()
                yield {"type": "token", "text": str(index)}
            progress.mark("finalizing")
            yield {"type": "done", "answer": "late"}

        events = list(bounded_events(
            source, 1, 0.25, stage_timeouts=limits(
                first_token=0.2, inter_token=0.2, total_generation=0.025
            ), progress=progress
        ))
        self.assertEqual(events[-1]["category"], "generation_deadline")
        self.assertTrue(events[-1]["partial"])
        self.assertNotIn("done", [event["type"] for event in events])

    def test_late_done_during_queue_wait_is_not_accepted(self):
        progress = StreamProgress()

        def source():
            progress.mark("first_token")
            time.sleep(0.05)
            progress.advance()
            progress.mark("finalizing")
            yield {"type": "done", "answer": "late"}

        events = list(bounded_events(
            source, 1, 0.25, stage_timeouts=limits(
                first_token=0.2, inter_token=0.2, total_generation=0.025
            ), progress=progress
        ))
        self.assertEqual(events[-1]["category"], "generation_deadline")
    def test_incomplete_stream_is_never_marked_answer_ready(self):
        request = registry.begin("incomplete-stage-test", "incomplete")

        def source():
            yield {"type": "token", "text": "partial"}

        events = list(bounded_events(source, 1, 0.005, request=request))
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(request.status, "failed")
        registry.cancel(request.session_id)

    def test_slow_consumer_does_not_create_model_timeout(self):
        progress = StreamProgress()

        def source():
            progress.mark("first_token")
            progress.advance()
            yield {"type": "token", "text": "ready"}
            progress.mark("finalizing")
            yield {"type": "done", "answer": "ready"}

        with closing(bounded_events(source, 0.01, 0.005, stage_timeouts=limits(
                first_token=0.03, inter_token=0.03, total_generation=0.03), progress=progress)) as stream:
            first = next(stream)
            self.assertEqual(first["type"], "token")
            time.sleep(0.08)
            self.assertEqual(next(stream)["type"], "done")


class SpeechCancellationTests(unittest.TestCase):
    def test_audio_timeout_invalidates_speech_but_next_request_recovers(self):
        old = registry.begin("speech-timeout-test", "old")
        old.speech_cancelled.set()
        with self.assertRaisesRegex(TTSEngineError, "replaced"):
            synthesize_speech(TTSRequest(text="Safe text.", session_id=old.session_id,
                                         request_id=old.request_id, language="en",
                                         response_format="wav"))
        new = registry.begin("speech-timeout-test", "new")
        self.assertFalse(new.speech_cancelled.is_set())
        registry.cancel(new.session_id)


class StreamingBenchmarkTests(unittest.TestCase):
    def test_benchmark_records_multiple_ordered_audio_chunks(self):
        from evals import benchmark_voice

        def stream(question, **kwargs):
            registry.begin(kwargs["session_id"], kwargs["request_id"])
            first = "The mine ventilation system moves clean air to every active working section."
            second = "Workers inspect the equipment and report unsafe conditions before a shift begins."
            yield {"type": "token", "text": first}
            yield {"type": "token", "text": " " + second}
            yield {"type": "done", "answer": first + " " + second,
                   "timings_ms": {"generation": 10}, "model_metrics": {},
                   "context_chars": 100}

        def speech(request):
            output = io.BytesIO()
            with wave.open(output, "wb") as clip:
                clip.setnchannels(1)
                clip.setsampwidth(2)
                clip.setframerate(8000)
                clip.writeframes(b"\0\0" * 8000)
            return output.getvalue(), "audio/wav"

        with patch.object(benchmark_voice, "stream_answer_events", side_effect=stream), patch.object(
            benchmark_voice, "synthesize_speech", side_effect=speech
        ):
            report = benchmark_voice.run("Explain mine ventilation", warmup=False)
        self.assertTrue(report["completed"])
        self.assertEqual(report["audio_chunks_generated"], 2)
        self.assertEqual([clip["sequence"] for clip in report["clips"]], [0, 1])
        self.assertEqual(report["context_chars"], 100)


class SpeechBufferTests(unittest.TestCase):
    def test_max_buffer_delay_releases_phrase_at_word_boundary(self):
        with patch("config.TTS_STREAM_MIN_CHARS", 25), patch("config.TTS_STREAM_MAX_CHARS", 120), patch(
                "config.TTS_STREAM_MAX_BUFFER_SECONDS", 0.1):
            buffer = IncrementalSpeechSegments()
            self.assertEqual(buffer.push(
                "The mine ventilation system moves clean air through each working section"), [])
            self.assertEqual(buffer.flush_due(now=buffer.buffered_at + 0.05), [])
            early = buffer.flush_due(now=buffer.buffered_at + 0.11)
            self.assertEqual(early, [
                "The mine ventilation system moves clean air through each working"
            ])
            self.assertEqual(buffer.finish(), ["section"])


if __name__ == "__main__":
    unittest.main()
