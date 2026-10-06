import threading
import io
import wave
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableGenerator

from services.model_config import thinking_setting
from services.qa_service import _select_answer_docs, _feedback_retrieval_query, _format_chat_history, _format_context, _is_general_query, _wants_source_locations, answer_question, stream_answer_events
from services.response_stream import AnswerTextFilter, CitationTextFilter, bounded_events
from services.request_lifecycle import registry
from services.speech_chunker import IncrementalSpeechSegments
from services.audio_chunks import join_audio_chunks


class AnswerFilterTests(unittest.TestCase):
    def test_every_possible_split_of_reasoning_tags(self):
        raw = '<think>private plan</think><answer>According to the document, the limit is 100 kg.</answer>'
        expected = 'According to the document, the limit is 100 kg.'
        for split in range(len(raw) + 1):
            parser = AnswerTextFilter()
            result = parser.feed(raw[:split]) + parser.feed(raw[split:]) + parser.feed('', final=True)
            self.assertEqual(result, expected, split)

    def test_unclosed_reasoning_is_never_spoken(self):
        parser = AnswerTextFilter()
        self.assertEqual(parser.feed('<think>private') + parser.feed('', final=True), '')

    def test_normal_text_and_less_than_sign_are_preserved(self):
        parser = AnswerTextFilter()
        text = 'They need clearance. The limit is < 10. Based on the documents, use PPE.'
        self.assertEqual(parser.feed(text) + parser.feed('', final=True), text)

    def test_citation_filter_handles_split_stream_and_preserves_subject_terms(self):
        raw = "From Reference 1, page 4, the limit is 100 kg [1]."
        for split in range(len(raw) + 1):
            cleaner = CitationTextFilter()
            result = cleaner.feed(raw[:split]) + cleaner.feed(raw[split:]) + cleaner.feed("", final=True)
            self.assertEqual(result, "The limit is 100 kg.", split)
        cleaner = CitationTextFilter()
        self.assertEqual(cleaner.feed("Reference voltage and page size are fields.", final=True),
                         "Reference voltage and page size are fields.")
        self.assertEqual(CitationTextFilter(enabled=False).feed(raw, final=True), raw)
        self.assertEqual(CitationTextFilter().feed("From Evidence 1, the limit is 100 kg [Evidence 1].", final=True), "The limit is 100 kg.")

    def test_first_speech_segment_is_ready_before_generation_ends(self):
        segmenter = IncrementalSpeechSegments()
        self.assertEqual(segmenter.push('The mine safety procedure requires checking equipment and ventilation before the shift begins.'),
                         ['The mine safety procedure requires checking equipment and ventilation before the shift begins.'])
        self.assertEqual(segmenter.push(' The next sentence is still being generated'), [])
        self.assertEqual(segmenter.finish(), ['The next sentence is still being generated'])

    def test_speech_segmenter_preserves_spaces_across_tokens(self):
        segmenter = IncrementalSpeechSegments()
        segmenter.push('The mine safety procedure requires checking ')
        parts = segmenter.push('equipment and ventilation before the shift begins.')
        self.assertEqual(parts, ['The mine safety procedure requires checking equipment and ventilation before the shift begins.'])

    def test_replay_wav_keeps_audio_chunks_in_order(self):
        def clip(frames):
            output = io.BytesIO()
            with wave.open(output, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(1)
                wav_file.setframerate(8000)
                wav_file.writeframes(frames)
            return output.getvalue()
        joined = join_audio_chunks([clip(b'abc'), clip(b'def')], 'audio/wav')
        with wave.open(io.BytesIO(joined), 'rb') as wav_file:
            self.assertEqual(wav_file.readframes(6), b'abcdef')


class ResponseDeliveryTests(unittest.TestCase):
    def test_greeting_does_not_swallow_a_document_question(self):
        self.assertFalse(_is_general_query('Hi, storage limit?'))
        self.assertFalse(_is_general_query('hello storage limit'))

    def setUp(self):
        patches = [
            patch('services.qa_service._load_environment'),
            patch('services.qa_service._vector_store'),
            patch('services.qa_service._retrieve_context', return_value=[Document(
                page_content='The storage limit is 100 kg.', metadata={'source': 'rules.pdf'},
            )]),
            patch.multiple('config', RESPONSE_LANGUAGE='english', QA_RESPONSE_TIMEOUT_SECONDS=3),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    @staticmethod
    def model(chunks):
        def generate(inputs):
            list(inputs)
            yield from chunks
        return RunnableGenerator(generate)

    def test_reasoning_channel_and_split_tags_do_not_hide_final_answer(self):
        model = self.model([
            AIMessageChunk(content='', additional_kwargs={'reasoning_content': 'private'}),
            AIMessageChunk(content='<thi'), AIMessageChunk(content='nk>private</think>'),
            AIMessageChunk(content='According to the document, '),
            AIMessageChunk(content='the storage limit is 100 kg.'),
        ])
        with patch('services.qa_service._llm', return_value=model):
            events = list(stream_answer_events('What is the storage limit?'))
        text = ''.join(e['text'] for e in events if e['type'] == 'token')
        self.assertEqual(text, events[-1]['answer'])
        self.assertIn('100 kg', text)
        self.assertNotIn('private', text)
        self.assertEqual(events[-1]['sources'], [])

    def test_answer_context_drops_distractors_but_keeps_version_comparisons(self):
        docs = [Document(page_content=str(i), metadata={"source": f"source-{i}"}) for i in range(5)]
        self.assertEqual(_select_answer_docs("What is a bucket wheel excavator?", docs, "document"), docs[:2])
        self.assertEqual(_select_answer_docs("Compare the 2023 and 2024 rules", docs, "document"), docs)
        self.assertEqual(_select_answer_docs("Explain mine safety", docs, "summary"), docs)

    def test_normal_context_hides_locations_but_explicit_request_can_use_them(self):
        docs = [Document(page_content="The storage limit is 100 kg.",
                         metadata={"source": "rules.pdf", "page": 4, "year": "2025"})]
        ordinary = _format_context(docs)
        requested = _format_context(docs, include_locations=True)
        self.assertIn("100 kg", ordinary)
        self.assertIn("year 2025", ordinary)
        self.assertNotIn("rules.pdf", ordinary)
        self.assertNotIn("page 5", ordinary)
        self.assertIn("rules.pdf", requested)
        self.assertIn("page 5", requested)
        self.assertTrue(_wants_source_locations("Which page in the document states the limit?"))
        self.assertTrue(_wants_source_locations("Sources?"))
        self.assertTrue(_wants_source_locations("\u0938\u094d\u0930\u094b\u0924 \u092c\u0924\u093e\u0907\u090f"))
        self.assertFalse(_wants_source_locations("What is the reference voltage?"))

    def test_streamed_citations_are_removed_before_history_or_tts(self):
        model = self.model([AIMessageChunk(content="From Ref"), AIMessageChunk(
            content="erence 1, page 4, the storage limit is 100 kg [1].")])
        with patch('services.qa_service._llm', return_value=model):
            events = list(stream_answer_events('What is the storage limit?'))
        streamed = ''.join(e['text'] for e in events if e['type'] == 'token')
        self.assertEqual(streamed, "The storage limit is 100 kg.")
        self.assertEqual(events[-1]['answer'], streamed)
        self.assertNotIn("Reference", _format_chat_history([
            {"role": "assistant", "content": "From Reference 1, the storage limit is 100 kg."}
        ]))
        with patch('services.qa_service._llm', return_value=model):
            explicit = list(stream_answer_events("Which page in the document states the limit?"))
        self.assertIn("Reference 1", explicit[-1]['answer'])
        self.assertEqual(explicit[-1]['sources'][0]['source'], 'rules.pdf')

    def test_reasoning_only_response_returns_error_not_generic_answer(self):
        model = self.model([AIMessageChunk(content='', response_metadata={'done_reason': 'length'})])
        with patch('services.qa_service._llm', return_value=model):
            events = list(stream_answer_events('What is the storage limit?'))
        self.assertEqual(events[-1]['type'], 'error')
        self.assertIn('length', events[-1]['message'])
        self.assertNotIn('done', [e['type'] for e in events])

    def test_generation_exception_is_visible(self):
        with patch('services.qa_service._llm', side_effect=RuntimeError('Ollama connection refused')):
            events = list(stream_answer_events('What is the storage limit?'))
        self.assertEqual(events[-1]['type'], 'error')
        self.assertIn('connection refused', events[-1]['message'])

    def test_retrieved_context_and_latest_question_reach_chat_messages(self):
        captured = []
        def generate(inputs):
            for value in inputs:
                captured.extend(value.messages if hasattr(value, "messages") else [value])
            yield AIMessageChunk(content="The storage limit is 100 kg.")
        with patch("services.qa_service._llm", return_value=RunnableGenerator(generate)):
            events = list(stream_answer_events("What is the storage limit?"))
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual([message.type for message in captured], ["system", "human"])
        self.assertIn("storage limit is 100 kg", captured[1].content)
        self.assertIn("What is the storage limit?", captured[1].content)

    def test_non_streaming_endpoint_uses_same_answer(self):
        with patch('services.qa_service._llm', return_value=self.model([AIMessageChunk(content='100 kg [1].')])):
            response = answer_question('What is the storage limit?')
        self.assertEqual(response.answer, '100 kg.')

    def test_feedback_keeps_the_question_being_explained(self):
        result = _feedback_retrieval_query('I do not understand; explain simpler', [
            {'role': 'user', 'content': 'What is the storage limit?'},
        ])
        self.assertIn('storage limit', result)

    def test_old_model_tokens_and_final_answer_are_discarded_after_replacement(self):
        release = threading.Event()
        first_token = threading.Event()
        def slow(inputs):
            list(inputs)
            first_token.set()
            yield AIMessageChunk(content='Old first sentence. ')
            release.wait(2)
            yield AIMessageChunk(content='Old late sentence.')
        old_events = []
        with patch('services.qa_service._llm', side_effect=[RunnableGenerator(slow),
                                                          self.model([AIMessageChunk(content='New answer.')])]):
            consumer = threading.Thread(target=lambda: old_events.extend(stream_answer_events(
                'Explain storage limit', session_id='test-qa-replacement', request_id='old')))
            consumer.start()
            self.assertTrue(first_token.wait(1))
            new_events = list(stream_answer_events('Explain storage rules', session_id='test-qa-replacement', request_id='new'))
            release.set()
            consumer.join(1)
        self.assertFalse(consumer.is_alive())
        self.assertTrue(any(event['type'] == 'cancelled' for event in old_events))
        self.assertFalse(any(event['type'] == 'done' for event in old_events))
        self.assertNotIn('Old late', ''.join(event.get('text', '') for event in old_events))
        self.assertEqual(new_events[-1]['answer'], 'New answer.')


class BoundedStreamTests(unittest.TestCase):
    def test_new_request_cancels_stalled_old_request_without_blocking_other_session(self):
        release = threading.Event()
        finished = threading.Event()
        started = threading.Event()
        def stalled():
            started.set()
            yield {'type': 'status', 'message': 'Searching documents'}
            try:
                release.wait(2)
                yield {'type': 'done', 'answer': 'late'}
            finally:
                finished.set()
        first = registry.begin('test-replace-session')
        events = []
        consumer = threading.Thread(target=lambda: events.extend(bounded_events(stalled, 2, .01, request=first)))
        consumer.start()
        self.assertTrue(started.wait(1))
        second = registry.begin('test-replace-session')
        independent = registry.begin('test-independent-session')
        try:
            consumer.join(1)
            self.assertFalse(consumer.is_alive())
            self.assertTrue(first.cancelled.is_set())
            self.assertFalse(second.cancelled.is_set())
            self.assertFalse(independent.cancelled.is_set())
            self.assertNotIn('late', [event.get('answer') for event in events])
        finally:
            release.set()
            registry.cancel(second.session_id)
            registry.cancel(independent.session_id)
            self.assertTrue(finished.wait(1))

    def test_truncated_stream_is_not_reported_as_success(self):
        def incomplete():
            yield {'type': 'token', 'text': 'partial'}
        events = list(bounded_events(incomplete, 1, .01))
        self.assertEqual(events[-1]['type'], 'error')


class ModelSelectionTests(unittest.TestCase):
    def test_mandatory_reasoning_model_is_rejected(self):
        with patch.multiple('config', OLLAMA_THINK='auto', QA_REASONING_ENABLED=False), patch(
            'services.model_config.model_metadata', return_value={
                'capabilities': ['completion', 'thinking'],
                'thinking': {'values': ['low', 'medium', 'high'], 'default': 'medium'},
            }
        ):
            with self.assertRaisesRegex(ValueError, 'supports thinking'):
                thinking_setting('http://localhost', 'effort-model')

    def test_non_reasoning_model_explicitly_disables_thinking(self):
        with patch('config.OLLAMA_THINK', 'auto'), patch(
            'services.model_config.model_metadata', return_value={
                'capabilities': ['completion'], 'thinking': {'values': [False], 'default': False},
            }
        ):
            self.assertIs(thinking_setting('http://localhost', 'arbitrary-chat-model'), False)

    def test_optional_reasoning_model_is_rejected(self):
        with patch.multiple('config', OLLAMA_THINK='auto', QA_REASONING_ENABLED=False), patch(
            'services.model_config.model_metadata', return_value={'capabilities': ['completion', 'thinking']}
        ):
            with self.assertRaisesRegex(ValueError, 'supports thinking'):
                thinking_setting('http://localhost', 'arbitrary-reasoning-model')

    def test_embedding_model_in_chat_slot_is_rejected(self):
        with patch('config.OLLAMA_THINK', 'auto'), patch(
            'services.model_config.model_metadata', return_value={'capabilities': ['embedding']}
        ):
            with self.assertRaisesRegex(ValueError, 'not a chat'):
                thinking_setting('http://localhost', 'embed-only')


if __name__ == '__main__':
    unittest.main()
