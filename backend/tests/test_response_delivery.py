import threading
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableGenerator

from services.model_config import thinking_setting
from services.qa_service import _feedback_retrieval_query, _is_general_query, answer_question, stream_answer_events
from services.response_stream import AnswerTextFilter, bounded_events


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
            patch('services.response_stream.RESPONSE_SLOT', threading.BoundedSemaphore(1)),
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
        self.assertEqual(events[-1]['sources'][0]['source'], 'rules.pdf')

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

    def test_non_streaming_endpoint_uses_same_answer(self):
        with patch('services.qa_service._llm', return_value=self.model([AIMessageChunk(content='100 kg [1].')])):
            response = answer_question('What is the storage limit?')
        self.assertEqual(response.answer, '100 kg [1].')

    def test_feedback_keeps_the_question_being_explained(self):
        result = _feedback_retrieval_query('I do not understand; explain simpler', [
            {'role': 'user', 'content': 'What is the storage limit?'},
        ])
        self.assertIn('storage limit', result)


class BoundedStreamTests(unittest.TestCase):
    def test_stalled_retrieval_times_out_and_does_not_spawn_more_workers(self):
        release = threading.Event()
        finished = threading.Event()
        def stalled():
            yield {'type': 'status', 'message': 'Searching documents'}
            try:
                release.wait(2)
                yield {'type': 'done', 'answer': 'late'}
            finally:
                finished.set()
        with patch('services.response_stream.RESPONSE_SLOT', threading.BoundedSemaphore(1)):
            try:
                events = list(bounded_events(stalled, .05, .01))
                self.assertEqual(events[-1]['type'], 'error')
                self.assertIn('Searching documents', events[-1]['message'])
                busy = list(bounded_events(stalled, .05, .01))
                self.assertIn('previous answer', busy[0]['message'])
            finally:
                release.set()
                self.assertTrue(finished.wait(1))

    def test_truncated_stream_is_not_reported_as_success(self):
        def incomplete():
            yield {'type': 'token', 'text': 'partial'}
        events = list(bounded_events(incomplete, 1, .01))
        self.assertEqual(events[-1]['type'], 'error')


class ModelSelectionTests(unittest.TestCase):
    def test_mandatory_reasoning_model_uses_advertised_low_effort(self):
        with patch.multiple('config', OLLAMA_THINK='auto', QA_REASONING_ENABLED=False), patch(
            'services.model_config.model_metadata', return_value={
                'capabilities': ['completion', 'thinking'],
                'thinking': {'values': ['low', 'medium', 'high'], 'default': 'medium'},
            }
        ):
            self.assertEqual(thinking_setting('http://localhost', 'effort-model'), 'low')

    def test_non_reasoning_model_omits_thinking_option(self):
        with patch('config.OLLAMA_THINK', 'auto'), patch(
            'services.model_config.model_metadata', return_value={'capabilities': ['completion']}
        ):
            self.assertIsNone(thinking_setting('http://localhost', 'arbitrary-chat-model'))

    def test_reasoning_model_can_disable_thinking(self):
        with patch.multiple('config', OLLAMA_THINK='auto', QA_REASONING_ENABLED=False), patch(
            'services.model_config.model_metadata', return_value={'capabilities': ['completion', 'thinking']}
        ):
            self.assertIs(thinking_setting('http://localhost', 'arbitrary-reasoning-model'), False)

    def test_embedding_model_in_chat_slot_is_rejected(self):
        with patch('config.OLLAMA_THINK', 'auto'), patch(
            'services.model_config.model_metadata', return_value={'capabilities': ['embedding']}
        ):
            with self.assertRaisesRegex(ValueError, 'not a chat'):
                thinking_setting('http://localhost', 'embed-only')


if __name__ == '__main__':
    unittest.main()
