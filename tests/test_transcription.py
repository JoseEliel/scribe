import unittest

from scribe.config import AppConfig
from scribe.transcription import (
    SpeechPipeline,
    is_apple_silicon_mac,
    resolve_transcription_backend,
)


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = SpeechPipeline(
            AppConfig(
                hf_token="",
                mlx_model_name="dummy-mlx",
                faster_whisper_model="dummy-fw",
                transcription_backend="auto",
                default_llm="dummy",
                default_save_dir=".",
                enable_remote_tokenizer_lookup=False,
                max_speakers=8,
            )
        )

    def test_backend_resolution_prefers_mlx_on_apple_silicon(self):
        self.assertTrue(is_apple_silicon_mac(system="Darwin", machine="arm64"))
        self.assertEqual(
            resolve_transcription_backend("auto", system="Darwin", machine="arm64"),
            "mlx",
        )

    def test_backend_resolution_uses_faster_whisper_off_apple_silicon(self):
        self.assertEqual(
            resolve_transcription_backend("auto", system="Linux", machine="x86_64"),
            "faster-whisper",
        )

    def test_forced_mlx_falls_back_when_unsupported(self):
        self.assertEqual(
            resolve_transcription_backend("mlx", system="Darwin", machine="x86_64"),
            "faster-whisper",
        )

    def test_align_and_reassemble_merges_adjacent_same_speaker_turns(self):
        words = [
            {"text": "Hello", "timestamp": (0.0, 0.5)},
            {"text": "there", "timestamp": (0.5, 0.9)},
            {"text": "again", "timestamp": (1.0, 1.2)},
        ]
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.5},
        ]
        turns = self.pipeline.align_and_reassemble(words, segments)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["speaker"], "SPEAKER_00")
        self.assertEqual(turns[0]["text"], "Hello there again")

    def test_format_transcript_for_llm_includes_timestamp_and_speaker(self):
        formatted = self.pipeline.format_transcript_for_llm(
            [{"speaker": "SPEAKER_01", "start_time": 65, "text": "Test line"}]
        )
        self.assertEqual(formatted, "[00:01:05] SPEAKER_01: Test line")

    def test_build_transcription_result_supports_object_segments(self):
        class Word:
            def __init__(self, word, start, end):
                self.word = word
                self.start = start
                self.end = end

        class Segment:
            def __init__(self, text, words):
                self.text = text
                self.words = words

        result = self.pipeline._build_transcription_result(
            [Segment("Hello world", [Word("Hello", 0.0, 0.3), Word("world", 0.3, 0.8)])]
        )
        self.assertEqual(result.raw_text, "Hello world")
        self.assertEqual(result.word_chunks[0]["text"], "Hello")
        self.assertEqual(result.word_chunks[1]["timestamp"], (0.3, 0.8))


if __name__ == "__main__":
    unittest.main()
