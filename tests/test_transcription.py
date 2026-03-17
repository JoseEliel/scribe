import unittest

from scribe.config import AppConfig
from scribe.transcription import SpeechPipeline


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = SpeechPipeline(
            AppConfig(
                hf_token="",
                model_name="dummy",
                default_llm="dummy",
                default_save_dir=".",
                enable_remote_tokenizer_lookup=False,
                max_speakers=8,
            )
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


if __name__ == "__main__":
    unittest.main()
