import unittest

from scribe.analysis import build_prompt, choose_default_model


class AnalysisTests(unittest.TestCase):
    def test_build_prompt_supports_double_brace_placeholder(self):
        prompt = build_prompt("hello world", "Summary\n\n{{transcript}}")
        self.assertIn("hello world", prompt)
        self.assertNotIn("{{transcript}}", prompt)

    def test_build_prompt_supports_single_brace_placeholder(self):
        prompt = build_prompt("hello world", "Summary\n\n{transcript}")
        self.assertIn("hello world", prompt)
        self.assertNotIn("{transcript}", prompt)

    def test_build_prompt_appends_transcript_when_placeholder_missing(self):
        prompt = build_prompt("hello world", "Summarize this.")
        self.assertTrue(prompt.startswith("Summarize this."))
        self.assertIn("Transcript:\nhello world", prompt)

    def test_choose_default_model_prefers_requested_model(self):
        models = ["mistral:latest", "gemma3:27b"]
        self.assertEqual(choose_default_model(models, preferred="gemma3:27b"), "gemma3:27b")


if __name__ == "__main__":
    unittest.main()

