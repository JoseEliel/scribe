import os
import tempfile
import unittest

from scribe.files import (
    _extract_existing_original_path,
    append_markdown_section,
    append_section_text,
    build_diar_save_path,
    resolve_output_directory,
)


class FilesTests(unittest.TestCase):
    def test_append_section_text_replaces_existing_section(self):
        initial = "=== One ===\nold\n\n=== Two ===\nkeep"
        updated = append_section_text(initial, "One", "new")
        self.assertIn("=== One ===\nnew", updated)
        self.assertNotIn("=== One ===\nold", updated)
        self.assertIn("=== Two ===\nkeep", updated)

    def test_append_markdown_section_replaces_existing_section(self):
        initial = "<!-- SECTION: One -->\n## One\n\nold"
        updated = append_markdown_section(initial, "One", "new")
        self.assertIn("## One", updated)
        self.assertIn("new", updated)
        self.assertNotIn("\nold", updated)

    def test_extract_existing_original_path_requires_absolute_existing_file(self):
        with tempfile.NamedTemporaryFile() as handle:
            valid = _extract_existing_original_path(handle.name)
            self.assertEqual(valid, os.path.abspath(handle.name))
        self.assertEqual(_extract_existing_original_path("relative/file.txt"), "")
        self.assertEqual(_extract_existing_original_path("/does/not/exist.txt"), "")

    def test_resolve_output_directory_prefers_custom_directory(self):
        with tempfile.TemporaryDirectory() as custom_dir:
            directory, mode = resolve_output_directory(
                audio_path="",
                default_save_dir=custom_dir,
                output_dir=custom_dir,
            )
            self.assertEqual(directory, custom_dir)
            self.assertEqual(mode, "custom")

    def test_build_diar_save_path_creates_unique_names(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = os.path.join(tmp_dir, "meeting.wav")
            open(source, "w", encoding="utf-8").close()
            first, _ = build_diar_save_path(source, 1, tmp_dir)
            open(first, "w", encoding="utf-8").close()
            second, _ = build_diar_save_path(source, 1, tmp_dir)
            self.assertNotEqual(first, second)
            self.assertTrue(second.endswith("_diarized_1.txt"))


if __name__ == "__main__":
    unittest.main()

