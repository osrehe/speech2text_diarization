import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import transcriber_gui as gui


class GuiHelperTests(unittest.TestCase):
    def test_build_output_path_uses_incremental_suffix(self):
        instance = gui.TranscriberGUI.__new__(gui.TranscriberGUI)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            output_dir.mkdir()
            (output_dir / "audio.txt").write_text("existente", encoding="utf-8")

            with patch.object(gui, "__file__", str(Path(tmp) / "transcriber_gui.py")):
                output = instance._build_output_path(str(Path(tmp) / "audio.m4a"))

        self.assertEqual(output.name, "audio_1.txt")

    def test_format_result_without_diarization(self):
        instance = gui.TranscriberGUI.__new__(gui.TranscriberGUI)
        result = {"text": "Texto final", "segments": [], "diarization": None}

        self.assertEqual(instance._format_result(result), "Texto final")

    def test_format_result_with_diarization_groups_speakers(self):
        instance = gui.TranscriberGUI.__new__(gui.TranscriberGUI)
        result = {
            "diarization": {"speakers": ["SPEAKER_00", "SPEAKER_01"]},
            "segments": [
                {"speaker": "SPEAKER_00", "text": "Hola"},
                {"speaker": "SPEAKER_00", "text": "como estas"},
                {"speaker": "SPEAKER_01", "text": "bien"},
            ],
        }

        formatted = instance._format_result(result)

        self.assertIn("SPEAKER_00:", formatted)
        self.assertIn("SPEAKER_01:", formatted)
        self.assertIn("Hola", formatted)
        self.assertIn("bien", formatted)

    def test_save_token_to_env_updates_hf_token(self):
        instance = gui.TranscriberGUI.__new__(gui.TranscriberGUI)

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OTHER=value\nHF_TOKEN=old\n", encoding="utf-8")

            with patch.object(gui, "ENV_PATH", env_path):
                instance._save_token_to_env("new-token")

            content = env_path.read_text(encoding="utf-8")

        self.assertIn("OTHER=value", content)
        self.assertIn("HF_TOKEN=new-token", content)
        self.assertNotIn("HF_TOKEN=old", content)


if __name__ == "__main__":
    unittest.main()
