import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import transcriber_dia as td


class StageLoggerTests(unittest.TestCase):
    def test_stage_logger_emits_progress_events(self):
        events = []
        logger = td.StageLogger(total_steps=3, progress_callback=events.append)

        with contextlib.redirect_stdout(io.StringIO()):
            logger.start("Validando")
            logger.progress(50, "Mitad")
            logger.done("Listo")

        self.assertEqual(events[0]["stage_name"], "Validando")
        self.assertEqual(events[0]["stage_index"], 1)
        self.assertEqual(events[0]["total_stages"], 3)
        self.assertEqual(events[0]["percent"], 0)
        self.assertEqual(events[1]["percent"], 50)
        self.assertEqual(events[-1]["percent"], 100)

    def test_pyannote_hook_formats_substage_name(self):
        events = []
        logger = td.StageLogger(total_steps=8, progress_callback=events.append)
        with contextlib.redirect_stdout(io.StringIO()):
            logger.start("Diarizando hablantes")

            hook = td.PyannoteProgressHook(logger)
            hook("segmentation", completed=5, total=10)
            hook("embeddings", completed=3, total=6)

        stage_names = [event["stage_name"] for event in events]
        self.assertIn("Diarizando hablantes - Segmentation", stage_names)
        self.assertIn("Diarizando hablantes - Embeddings", stage_names)


class FormattingTests(unittest.TestCase):
    def test_format_seconds(self):
        self.assertEqual(td.format_seconds(None), "desconocido")
        self.assertEqual(td.format_seconds(12.345), "12.35s")
        self.assertEqual(td.format_seconds(75), "01:15")
        self.assertEqual(td.format_seconds(3661), "01:01:01")

    def test_format_bytes(self):
        self.assertEqual(td.format_bytes(512), "512.0B")
        self.assertEqual(td.format_bytes(1024), "1.0KB")
        self.assertEqual(td.format_bytes(1024 * 1024), "1.0MB")


class AudioPreparationTests(unittest.TestCase):
    def test_native_pyannote_format_does_not_create_temp_file(self):
        path, temp_path = td.prepare_audio_for_pyannote("audio.wav")

        self.assertEqual(path, "audio.wav")
        self.assertIsNone(temp_path)


class SpeakerAssignmentTests(unittest.TestCase):
    def test_assigns_speaker_by_overlap(self):
        transcription_segments = [
            {"start": 0.0, "end": 2.0, "text": "Hola"},
            {"start": 2.0, "end": 4.0, "text": "Chao"},
        ]
        diarization_segments = [
            {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
            {"start": 2.5, "end": 4.0, "speaker": "SPEAKER_01"},
        ]

        result = td.assign_speaker_to_transcription(
            transcription_segments,
            diarization_segments,
            show_progress=False,
        )

        self.assertEqual(result[0]["speaker"], "SPEAKER_00")
        self.assertEqual(result[1]["speaker"], "SPEAKER_01")
        self.assertGreater(result[0]["overlap_confidence"], 0)


class SaveTranscriptionTests(unittest.TestCase):
    def test_save_transcription_writes_expected_files(self):
        result = {
            "text": "Hola mundo",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "Hola mundo",
                    "speaker": "SPEAKER_00",
                    "overlap_confidence": 0.9,
                }
            ],
            "language": "es",
        }

        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / "salida.txt"
            td.save_transcription_with_speakers(result, str(output_file))

            txt = output_file.read_text(encoding="utf-8")
            detailed = (Path(tmp) / "salida_detailed.txt").read_text(encoding="utf-8")
            data = json.loads((Path(tmp) / "salida.json").read_text(encoding="utf-8"))

        self.assertIn("SPEAKER_00", txt)
        self.assertIn("Hola mundo", txt)
        self.assertIn("TRANSCRIPCION CON SPEAKERS", detailed)
        self.assertEqual(data["language"], "es")


if __name__ == "__main__":
    unittest.main()
