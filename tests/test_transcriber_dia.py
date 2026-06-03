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

    def test_info_does_not_animate_but_explicit_indeterminate_does(self):
        events = []
        logger = td.StageLogger(total_steps=3, progress_callback=events.append)

        with contextlib.redirect_stdout(io.StringIO()):
            logger.start("Cargando")
            logger.info("mensaje de log", always=True)
            logger.progress(None, "cargando modelo", indeterminate=True)

        # Un mensaje de log normal NO debe activar la barra animada.
        log_events = [e for e in events if e["message"] == "mensaje de log"]
        self.assertTrue(log_events)
        self.assertFalse(log_events[0]["indeterminate"])

        # Solo la fase marcada explicitamente es indeterminada.
        indeterminate = [e for e in events if e.get("indeterminate")]
        self.assertEqual(len(indeterminate), 1)
        self.assertIsNone(indeterminate[0]["percent"])

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


class _FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class _FakeInfo:
    def __init__(self, duration, language):
        self.duration = duration
        self.language = language


class _FakeModel:
    """Imita la API de faster_whisper.WhisperModel.transcribe."""

    def __init__(self, segments, info):
        self._segments = segments
        self._info = info
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter(self._segments), self._info


class FasterWhisperTranscriptionTests(unittest.TestCase):
    def test_builds_result_dict_and_emits_eta(self):
        events = []
        logger = td.StageLogger(total_steps=6, progress_callback=events.append)
        model = _FakeModel(
            [_FakeSegment(0.0, 2.0, " Hola"), _FakeSegment(2.0, 4.0, " mundo")],
            _FakeInfo(4.0, "es"),
        )

        with contextlib.redirect_stdout(io.StringIO()):
            logger.start("Transcribiendo audio")
            result = td.transcribe_with_faster_whisper(
                model,
                "audio.wav",
                {"beam_size": 5, "vad_filter": True},
                total_audio_seconds=4.0,
                logger=logger,
                show_terminal_progress=False,
            )

        self.assertEqual(result["language"], "es")
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["segments"][0]["start"], 0.0)
        self.assertEqual(result["segments"][0]["end"], 2.0)
        self.assertEqual(result["text"], "Hola mundo")

        etas = [e.get("eta_seconds") for e in events if e.get("eta_seconds") is not None]
        self.assertTrue(etas, "se esperaba al menos un evento con eta_seconds")

    def test_raises_when_cancelled(self):
        logger = td.StageLogger(total_steps=6)
        model = _FakeModel(
            [_FakeSegment(0.0, 2.0, "a"), _FakeSegment(2.0, 4.0, "b")],
            _FakeInfo(4.0, "es"),
        )

        with self.assertRaises(td.TranscriptionCancelled):
            td.transcribe_with_faster_whisper(
                model,
                "audio.wav",
                {},
                total_audio_seconds=4.0,
                logger=logger,
                show_terminal_progress=False,
                should_cancel=lambda: True,
            )

    def test_pyannote_hook_raises_when_cancelled(self):
        logger = td.StageLogger(total_steps=8)
        with contextlib.redirect_stdout(io.StringIO()):
            logger.start("Diarizando hablantes")
            hook = td.PyannoteProgressHook(logger, should_cancel=lambda: True)
            with self.assertRaises(td.TranscriptionCancelled):
                hook("segmentation", completed=1, total=10)


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
