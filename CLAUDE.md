# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Speech2Text is a local audio/video transcription tool built on OpenAI Whisper, with optional speaker diarization via pyannote.audio. It runs fully offline (models cached locally) and exposes the same engine through a CLI and a Tkinter GUI. The codebase and all user-facing strings are in Spanish.

## Commands

Tests (no Whisper models or pyannote are loaded — they exercise helpers, progress events, speaker assignment, and output writing):

```powershell
python -m unittest discover -s tests          # full suite
python -m unittest tests.test_transcriber_dia # single module
python -m unittest tests.test_transcriber_dia.StageLoggerTests.test_stage_logger_emits_progress_events  # single test
```

Run the apps:

```powershell
python transcriber_gui.py                                              # GUI
python transcriber_dia.py audio.m4a -m base -l es -o output/audio.txt  # CLI
python transcriber_dia.py audio.m4a -m base -l es -o output/audio.txt --diarize --hf-token TOKEN --num-speakers 2
```

There is no build or lint step. `ffmpeg`/`ffprobe` must be on `PATH` (used for audio conversion and duration). Dependencies install via `pip install -r requirements.txt` (venv) or `conda env create -f environment.yaml` (the README references `environment.yml`, but the actual file is `environment.yaml`).

## Architecture

**`transcriber_dia.py` is the engine; `transcriber_gui.py` is a thin frontend over it.** The GUI imports `transcribe_audio_with_diarization` lazily inside a worker thread (so heavy deps load off the UI thread) and never duplicates transcription logic.

`transcribe_audio_with_diarization(...)` is the single entry point for all processing. It runs a fixed sequence of stages (6 without diarization, 8 with) and always returns a result dict that carries `segments`, `text`, `language`, `device`, `fp16`, and `diarization`. The diarization pipeline: Whisper produces transcript segments → pyannote produces speaker turns → `assign_speaker_to_transcription` matches them by maximum time-overlap (falling back to nearest-midpoint when no overlap), tagging each segment with `speaker` and `overlap_confidence`.

**Progress is the central cross-cutting concern.** `StageLogger` both prints to the terminal and emits structured progress events to an optional `progress_callback`. Every event is a dict: `{stage_name, stage_index, total_stages, percent, message}`. Three adapters translate third-party progress into these same events:
- `WhisperProgressBar` — monkeypatches `whisper.transcribe.tqdm.tqdm` to intercept Whisper's internal progress (restored in a `finally`).
- `PyannoteProgressHook` — implements pyannote's hook protocol, mapping sub-steps (segmentation/embeddings/clustering) to sub-stage names.
- `prepare_audio_for_pyannote` — parses `ffmpeg -progress` output to report conversion percent.

When adding a stage or progress source, route it through `StageLogger` so both CLI and GUI stay in sync.

**GUI ↔ engine communication is a thread-safe queue.** The worker thread pushes `{"type": "progress"|"result"|"error", ...}` onto `self.events`; `_process_events` drains it on a 100ms Tk `after` loop and updates widgets. Never touch Tk widgets from the worker thread — go through the queue. The GUI passes `show_progress=False` (no terminal bars) but `verbose=True` and its `progress_callback`, so all feedback flows through events.

**Optional dependencies are imported defensively.** `torch`, `whisper`, `pyannote.audio`, and `tqdm` are wrapped in try/except at import and set to `None` when missing; `ensure_dependency(...)` raises a Spanish install hint at the point of use. This lets the test suite import the module without the heavy stack installed.

**Conventions worth preserving:**
- Whisper models are cached in `models/` (gitignored); `ensure_whisper_model` downloads with checksum verification and reuses existing files.
- Output always writes three files from one base path: `<base>.txt` (grouped by speaker), `<base>_detailed.txt` (timestamps + confidence), `<base>.json` (full result). The GUI's `_build_output_path` auto-increments a numeric suffix to avoid overwriting.
- Device/`fp16` selection is automatic: CUDA + fp16 when a GPU is available, else CPU.
- The HF token resolves from (in order) CLI `--hf-token`, the `HF_TOKEN` env var, or the gitignored `.env` file; the GUI persists it to `.env`.
