#!/usr/bin/env python3
# transcriptor.py
import os, sys, time, argparse, subprocess, tempfile
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import whisper

# ─────────── dependencias opcionales ────────────
try:
    from pyannote.audio import Pipeline
    import torch
except ImportError:
    Pipeline = None
# ------------------------------------------------


# ═════════════════ utilidades varias ═════════════
class ProgressCallback:
    """Barra de progreso durante la transcripción de Whisper"""
    def __init__(self, audio_duration=None):
        self.audio_duration = audio_duration
        self.pbar = None
    def __call__(self, chunk):
        if self.pbar is None:
            total = 100 if self.audio_duration is None else int(self.audio_duration)
            self.pbar = tqdm(total=total, desc="Transcribiendo",
                             unit="%" if self.audio_duration is None else "s",
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        if hasattr(chunk, 'end') and self.audio_duration:
            self.pbar.n = min(chunk.end, self.audio_duration)
            self.pbar.refresh()
    def close(self):
        if self.pbar:
            self.pbar.close()


def get_audio_duration(path: str) -> float | None:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
           "-show_entries", "stream=duration", "-of", "csv=p=0", path]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return float(out)
    except Exception:
        return None
# ═════════════════ fin utilidades ════════════════


# ══════════════════ DIARIZACIÓN ══════════════════
SAFE_EXT = {".wav", ".flac", ".ogg"}          # formatos leídos por libsndfile

def _convert_to_wav(src: str) -> str:
    """Convierte <src> a WAV 16 kHz mono y devuelve ruta temporal."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error",
           "-i", src, "-ar", "16000", "-ac", "1", wav_path]
    subprocess.run(cmd, check=True)
    return wav_path


def diarize_audio(path: str, token: str, num_speakers: int | None = None,
                  show_progress: bool = True) -> List[Tuple[float, float, str]]:
    if Pipeline is None:
        raise RuntimeError("pyannote.audio no está instalado")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # convertir a WAV si hace falta
    tmp_wav = None
    if Path(path).suffix.lower() not in SAFE_EXT:
        tmp_wav = _convert_to_wav(path)
        audio_for_pipe = tmp_wav
    else:
        audio_for_pipe = path

    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token=token)
    pipe.to(device)

    # ─── barra de progreso para diarización ─────────────────────────
     # ─── barra de progreso para diarización ─────────────────────────
    pbar = tqdm(total=100, desc="Diarizando", unit="%") if show_progress else None

    def hook(*args, **kw):
        if not pbar:
            return

        # Caso 1: llamado como (completed, total)
        if len(args) == 2 and all(isinstance(a, (int, float)) for a in args):
            completed, total = args

        # Caso 2: llamado con keywords
        elif "completed" in kw and "total" in kw:
            completed, total = kw["completed"], kw["total"]

        else:
            # llamado con ("segmentation", objeto) u otros; ignorar
            return

        if total:
            pbar.n = int(completed / total * 100)
            pbar.refresh()
    # ────────────────────────────────────────────────────────────────

    if num_speakers is None:
        diar = pipe(audio_for_pipe, hook=hook)
    else:
        diar = pipe({"audio": audio_for_pipe, "num_speakers": num_speakers},
                    hook=hook)

    if pbar:
        pbar.close()
    if tmp_wav and os.path.exists(tmp_wav):
        os.remove(tmp_wav)

    return [(t.start, t.end, spk)
            for t, _, spk in diar.itertracks(yield_label=True)]
# ═════════════════ fin diarización ═══════════════


def transcribe_audio(audio_path: str,
                     model_size="base",
                     language=None,
                     output_file=None,
                     show_progress=True,
                     enable_diarization=False,
                     hf_token=None,
                     num_speakers=None):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    print(f"Cargando modelo Whisper '{model_size}'…")
    model = whisper.load_model(model_size)

    duration = get_audio_duration(audio_path) if show_progress else None
    cb = ProgressCallback(duration) if show_progress else None

    options = {"fp16": False, "verbose": False}
    if language:
        options["language"] = language
    result = model.transcribe(audio_path, **options)
    if cb:
        cb.close()

    if enable_diarization:
        if not hf_token:
            raise ValueError("Debe proporcionar --hf-token o variable HF_TOKEN")
        turns = diarize_audio(audio_path, hf_token, num_speakers, show_progress)
        for seg in result["segments"]:
            best_label, best_ov = "unknown", 0.0
            for s, e, lab in turns:
                ov = max(0, min(seg['end'], e) - max(seg['start'], s))
                if ov > best_ov:
                    best_ov, best_label = ov, lab
            seg["speaker"] = best_label

    if output_file:
        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write(result["text"])

    return result


# ═══════════════════ CLI ══════════════════════════
def main():
    p = argparse.ArgumentParser(description="Transcriptor Whisper + diarización")
    p.add_argument("audio_file")
    p.add_argument("-m", "--model", default="base",
                   choices=["tiny", "base", "small", "medium", "large"])
    p.add_argument("-l", "--language", help="Código ISO (ej. es, en)")
    p.add_argument("-o", "--output", help="Archivo de salida .txt")
    p.add_argument("--no-progress", action="store_true",
                   help="Desactiva barras de progreso")
    p.add_argument("--diarize", action="store_true",
                   help="Habilita identificación de hablantes")
    p.add_argument("--hf-token", help="Token de HuggingFace")
    p.add_argument("--num-speakers", type=int,
                   help="Fijar nº exacto de hablantes")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Muestra estadísticas extra")
    args = p.parse_args()

    res = transcribe_audio(
        args.audio_file,
        model_size=args.model,
        language=args.language,
        output_file=args.output,
        show_progress=not args.no_progress,
        enable_diarization=args.diarize,
        hf_token=args.hf_token or os.getenv("HF_TOKEN"),
        num_speakers=args.num_speakers
    )

    print("\n" + "="*60)
    print("TRANSCRIPCIÓN:")
    print("="*60)
    for seg in res["segments"]:
        tag = f"{seg.get('speaker','')}: " if args.diarize else ""
        print(f"[{seg['start']:.2f}-{seg['end']:.2f}s] {tag}{seg['text']}")

    if args.verbose:
        print("\nIdioma detectado:", res["language"])
        print("Segmentos:", len(res["segments"]))


if __name__ == "__main__":
    main()
