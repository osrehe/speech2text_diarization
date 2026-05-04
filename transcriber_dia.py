import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    import whisper
except ImportError:
    whisper = None

try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
except ImportError:
    Pipeline = None
    ProgressHook = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac", ".mp4"}
PYANNOTE_NATIVE_FORMATS = {".wav", ".flac", ".ogg"}


ProgressCallback = Callable[[Dict[str, Any]], None]


class StageLogger:
    """Muestra avance por etapas y tiempos de ejecucion."""

    def __init__(
        self,
        total_steps: int,
        verbose: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self.total_steps = total_steps
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.current_step = 0
        self.current_stage = ""
        self.stage_start = None
        self.global_start = time.perf_counter()

    def start(self, message: str) -> None:
        self.current_step += 1
        self.current_stage = message
        self.stage_start = time.perf_counter()
        print(f"\n[{self.current_step}/{self.total_steps}] {message}...")
        self.progress(0, f"{message}...")

    def done(self, message: str = "OK") -> None:
        elapsed = time.perf_counter() - self.stage_start if self.stage_start else 0
        print(f"[OK] {message} ({format_seconds(elapsed)})")
        self.progress(100, f"{message} ({format_seconds(elapsed)})")

    def info(self, message: str, always: bool = False) -> None:
        if always or self.verbose:
            print(f"  - {message}")
        self.progress(None, message)

    def summary(self) -> None:
        elapsed = time.perf_counter() - self.global_start
        print(f"\nProceso completado en {format_seconds(elapsed)}")
        self.progress(100, f"Proceso completado en {format_seconds(elapsed)}")

    def progress(
        self,
        percent: Optional[int],
        message: str = "",
        stage_name: Optional[str] = None,
    ) -> None:
        if not self.progress_callback:
            return

        event = {
            "stage_name": stage_name or self.current_stage,
            "stage_index": self.current_step,
            "total_stages": self.total_steps,
            "percent": percent,
            "message": message,
        }
        self.progress_callback(event)


def format_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "desconocido"

    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, remaining_seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def format_bytes(num_bytes: float) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def get_audio_duration(audio_path: str) -> Optional[float]:
    """Obtener duracion del audio usando ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-show_entries",
            "format=duration", "-of", "csv=p=0", audio_path
        ], capture_output=True, text=True)

        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def prepare_audio_for_pyannote(audio_path: str, logger: Optional[StageLogger] = None) -> tuple[str, Optional[str]]:
    """Convierte formatos no nativos a WAV 16 kHz mono para pyannote."""
    extension = Path(audio_path).suffix.lower()
    if extension in PYANNOTE_NATIVE_FORMATS:
        return audio_path, None

    temp_file = tempfile.NamedTemporaryFile(prefix="pyannote_", suffix=".wav", delete=False)
    temp_file.close()

    if logger:
        logger.info("Convirtiendo audio para diarizacion: WAV 16kHz mono", always=True)

    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        audio_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        temp_file.name,
    ]

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        Path(temp_file.name).unlink(missing_ok=True)
        raise RuntimeError("ffmpeg no esta instalado o no esta disponible en PATH") from error
    except subprocess.CalledProcessError as error:
        Path(temp_file.name).unlink(missing_ok=True)
        raise RuntimeError("No se pudo convertir el audio para diarizacion con ffmpeg") from error

    if logger:
        logger.info(f"Audio temporal para diarizacion: {temp_file.name}", always=True)

    return temp_file.name, temp_file.name


def get_runtime_device() -> str:
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_cuda_name() -> Optional[str]:
    if torch is None:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def ensure_dependency(module: Any, package_name: str, install_hint: str) -> None:
    if module is None:
        raise RuntimeError(
            f"No se pudo importar {package_name}. Instala las dependencias con: {install_hint}"
        )


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_whisper_model(model_size: str, models_dir: Path, logger: StageLogger) -> None:
    """Descarga el modelo Whisper con progreso controlado por la aplicacion."""
    model_urls = getattr(whisper, "_MODELS", {})
    model_url = model_urls.get(model_size)

    if not model_url:
        logger.info("No se pudo obtener URL interna del modelo; Whisper usara su descarga por defecto", always=True)
        return

    target = models_dir / os.path.basename(model_url)
    expected_sha256 = model_url.split("/")[-2] if "/" in model_url else None

    if target.exists():
        logger.progress(15, "Verificando modelo local")
        if not expected_sha256 or sha256_file(target) == expected_sha256:
            logger.info(f"Modelo local encontrado: {target.name} ({format_bytes(target.stat().st_size)})", always=True)
            logger.progress(85, "Modelo local listo")
            return

        logger.info(f"Modelo local incompleto o corrupto: {target.name}. Se descargara nuevamente.", always=True)
        target.unlink()

    temp_target = target.with_suffix(target.suffix + ".download")
    if temp_target.exists():
        temp_target.unlink()

    logger.info(f"Descargando modelo {model_size}: {target.name}", always=True)
    start_time = time.perf_counter()
    last_logged_percent = -5
    downloaded = 0

    with urllib.request.urlopen(model_url) as source, open(temp_target, "wb") as output:
        total_size = int(source.info().get("Content-Length") or 0)
        if total_size:
            logger.info(f"Tamano total: {format_bytes(total_size)}", always=True)

        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break

            output.write(chunk)
            downloaded += len(chunk)
            elapsed = max(time.perf_counter() - start_time, 0.001)
            speed = downloaded / elapsed

            if total_size:
                percent = int(downloaded / total_size * 100)
                logger.progress(percent, "")

                if percent >= last_logged_percent + 5 or percent == 100:
                    logger.info(
                        f"Descarga {percent}% | {format_bytes(downloaded)}/{format_bytes(total_size)} | {format_bytes(speed)}/s",
                        always=True,
                    )
                    last_logged_percent = percent
            else:
                logger.progress(None, f"Descargado {format_bytes(downloaded)} | {format_bytes(speed)}/s")

    logger.progress(98, "Verificando descarga")
    if expected_sha256 and sha256_file(temp_target) != expected_sha256:
        temp_target.unlink(missing_ok=True)
        raise RuntimeError("La descarga del modelo no coincide con el checksum esperado")

    temp_target.replace(target)
    logger.info(f"Modelo guardado en: {target}", always=True)


class WhisperProgressBar:
    """Adapta el progreso interno de Whisper a StageLogger."""

    def __init__(
        self,
        original_tqdm: Any,
        logger: StageLogger,
        show_terminal_progress: bool,
        *args: Any,
        **kwargs: Any,
    ):
        self.logger = logger
        self.total = int(kwargs.get("total") or 0)
        self.n = int(kwargs.get("initial") or 0)
        self.start_time = time.perf_counter()
        self.last_logged_percent = -5
        self.last_emit_time = 0.0
        self.real_bar = original_tqdm(*args, **kwargs) if show_terminal_progress else None

    def __enter__(self) -> "WhisperProgressBar":
        if self.real_bar:
            self.real_bar.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.real_bar:
            self.real_bar.__exit__(exc_type, exc, tb)

    def update(self, amount: int = 1) -> None:
        if self.real_bar:
            self.real_bar.update(amount)
            self.n = int(getattr(self.real_bar, "n", self.n + amount))
        else:
            self.n += amount

        self._emit_progress()

    def close(self) -> None:
        if self.real_bar:
            self.real_bar.close()

    def set_description(self, *args: Any, **kwargs: Any) -> None:
        if self.real_bar:
            self.real_bar.set_description(*args, **kwargs)

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        if self.real_bar:
            self.real_bar.set_postfix(*args, **kwargs)

    def _emit_progress(self) -> None:
        if self.total <= 0:
            self.logger.progress(None, f"Frames procesados: {self.n}")
            return

        percent = min(100, int(self.n / self.total * 100))
        self.logger.progress(percent, "")

        now = time.perf_counter()
        if percent < self.last_logged_percent + 5 and now - self.last_emit_time < 2:
            return

        elapsed = max(now - self.start_time, 0.001)
        rate = self.n / elapsed
        remaining_frames = max(self.total - self.n, 0)
        remaining_seconds = remaining_frames / rate if rate > 0 else None

        self.logger.info(
            f"Transcripcion {self.n}/{self.total} "
            f"[{format_seconds(elapsed)}<{format_seconds(remaining_seconds)}, {rate:.2f}frames/s]",
            always=True,
        )
        self.last_logged_percent = percent
        self.last_emit_time = now

    def __getattr__(self, name: str) -> Any:
        if self.real_bar:
            return getattr(self.real_bar, name)
        raise AttributeError(name)


def transcribe_with_progress(
    model: Any,
    audio_file_path: str,
    transcribe_options: Dict[str, Any],
    logger: StageLogger,
    show_terminal_progress: bool,
) -> Dict[str, Any]:
    """Ejecuta Whisper conectando su progreso interno a la GUI."""
    try:
        transcribe_module = importlib.import_module("whisper.transcribe")
        original_tqdm = transcribe_module.tqdm.tqdm
    except Exception:
        return model.transcribe(audio_file_path, **transcribe_options)

    def progress_factory(*args: Any, **kwargs: Any) -> WhisperProgressBar:
        return WhisperProgressBar(original_tqdm, logger, show_terminal_progress, *args, **kwargs)

    transcribe_module.tqdm.tqdm = progress_factory
    try:
        return model.transcribe(audio_file_path, **transcribe_options)
    finally:
        transcribe_module.tqdm.tqdm = original_tqdm


class PyannoteProgressHook:
    """Envia a la GUI el progreso real reportado por pyannote."""

    def __init__(self, logger: StageLogger):
        self.logger = logger
        self.last_percent = -1
        self.last_message_at = 0.0
        self.last_step_name = None

    def __enter__(self) -> "PyannoteProgressHook":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def _format_step_name(self, step_name: Any) -> str:
        normalized = str(step_name or "procesando").strip()
        aliases = {
            "segmentation": "Segmentation",
            "embedding": "Embeddings",
            "embeddings": "Embeddings",
            "clustering": "Clustering",
        }
        return aliases.get(normalized.lower(), normalized.replace("_", " ").title())

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        completed = kwargs.get("completed")
        total = kwargs.get("total")
        step_name = kwargs.get("step_name") or kwargs.get("name")

        if step_name is None and args:
            step_name = str(args[0])

        if completed is None or total is None:
            numeric_args = [arg for arg in args if isinstance(arg, (int, float))]
            if len(numeric_args) >= 2:
                completed, total = numeric_args[0], numeric_args[1]

        if not total:
            if step_name:
                pretty_step = self._format_step_name(step_name)
                display_stage = f"{self.logger.current_stage} - {pretty_step}"
                self.logger.progress(None, f"Diarizacion: {pretty_step}", stage_name=display_stage)
            return

        substage = self._format_step_name(step_name)
        display_stage = f"{self.logger.current_stage} - {substage}"

        if substage != self.last_step_name:
            self.logger.progress(0, f"Iniciando subetapa {substage}", stage_name=display_stage)
            self.last_percent = -1
            self.last_step_name = substage

        percent = max(0, min(100, int(float(completed) / float(total) * 100)))
        self.logger.progress(percent, "", stage_name=display_stage)

        now = time.perf_counter()
        if percent == self.last_percent and now - self.last_message_at < 2:
            return

        label = f"Diarizacion {substage}" if substage else "Diarizacion"
        self.logger.info(f"{label}: {completed}/{total} ({percent}%)", always=True)
        self.last_percent = percent
        self.last_message_at = now


def diarize_audio(
    audio_file_path: str,
    hf_token: str,
    device: str,
    show_progress: bool = True,
    num_speakers: Optional[int] = None,
    logger: Optional[StageLogger] = None,
) -> Optional[Dict[str, Any]]:
    """Realizar diarizacion de speakers usando pyannote.audio."""
    temp_audio_path = None
    try:
        ensure_dependency(torch, "torch", "python -m pip install -r requirements.txt")
        ensure_dependency(Pipeline, "pyannote.audio", "python -m pip install -r requirements.txt")
        ensure_dependency(ProgressHook, "pyannote.audio ProgressHook", "python -m pip install -r requirements.txt")

        pipeline = None
        selected_pipeline = None
        pipeline_candidates = [
            "pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization",
        ]

        last_error = None
        for pipeline_name in pipeline_candidates:
            try:
                if logger:
                    logger.info(f"Cargando pipeline {pipeline_name}", always=True)

                pipeline = Pipeline.from_pretrained(
                    pipeline_name,
                    use_auth_token=hf_token
                )

                if pipeline is None:
                    raise RuntimeError("Pipeline.from_pretrained devolvio None")

                selected_pipeline = pipeline_name
                break
            except Exception as error:
                last_error = error
                if logger:
                    logger.info(f"No se pudo cargar {pipeline_name}: {error}", always=True)

        if pipeline is None:
            raise RuntimeError(
                "No se pudo cargar ningun pipeline de diarizacion. "
                "Verifica token, permisos y condiciones aceptadas en Hugging Face."
            ) from last_error

        if logger:
            logger.info(f"Pipeline activo: {selected_pipeline}", always=True)

        pipeline.to(torch.device(device))

        if logger:
            logger.info(f"Diarizacion usando {device.upper()}", always=True)
            if num_speakers:
                logger.info(f"Numero de hablantes fijado: {num_speakers}", always=True)

        audio_for_diarization, temp_audio_path = prepare_audio_for_pyannote(audio_file_path, logger)

        diarization_input: Dict[str, Any] = {"audio": audio_for_diarization}
        if num_speakers:
            diarization_input["num_speakers"] = num_speakers

        if logger:
            with PyannoteProgressHook(logger) as hook:
                diarization = pipeline(diarization_input, hook=hook)
        elif show_progress:
            with ProgressHook() as hook:
                diarization = pipeline(diarization_input, hook=hook)
        else:
            diarization = pipeline(diarization_input)

        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "duration": turn.end - turn.start
            })

        speakers = sorted(set(seg["speaker"] for seg in speaker_segments))
        speaker_stats = {}

        for speaker in speakers:
            speaker_segs = [seg for seg in speaker_segments if seg["speaker"] == speaker]
            total_time = sum(seg["duration"] for seg in speaker_segs)
            speaker_stats[speaker] = {
                "total_time": total_time,
                "segments": len(speaker_segs)
            }

        return {
            "segments": speaker_segments,
            "speakers": speakers,
            "speaker_stats": speaker_stats
        }

    except Exception as e:
        print(f"Error en diarizacion: {e}")
        print("Nota: La diarizacion requiere pyannote.audio, acceso al modelo y token de Hugging Face")
        return None
    finally:
        if temp_audio_path:
            Path(temp_audio_path).unlink(missing_ok=True)


def assign_speaker_to_transcription(
    transcription_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]],
    show_progress: bool = True,
    logger: Optional[StageLogger] = None,
) -> List[Dict[str, Any]]:
    """Asignar speakers a segmentos de transcripcion."""
    sorted_diarization = sorted(diarization_segments, key=lambda seg: seg["start"])
    enhanced_segments = []

    if show_progress and tqdm is not None:
        iterator = tqdm(transcription_segments, desc="Asignando speakers", unit="seg")
    else:
        iterator = transcription_segments

    total_segments = len(transcription_segments)
    for index, trans_seg in enumerate(iterator, 1):
        trans_start = trans_seg["start"]
        trans_end = trans_seg["end"]
        trans_mid = (trans_start + trans_end) / 2

        best_speaker = None
        best_overlap = 0

        for diar_seg in sorted_diarization:
            diar_start = diar_seg["start"]
            diar_end = diar_seg["end"]

            if diar_start > trans_end:
                break
            if diar_end < trans_start:
                continue

            overlap_start = max(trans_start, diar_start)
            overlap_end = min(trans_end, diar_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg["speaker"]

        if best_speaker is None:
            min_distance = float("inf")
            for diar_seg in sorted_diarization:
                diar_mid = (diar_seg["start"] + diar_seg["end"]) / 2
                distance = abs(trans_mid - diar_mid)
                if distance < min_distance:
                    min_distance = distance
                    best_speaker = diar_seg["speaker"]

        enhanced_segments.append({
            **trans_seg,
            "speaker": best_speaker or "SPEAKER_UNKNOWN",
            "overlap_confidence": best_overlap / (trans_end - trans_start) if trans_end > trans_start else 0
        })

        if logger and total_segments:
            percent = int(index / total_segments * 100)
            logger.progress(percent, f"Asignados {index}/{total_segments} segmentos")

    return enhanced_segments


def transcribe_audio_with_diarization(
    audio_file_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    output_file: Optional[str] = None,
    show_progress: bool = True,
    hf_token: Optional[str] = None,
    enable_diarization: bool = False,
    num_speakers: Optional[int] = None,
    verbose: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Transcribir audio y, opcionalmente, identificar speakers."""
    total_steps = 8 if enable_diarization else 6
    logger = StageLogger(
        total_steps=total_steps,
        verbose=verbose,
        progress_callback=progress_callback,
    )

    logger.start("Validando archivo de audio")
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"El archivo {audio_file_path} no existe")

    file_extension = Path(audio_file_path).suffix.lower()
    if file_extension not in SUPPORTED_FORMATS:
        logger.info(f"Advertencia: el formato {file_extension} puede no ser compatible", always=True)
        logger.info(f"Formatos esperados: {', '.join(sorted(SUPPORTED_FORMATS))}", always=True)

    file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
    logger.info(f"Archivo: {audio_file_path}", always=True)
    logger.info(f"Tamano: {file_size_mb:.2f} MB")
    logger.done("Archivo validado")

    logger.start("Detectando hardware")
    ensure_dependency(torch, "torch", "python -m pip install -r requirements.txt")
    ensure_dependency(whisper, "openai-whisper", "python -m pip install -r requirements.txt")
    device = get_runtime_device()
    cuda_name = get_cuda_name()
    use_fp16 = device == "cuda"

    logger.info(f"Dispositivo: {device.upper()}", always=True)
    if cuda_name:
        logger.info(f"GPU: {cuda_name}", always=True)
    logger.info(f"fp16: {'activado' if use_fp16 else 'desactivado'}", always=True)
    logger.done("Hardware detectado")

    logger.start("Analizando duracion del audio")
    audio_duration = get_audio_duration(audio_file_path)
    if audio_duration:
        logger.info(f"Duracion: {format_seconds(audio_duration)}", always=True)
    else:
        logger.info("No se pudo obtener la duracion con ffprobe", always=True)
    logger.done("Duracion analizada")

    logger.start(f"Cargando modelo Whisper '{model_size}'")
    logger.progress(10, "Preparando carga del modelo")
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)
    logger.info(f"Carpeta de modelos: {models_dir}", always=True)
    ensure_whisper_model(model_size, models_dir, logger)
    logger.progress(99, "Inicializando modelo en memoria")
    model = whisper.load_model(model_size, device=device, download_root=str(models_dir))
    logger.done("Modelo cargado")

    logger.start("Transcribiendo audio")
    logger.progress(5, "Preparando transcripcion")
    transcribe_options = {
        "fp16": use_fp16,
        "verbose": show_progress,
    }

    if language:
        transcribe_options["language"] = language
        logger.info(f"Idioma fijado: {language}", always=True)
    else:
        logger.info("Idioma no fijado: Whisper lo detectara automaticamente", always=True)

    result = transcribe_with_progress(
        model,
        audio_file_path,
        transcribe_options,
        logger,
        show_terminal_progress=show_progress,
    )
    logger.progress(95, "Procesando segmentos de Whisper")
    segment_count = len(result.get("segments", []))
    result["device"] = device
    result["fp16"] = use_fp16
    logger.info(f"Idioma detectado: {result.get('language', 'desconocido')}", always=True)
    logger.info(f"Segmentos generados: {segment_count}", always=True)
    logger.done("Transcripcion completada")

    diarization_result = None
    if enable_diarization:
        logger.start("Diarizando hablantes")
        if not hf_token:
            raise ValueError("Debe proporcionar --hf-token o configurar la variable de entorno HF_TOKEN para usar --diarize")

        diarization_result = diarize_audio(
            audio_file_path,
            hf_token=hf_token,
            device=device,
            show_progress=show_progress,
            num_speakers=num_speakers,
            logger=logger
        )

        if diarization_result:
            logger.info(f"Speakers detectados: {len(diarization_result['speakers'])}", always=True)
            for speaker, stats in diarization_result["speaker_stats"].items():
                logger.info(f"{speaker}: {format_seconds(stats['total_time'])} en {stats['segments']} segmentos", always=True)
            logger.done("Diarizacion completada")
        else:
            logger.done("Diarizacion omitida por error")

        logger.start("Asignando speakers a la transcripcion")
        if diarization_result and diarization_result["segments"]:
            result["segments"] = assign_speaker_to_transcription(
                result["segments"],
                diarization_result["segments"],
                show_progress=show_progress,
                logger=logger
            )
            logger.done("Speakers asignados")
        else:
            logger.done("No hay segmentos de diarizacion para asignar")

    result["diarization"] = diarization_result

    logger.start("Guardando salida y finalizando")
    if output_file:
        logger.progress(20, "Escribiendo archivos")
        save_transcription_with_speakers(result, output_file)
        base_path = Path(output_file).with_suffix("")
        logger.info(f"Texto: {base_path}.txt", always=True)
        logger.info(f"Detalle: {base_path}_detailed.txt", always=True)
        logger.info(f"JSON: {base_path}.json", always=True)
        logger.done("Archivos guardados")
    else:
        logger.done("Salida lista en memoria")

    logger.summary()
    return result


def save_transcription_with_speakers(result: Dict[str, Any], output_file: str) -> None:
    """Guardar transcripcion en texto, detalle y JSON."""
    base_path = Path(output_file).with_suffix("")
    has_speakers = any("speaker" in segment for segment in result["segments"])

    with open(f"{base_path}.txt", "w", encoding="utf-8") as f:
        current_speaker = None
        for segment in result["segments"]:
            speaker = segment.get("speaker", "SPEAKER_UNKNOWN")
            text = segment["text"].strip()

            if has_speakers and speaker != current_speaker:
                f.write(f"\n{speaker}:\n")
                current_speaker = speaker

            f.write(f"{text} ")

    with open(f"{base_path}_detailed.txt", "w", encoding="utf-8") as f:
        title = "TRANSCRIPCION CON SPEAKERS" if has_speakers else "TRANSCRIPCION"
        f.write(f"{title}\n")
        f.write("="*50 + "\n\n")

        for i, segment in enumerate(result["segments"], 1):
            speaker = segment.get("speaker", "SPEAKER_UNKNOWN")
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            confidence = segment.get("overlap_confidence", 0)

            speaker_text = f" {speaker} (conf: {confidence:.2f})" if has_speakers else ""
            f.write(f"{i:3d}. [{start:6.2f}s - {end:6.2f}s]{speaker_text}\n")
            f.write(f"     {text}\n\n")

    with open(f"{base_path}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Transcriptor de audio con Whisper y diarizacion opcional")
    parser.add_argument("audio_file", help="Ruta al archivo de audio")
    parser.add_argument("-m", "--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Tamano del modelo Whisper (default: base)")
    parser.add_argument("-l", "--language", help="Codigo de idioma (ej: es, en, fr)")
    parser.add_argument("-o", "--output", help="Archivo de salida para la transcripcion")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Mostrar informacion detallada y feedback por etapa")
    parser.add_argument("--no-progress", action="store_true",
                        help="Desactivar barras de progreso internas")
    parser.add_argument("--diarize", action="store_true",
                        help="Activar identificacion de speakers con pyannote")
    parser.add_argument("--no-diarization", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--hf-token", help="Token de Hugging Face para modelos de diarizacion")
    parser.add_argument("--num-speakers", type=int,
                        help="Numero esperado de speakers para acelerar/mejorar diarizacion")

    args = parser.parse_args()

    try:
        enable_diarization = args.diarize and not args.no_diarization
        result = transcribe_audio_with_diarization(
            args.audio_file,
            args.model,
            args.language,
            args.output,
            show_progress=not args.no_progress,
            hf_token=args.hf_token or os.getenv("HF_TOKEN"),
            enable_diarization=enable_diarization,
            num_speakers=args.num_speakers,
            verbose=args.verbose
        )

        print("\n" + "="*50)
        print("TRANSCRIPCION:")
        print("="*50)

        if enable_diarization:
            current_speaker = None
            for segment in result["segments"]:
                speaker = segment.get("speaker", "SPEAKER_UNKNOWN")
                text = segment["text"].strip()

                if speaker != current_speaker:
                    print(f"\n{speaker}:")
                    current_speaker = speaker

                print(f"{text} ", end="")
        else:
            print(result.get("text", "").strip())

        print("\n")

        if args.verbose:
            print("\n" + "="*50)
            print("INFORMACION DETALLADA:")
            print("="*50)
            print(f"Idioma detectado: {result.get('language', 'desconocido')}")
            print(f"Dispositivo: {result.get('device', 'desconocido')}")
            print(f"fp16: {result.get('fp16', False)}")

            total_segments = len(result["segments"])
            total_duration = result["segments"][-1]["end"] if result["segments"] else 0

            print(f"Total de segmentos: {total_segments}")
            print(f"Duracion total: {format_seconds(total_duration)}")

            if result.get("diarization"):
                print(f"Speakers detectados: {len(result['diarization']['speakers'])}")
                for speaker, stats in result["diarization"]["speaker_stats"].items():
                    percentage = (stats["total_time"] / total_duration) * 100 if total_duration else 0
                    print(f"  - {speaker}: {format_seconds(stats['total_time'])} ({percentage:.1f}%)")

            print("\nSegmentos detallados:")
            for i, segment in enumerate(result["segments"], 1):
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
                speaker = segment.get("speaker")
                confidence = segment.get("overlap_confidence")

                if speaker:
                    print(f"{i:2d}. [{start:6.2f}s - {end:6.2f}s] {speaker} (conf: {confidence:.2f}): {text}")
                else:
                    print(f"{i:2d}. [{start:6.2f}s - {end:6.2f}s]: {text}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTranscripcion interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
