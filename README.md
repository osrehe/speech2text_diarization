# Speech2Text ✨ (Whisper + Diarización)

Transcribe audios con **OpenAI Whisper** y detecta interlocutores con **pyannote.audio**.\
Funciona en Windows, Linux y macOS.

---

## 1·Instalación rápida

```bash
# clona el repo y entra
conda env create -f environment.yml
conda activate whisper_env

# (Windows sin modo desarrollador)
setx HF_HUB_DISABLE_SYMLINKS 1
setx SPEECHBRAIN_DOWNLOAD_STRATEGY copy
```

> **GPU NVIDIA** – editar `environment.yml` y cambia el bloque *CPU* por el de *CUDA*.

---

## 2·Configurar Hugging Face

1. Crear un **token classic** → Settings› Tokens → *New token* → permiso **Read** + *Access public gated repos*.
2. Aceptar las licencias en:\
   •[https://huggingface.co/pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)\
   •[https://huggingface.co/pyannote/segmentation](https://huggingface.co/pyannote/segmentation)\
   •[https://huggingface.co/pyannote/embedding](https://huggingface.co/pyannote/embedding)
3. Guardar el token:

```powershell
setx HF_TOKEN "hf_xxxxxxxxxxxxxxxxx"   # Windows
# export HF_TOKEN=hf_xxx                # Bash
```

---

## 3·Uso básico

```bash
# Transcripción simple
python transcriptor.py ejemplo.m4a -m base

# Transcripción + diarización
python transcriptor.py ejemplo.m4a --diarize

# Forzar 2 speakers y exportar a txt
python transcriptor.py ejemplo.m4a --diarize --num-speakers 2 -o salida.txt
```

### Flags más útiles

| Flag               | Descripción                        |
| ------------------ | ---------------------------------- |
| `-m, --model`      | Modelo Whisper (tiny … large)      |
| `--diarize`        | Activa identificación de hablantes |
| `--num-speakers N` | Fija nº exacto de speakers         |
| `--hf-token`       | Pasa token si no usas `HF_TOKEN`   |
| `-o, --output`     | Guarda transcripción en archivo    |
| `-v, --verbose`    | Muestra métricas extras            |

---

## 4·Solución de problemas

| Mensaje                                  | Arreglo rápido                                                             |
| ---------------------------------------- | -------------------------------------------------------------------------- |
| `LibsndfileError`                        | FFmpeg no reconoce formato -> el script lo convierte a WAV automáticamente. |
| `403 Forbidden`                          | Token sin permiso o faltó aceptar licencias.                               |
| Warnings de versión `pyannote.audio 0.x` | Ignora: el modelo funciona con 3.x (warnings informativos).                |

---

## 5·Licencia

Código MIT.\
Modelos de terceros mantienen sus propias licencias.

---

¡Felices transcripciones! 🚀

