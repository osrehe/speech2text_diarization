# Speech2Text

Aplicacion local para transcribir audio con Whisper y, opcionalmente, realizar diarizacion de hablantes con pyannote.

La solucion se centra en dos programas principales:

- `transcriber_gui.py`: interfaz grafica para usuarios finales.
- `transcriber_dia.py`: motor de transcripcion/diarizacion usable por consola.

## Requisitos

- Python 3.10 o superior.
- `ffmpeg` instalado y disponible en el `PATH`.
- Dependencias Python instaladas dentro de un ambiente virtual.

## Ambiente Virtual

Crear el ambiente virtual:

```powershell
python -m venv whisper_env
```

Activarlo en PowerShell:

```powershell
.\whisper_env\Scripts\Activate.ps1
```

Activarlo en CMD:

```cmd
whisper_env\Scripts\activate.bat
```

Si PowerShell bloquea la activacion por politicas de ejecucion, ejecuta:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Luego instala las dependencias dentro del ambiente virtual:

```powershell
python -m pip install -r requirements.txt
```

## Uso con Interfaz Grafica

Ejecuta:

```powershell
python transcriber_gui.py
```

La GUI permite:

- Seleccionar un archivo de audio o video.
- Elegir modelo Whisper: `tiny`, `base`, `small`, `medium`, `large`.
- Definir idioma, por ejemplo `es`.
- Procesar audio con una barra de avance por etapa.
- Ver la transcripcion en pantalla.
- Copiar la transcripcion al portapapeles.
- Guardar automaticamente los resultados en la carpeta `output/`.

Los archivos de salida se guardan con el nombre del audio. Si ya existe una salida con el mismo nombre, se agrega un sufijo numerico:

```text
output/audio.txt
output/audio_detailed.txt
output/audio.json

output/audio_1.txt
output/audio_1_detailed.txt
output/audio_1.json
```

## Diarizacion

La diarizacion identifica hablantes usando pyannote.

### Configurar Hugging Face

1. Crea un token classic en Hugging Face:
   - Entra a `Settings > Tokens`.
   - Selecciona `New token`.
   - Usa permiso `Read`.
   - Habilita acceso a repositorios publicos gated si la opcion esta disponible.

2. Acepta las licencias/condiciones de los modelos pyannote:
   - https://huggingface.co/pyannote/speaker-diarization
   - https://huggingface.co/pyannote/segmentation
   - https://huggingface.co/pyannote/embedding

3. Guarda el token como variable de entorno.

En Windows:

```powershell
setx HF_TOKEN "hf_xxxxxxxxxxxxxxxxx"
```

En Bash:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

Para activarla en la GUI:

1. Abre `Opciones > Carga de HF token`.
2. Ingresa tu token de Hugging Face.
3. Activa la opcion `Diarizacion`.
4. Opcionalmente indica `Nro. hablantes`.

El token se guarda localmente en `.env`:

```env
HF_TOKEN=tu_token
```

`.env` esta ignorado por Git y no debe compartirse.

Pyannote puede requerir aceptar condiciones en Hugging Face para los modelos:

- `pyannote/speaker-diarization-3.1`
- `pyannote/speaker-diarization`
- `pyannote/segmentation-3.0`

El programa intenta primero `pyannote/speaker-diarization-3.1` y, si no esta disponible, usa como fallback `pyannote/speaker-diarization`.

## Uso por Consola

Transcripcion simple:

```powershell
python transcriber_dia.py audio.m4a -m base -l es -o output/audio.txt
```

Con diarizacion:

```powershell
python transcriber_dia.py audio.m4a -m base -l es -o output/audio.txt --diarize --hf-token TU_TOKEN --num-speakers 2
```

Opciones principales:

```text
-m, --model          Modelo Whisper: tiny, base, small, medium, large
-l, --language       Codigo de idioma, por ejemplo es
-o, --output         Archivo de salida base
--diarize            Activa diarizacion de hablantes
--hf-token           Token de Hugging Face
--num-speakers       Numero esperado de hablantes
--no-progress        Desactiva barras de progreso internas
-v, --verbose        Muestra informacion detallada
```

## Modelos Whisper

Los modelos Whisper se descargan y cargan desde la carpeta local:

```text
models/
```

Si el modelo no existe, se descarga automaticamente. Si ya existe, se reutiliza.

## Salidas

Al indicar un archivo de salida, se generan:

- `.txt`: transcripcion simple.
- `_detailed.txt`: transcripcion con timestamps y hablantes si aplica.
- `.json`: resultado completo.

## Notas

- La primera ejecucion de un modelo puede tardar porque descarga pesos.
- La diarizacion puede tardar bastante mas que la transcripcion.
- Para audios `.m4a`, `.mp3` o `.mp4`, el programa convierte temporalmente a WAV 16 kHz mono para pyannote.
- Si hay GPU CUDA disponible, Whisper usa GPU y `fp16` automaticamente.

## Licencia

Este proyecto se distribuye bajo licencia MIT.

## Instalacion Opcional con Conda

La instalacion principal recomendada es con `venv`, pero Conda puede ser util si necesitas controlar mejor versiones de `torch`, `ffmpeg`, CUDA o dependencias cientificas.

Si existe un archivo `environment.yml`, puedes crear el ambiente con:

```powershell
conda env create -f environment.yml
conda activate whisper_env
```

En Windows, si no tienes modo desarrollador activado, Hugging Face y SpeechBrain pueden mostrar advertencias por uso de symlinks. Para evitarlo, configura:

```powershell
setx HF_HUB_DISABLE_SYMLINKS 1
setx SPEECHBRAIN_DOWNLOAD_STRATEGY copy
```

Despues de ejecutar `setx`, cierra y vuelve a abrir la terminal para que las variables queden disponibles.

### GPU NVIDIA

Si tienes GPU NVIDIA compatible, conviene instalar PyTorch con CUDA para acelerar Whisper y parte del procesamiento de diarizacion.

En ese caso, ajusta el bloque de `torch` en `environment.yml` para usar CUDA en lugar de CPU. La configuracion exacta depende de tu version de CUDA y del soporte de PyTorch instalado.

Si no tienes GPU NVIDIA o no quieres configurar CUDA, usa la version CPU. Funcionara, pero sera mas lenta.