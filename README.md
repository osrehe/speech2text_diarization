# Speech2Text

Speech2Text es una solucion local para convertir archivos de audio o video en texto mediante Whisper. Permite procesar reuniones, entrevistas, llamadas, clases, grabaciones de trabajo u otros registros de voz sin depender de una interfaz web externa para la transcripcion.

La solucion sirve para:

- Transcribir audio a texto en español u otros idiomas soportados por Whisper.
- Generar archivos de salida en formato `.txt`, `_detailed.txt` y `.json`.
- Identificar hablantes mediante diarizacion cuando se configura un token de Hugging Face.
- Revisar el avance del proceso por etapas desde una interfaz grafica.
- Ejecutar el mismo motor desde consola para automatizar o integrar procesos.

La aplicacion mantiene los modelos Whisper en la carpeta local `models/` y guarda los resultados en `output/`. Para diarizacion, convertir automaticamente formatos no nativos a WAV 16 kHz mono antes de invocar pyannote.

La solucion se centra en dos programas principales:

- `transcriber_gui.py`: interfaz grafica para usuarios finales.
- `transcriber_dia.py`: motor de transcripcion/diarizacion usable por consola.

## Requisitos

- Python 3.10 o superior.
- `ffmpeg` instalado y disponible en el `PATH`.
- Dependencias Python instaladas dentro de un ambiente virtual.

## mbiente Virtual

Crear el ambiente virtual:

```powershell 
python -m venv whisper_env
```

Activar el ambiente virtual en PowerShell:

```powershell
.\whisper_env\Scripts\Activate.ps1
```

Activar el ambiente virtual en CMD:

```cmd
whisper_env\Scripts\activate.bat
```

Si PowerShell bloquea la activacion por politicas de ejecucion, ejecutar:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Instalar las dependencias dentro del ambiente virtual:

```powershell
python -m pip install -r requirements.txt
```

## Uso con Interfaz Grafica

Ejecutar (con el ambiente virtual activado):

```powershell
python transcriber_gui.py
```

### Inicio rapido con `Iniciar_GUI.bat`

En Windows puedes hacer **doble clic** en `Iniciar_GUI.bat` para abrir la GUI directamente. El archivo usa el Python del entorno virtual `whisper_env` (no el Python global), de modo que las dependencias como `faster-whisper` esten disponibles.

Esto evita el error `No se pudo importar faster-whisper`, que ocurre cuando la GUI se lanza con un Python que no tiene instaladas las dependencias (por ejemplo, al hacer doble clic en `transcriber_gui.py`, que Windows abre con el Python global).

Si el entorno `whisper_env` no existe, el `.bat` muestra las instrucciones para crearlo. Opcionalmente puedes crear un acceso directo del `.bat` en el escritorio.

La GUI permite:

- Seleccionar un archivo de audio o video, o arrastrarlo y soltarlo sobre la ventana (requiere `tkinterdnd2`).
- Elegir modelo Whisper: `tiny`, `base`, `small`, `medium`, `large`.
- Definir idioma, por ejemplo `es`.
- Procesar audio con una barra de avance por etapa y estimacion de tiempo restante (ETA).
- Cancelar un proceso en curso con el boton `Cancelar`.
- Ver la transcripcion en pantalla.
- Copiar la transcripcion al portapapeles.
- Guardar automaticamente los resultados en la carpeta `output/` y abrirla con `Abrir carpeta`.

Guardar los archivos de salida con el nombre del audio. Si ya existe una salida con el mismo nombre, agregar un sufijo numerico:

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

1. Crear un token classic en Hugging Face:
   - Entrar a `Settings > Tokens`.
   - Seleccionar `New token`.
   - Usar permiso `Read`.
   - Habilitar acceso a repositorios publicos gated si la opcion esta disponible.

2. Aceptar las licencias/condiciones de los modelos pyannote:
   - https://huggingface.co/pyannote/speaker-diarization
   - https://huggingface.co/pyannote/segmentation
   - https://huggingface.co/pyannote/embedding

3. Guardar el token como variable de entorno.

En Windows:

```powershell
setx HF_TOKEN "hf_xxxxxxxxxxxxxxxxx"
```

En Bash:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

Para activar la diarizacion en la GUI:

1. Abrir `Opciones > Carga de HF token`.
2. Ingresar el token de Hugging Face.
3. Activar la opcion `Diarizacion`.
4. Opcionalmente indicar `Nro. hablantes`.

El token se guarda localmente en `.env`:

```env
HF_TOKEN=tu_token
```

`.env` esta ignorado por Git y no debe compartirse.

Pyannote puede requerir aceptar condiciones en Hugging Face para los modelos:

- `pyannote/speaker-diarization-3.1`
- `pyannote/speaker-diarization`
- `pyannote/segmentation-3.0`

El programa intenta cargar primero `pyannote/speaker-diarization-3.1` y, si no esta disponible, usar como fallback `pyannote/speaker-diarization`.

## Uso por Consola

Ejecutar transcripcion simple:

```powershell
python transcriber_dia.py audio.m4a -m base -l es -o output/audio.txt
```

Ejecutar con diarizacion:

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

La transcripcion usa **faster-whisper** (CTranslate2), que es notablemente mas rapido que Whisper de OpenAI en CPU y consume menos memoria, manteniendo la misma precision. La transcripcion aplica filtrado por VAD (deteccion de voz) para saltar silencios y acelerar el procesamiento.

Los modelos (formato CTranslate2) se descargan y cachean en la carpeta local:

```text
models/
```

Si el modelo no existe, faster-whisper lo descarga automaticamente. Si ya existe, lo reutiliza. En CPU se usa cuantizacion `int8`; con GPU CUDA disponible se usa `float16` de forma automatica.

## Salidas

Al indicar un archivo de salida, generar:

- `.txt`: transcripcion simple.
- `_detailed.txt`: transcripcion con timestamps y hablantes si aplica.
- `.json`: resultado completo.

## Pruebas

Ejecutar la suite de pruebas unitarias:

```powershell
python -m unittest discover -s tests
```

Las pruebas cubren funciones auxiliares del motor, eventos de progreso, asignacion de hablantes, escritura de salidas y logica base de la interfaz grafica sin cargar modelos Whisper ni ejecutar pyannote.

## Notas

- Considerar que la primera ejecucion de un modelo puede tardar porque descarga pesos.
- Considerar que la diarizacion puede tardar bastante mas que la transcripcion.
- Convertir temporalmente audios `.m4a`, `.mp3` o `.mp4` a WAV 16 kHz mono para pyannote.
- Usar GPU y `fp16` automaticamente si hay GPU CUDA disponible.

## Licencia

Distribuir este proyecto bajo licencia MIT.

## Instalacion Opcional con Conda

La instalacion principal recomendada es con `venv`, pero Conda puede ser util si necesitas controlar mejor versiones de `torch`, `ffmpeg`, CUDA o dependencias cientificas.

Si existe un archivo `environment.yml`, crear el ambiente con:

```powershell
conda env create -f environment.yml
conda activate whisper_env
```

En Windows, si no se tiene modo desarrollador activado, Hugging Face y SpeechBrain pueden mostrar advertencias por uso de symlinks. Para evitarlo, configurar:

```powershell
setx HF_HUB_DISABLE_SYMLINKS 1
setx SPEECHBRAIN_DOWNLOAD_STRATEGY copy
```

Despues de ejecutar `setx`, cerrar y volver a abrir la terminal para que las variables queden disponibles.

### GPU NVIDIA

Si se dispone de GPU NVIDIA compatible, instalar PyTorch con CUDA para acelerar Whisper y parte del procesamiento de diarizacion.

En ese caso, ajustar el bloque de `torch` en `environment.yml` para usar CUDA en lugar de CPU. La configuracion exacta depende de la version de CUDA y del soporte de PyTorch instalado.

Si no se dispone de GPU NVIDIA o no se requiere configurar CUDA, usar la version CPU. Funcionara, pero sera mas lenta.
