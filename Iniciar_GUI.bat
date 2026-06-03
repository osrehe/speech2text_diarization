@echo off

setlocal
cd /d "%~dp0"

set "PYW=%~dp0whisper_env\Scripts\pythonw.exe"
set "PY=%~dp0whisper_env\Scripts\python.exe"

if not exist "%PY%" (
    echo No se encontro el entorno virtual en: %~dp0whisper_env
    echo.
    echo Crea el entorno e instala las dependencias:
    echo     python -m venv whisper_env
    echo     whisper_env\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM pythonw.exe lanza la GUI sin abrir una ventana de consola.
start "Speech2Text" "%PYW%" "%~dp0transcriber_gui.py"

endlocal
