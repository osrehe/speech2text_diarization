import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict, Optional

ENV_PATH = Path(__file__).resolve().parent / ".env"
SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac", ".mp4"}


class TranscriberGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Speech2Text - Transcriptor")
        self.root.geometry("1040x800")
        self.root.minsize(900, 680)

        self.events: queue.Queue[Dict[str, Any]] = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.running = False
        self.current_percent = 0
        self.current_stage_index = 0
        self.current_stage_name = ""
        self.last_progress_at = 0.0
        self.last_result_text = ""
        self.log_attached = False

        self.audio_path = tk.StringVar()
        self.model = tk.StringVar(value="base")
        self.language = tk.StringVar(value="es")
        self.enable_diarization = tk.BooleanVar(value=False)
        self.hf_token = tk.StringVar(value=self._load_token_from_env())
        self.num_speakers = tk.StringVar()
        self.show_log = tk.BooleanVar(value=False)

        self._configure_style()
        self._build_menu()
        self._build_ui()
        self._sync_diarization_state()
        self.root.after(100, self._process_events)

    def _configure_style(self) -> None:
        self.colors = {
            "bg": "#f5f7fb",
            "card": "#ffffff",
            "text": "#172033",
            "muted": "#667085",
            "border": "#d9e2ec",
            "primary": "#2563eb",
            "primary_hover": "#1d4ed8",
            "success": "#16a34a",
            "input": "#ffffff",
        }

        self.root.configure(bg=self.colors["bg"])
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

        default_font = ("Segoe UI", 10)
        title_font = ("Segoe UI", 20, "bold")
        subtitle_font = ("Segoe UI", 10)
        section_font = ("Segoe UI", 10, "bold")

        self.style.configure(".", font=default_font, background=self.colors["bg"], foreground=self.colors["text"])
        self.style.configure("App.TFrame", background=self.colors["bg"])
        self.style.configure("Header.TFrame", background=self.colors["bg"])
        self.style.configure("Card.TFrame", background=self.colors["card"], relief="flat")
        self.style.configure("Title.TLabel", font=title_font, background=self.colors["bg"], foreground=self.colors["text"])
        self.style.configure("Subtitle.TLabel", font=subtitle_font, background=self.colors["bg"], foreground=self.colors["muted"])
        self.style.configure("Muted.TLabel", background=self.colors["card"], foreground=self.colors["muted"])
        self.style.configure("Card.TLabel", background=self.colors["card"], foreground=self.colors["text"])
        self.style.configure("Stage.TLabel", font=("Segoe UI", 11, "bold"), background=self.colors["card"], foreground=self.colors["text"])
        self.style.configure(
            "Card.TLabelframe",
            background=self.colors["card"],
            bordercolor=self.colors["border"],
            relief="solid",
        )
        self.style.configure(
            "Card.TLabelframe.Label",
            font=section_font,
            background=self.colors["card"],
            foreground=self.colors["text"],
        )
        self.style.configure("TEntry", fieldbackground=self.colors["input"], bordercolor=self.colors["border"], padding=6)
        self.style.configure("TCombobox", fieldbackground=self.colors["input"], bordercolor=self.colors["border"], padding=6)
        self.style.configure("TCheckbutton", background=self.colors["card"], foreground=self.colors["text"])
        self.style.configure(
            "Primary.TButton",
            background=self.colors["primary"],
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            padding=(16, 8),
        )
        self.style.map("Primary.TButton", background=[("active", self.colors["primary_hover"]), ("disabled", "#9db7f5")])
        self.style.configure("Secondary.TButton", background="#eef2f7", foreground=self.colors["text"], borderwidth=0, padding=(14, 8))
        self.style.map("Secondary.TButton", background=[("active", "#e2e8f0")])
        self.style.configure("Success.TButton", background=self.colors["success"], foreground="#ffffff", borderwidth=0, padding=(14, 8))
        self.style.map("Success.TButton", background=[("active", "#15803d"), ("disabled", "#a7d7b5")])
        self.style.configure(
            "Modern.Horizontal.TProgressbar",
            troughcolor="#e5e7eb",
            background=self.colors["primary"],
            bordercolor="#e5e7eb",
            lightcolor=self.colors["primary"],
            darkcolor=self.colors["primary"],
            thickness=12,
        )

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Abrir audio", command=self._select_audio)
        file_menu.add_command(label="Guardar salida", command=self._save_visible_result)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="Archivo", menu=file_menu)

        options_menu = tk.Menu(menubar, tearoff=0)
        options_menu.add_command(label="Carga de HF token", command=self._load_hf_token)
        options_menu.add_checkbutton(label="Log", variable=self.show_log, command=self._toggle_log)
        menubar.add_cascade(label="Opciones", menu=options_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Mostrar comandos", command=self._show_commands)
        help_menu.add_command(label="Acerca de", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _load_token_from_env(self) -> str:
        token = os.getenv("HF_TOKEN", "")
        if ENV_PATH.exists():
            for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue

                key, value = stripped.split("=", 1)
                if key.strip() == "HF_TOKEN":
                    token = value.strip().strip('"').strip("'")
                    break

        if token:
            os.environ["HF_TOKEN"] = token
        return token

    def _save_token_to_env(self, token: str) -> None:
        lines = []
        found = False
        if ENV_PATH.exists():
            lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

        next_lines = []
        for line in lines:
            if line.strip().startswith("HF_TOKEN="):
                if token:
                    next_lines.append(f"HF_TOKEN={token}")
                found = True
            else:
                next_lines.append(line)

        if token and not found:
            next_lines.append(f"HF_TOKEN={token}")

        ENV_PATH.write_text("\n".join(next_lines).strip() + ("\n" if next_lines else ""), encoding="utf-8")

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=20, style="App.TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        options = ttk.LabelFrame(main, text="Opciones de procesamiento", padding=16, style="Card.TLabelframe")
        options.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(options, text="Audio", style="Card.TLabel").grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=6)
        ttk.Entry(options, textvariable=self.audio_path).grid(row=0, column=1, columnspan=4, sticky=tk.EW, pady=4)
        ttk.Button(
            options,
            text="Seleccionar",
            command=self._select_audio,
            style="Secondary.TButton",
        ).grid(row=0, column=5, padx=(10, 0), pady=4)

        controls_row = ttk.Frame(options, style="Card.TFrame")
        controls_row.grid(row=1, column=0, columnspan=6, sticky=tk.EW, pady=(10, 4))

        self.output_hint = ttk.Label(
            controls_row,
            text="Salida: output/<nombre_audio>.txt",
            style="Muted.TLabel",
            anchor=tk.E,
        )
        self.output_hint.pack(side=tk.RIGHT, padx=(16, 0))

        controls_left = ttk.Frame(controls_row, style="Card.TFrame")
        controls_left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(controls_left, text="Modelo", style="Card.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Combobox(
            controls_left,
            textvariable=self.model,
            values=("tiny", "base", "small", "medium", "large"),
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT, padx=(0, 22))

        ttk.Label(controls_left, text="Idioma", style="Card.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(controls_left, textvariable=self.language, width=8).pack(side=tk.LEFT, padx=(0, 22))

        ttk.Label(controls_left, text="Nro. hablantes", style="Card.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(controls_left, textvariable=self.num_speakers, width=8).pack(side=tk.LEFT, padx=(0, 22))

        self.diarization_check = ttk.Checkbutton(
            controls_left,
            text="Diarizacion",
            variable=self.enable_diarization,
        )
        self.diarization_check.pack(side=tk.LEFT)

        options.columnconfigure(1, weight=1)
        options.columnconfigure(2, weight=1)
        options.columnconfigure(4, weight=1)

        actions = ttk.Frame(main, style="App.TFrame")
        actions.pack(fill=tk.X, pady=(0, 12))

        self.start_button = ttk.Button(actions, text="Procesar audio", command=self._start_processing, style="Primary.TButton")
        self.start_button.pack(side=tk.LEFT)
        ttk.Button(actions, text="Limpiar", command=self._clear_output, style="Secondary.TButton").pack(side=tk.LEFT, padx=(8, 0))
        self.copy_button = ttk.Button(
            actions,
            text="Copiar transcripcion",
            command=self._copy_result,
            state=tk.DISABLED,
            style="Success.TButton",
        )
        self.copy_button.pack(side=tk.LEFT, padx=(8, 0))

        progress_frame = ttk.LabelFrame(main, text="Avance del proceso", padding=16, style="Card.TLabelframe")
        progress_frame.pack(fill=tk.X, pady=(0, 12))

        self.stage_label = ttk.Label(progress_frame, text="Esperando audio...", style="Stage.TLabel")
        self.stage_label.pack(anchor=tk.W)

        progress_row = ttk.Frame(progress_frame, style="Card.TFrame")
        progress_row.pack(fill=tk.X, pady=(10, 0))

        self.progress = ttk.Progressbar(
            progress_row,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=100,
            style="Modern.Horizontal.TProgressbar",
        )
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.percent_label = ttk.Label(progress_row, text="0%", width=6, anchor=tk.E, style="Muted.TLabel")
        self.percent_label.pack(side=tk.LEFT, padx=(8, 0))

        panes = ttk.PanedWindow(main, orient=tk.VERTICAL)
        panes.pack(fill=tk.BOTH, expand=True)

        self.panes = panes

        self.log_frame = ttk.LabelFrame(panes, text="Log de avance", padding=12, style="Card.TLabelframe")
        self.log_text = ScrolledText(self.log_frame, height=9, wrap=tk.WORD)
        self._style_text_widget(self.log_text)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.result_frame = ttk.LabelFrame(panes, text="Resultado de la transcripcion", padding=12, style="Card.TLabelframe")
        self.result_text = ScrolledText(self.result_frame, height=14, wrap=tk.WORD)
        self._style_text_widget(self.result_text)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        panes.add(self.result_frame, weight=2)

    def _style_text_widget(self, widget: ScrolledText) -> None:
        widget.configure(
            background="#fbfdff",
            foreground=self.colors["text"],
            insertbackground=self.colors["text"],
            selectbackground="#bfdbfe",
            selectforeground=self.colors["text"],
            relief=tk.FLAT,
            borderwidth=0,
            padx=12,
            pady=10,
            font=("Segoe UI", 10),
        )

    def _select_audio(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_FORMATS))
        path = filedialog.askopenfilename(
            title="Seleccionar audio",
            filetypes=(("Audio y video soportado", patterns), ("Todos los archivos", "*.*")),
        )
        if not path:
            return

        self.audio_path.set(path)
        output_path = self._build_output_path(path)
        self.output_hint.configure(text=f"Salida: {output_path.parent.name}/{output_path.name}")

    def _build_output_path(self, audio_path: str) -> Path:
        output_dir = Path(__file__).resolve().parent / "output"
        output_dir.mkdir(exist_ok=True)

        stem = Path(audio_path).stem
        index = 0
        while True:
            suffix = "" if index == 0 else f"_{index}"
            candidate = output_dir / f"{stem}{suffix}.txt"
            detailed = output_dir / f"{stem}{suffix}_detailed.txt"
            json_file = output_dir / f"{stem}{suffix}.json"
            if not candidate.exists() and not detailed.exists() and not json_file.exists():
                return candidate
            index += 1

    def _load_hf_token(self) -> None:
        token = simpledialog.askstring(
            "Carga de HF token",
            "Ingresa tu token de Hugging Face para activar diarizacion:",
            show="*",
            parent=self.root,
        )
        if token is None:
            return

        token = token.strip()
        self.hf_token.set(token)
        self._save_token_to_env(token)
        if token:
            os.environ["HF_TOKEN"] = token
        else:
            os.environ.pop("HF_TOKEN", None)
        self._sync_diarization_state()

        if self.hf_token.get():
            messagebox.showinfo("HF Token", "Token cargado. La diarizacion esta disponible.")
        else:
            messagebox.showinfo("HF Token", "Token vacio. La diarizacion quedo desactivada.")

    def _sync_diarization_state(self) -> None:
        has_token = bool(self.hf_token.get().strip())
        state = tk.NORMAL if has_token else tk.DISABLED
        self.diarization_check.configure(state=state)
        if not has_token:
            self.enable_diarization.set(False)

    def _toggle_log(self) -> None:
        if self.show_log.get() and not self.log_attached:
            self.panes.forget(self.result_frame)
            self.panes.add(self.log_frame, weight=1)
            self.panes.add(self.result_frame, weight=2)
            self.log_attached = True
        elif not self.show_log.get() and self.log_attached:
            self.panes.forget(self.log_frame)
            self.log_attached = False

    def _save_visible_result(self) -> None:
        text = self.result_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Sin resultado", "No hay transcripcion para guardar.")
            return

        path = filedialog.asksaveasfilename(
            title="Guardar salida",
            defaultextension=".txt",
            filetypes=(("Texto", "*.txt"), ("Todos los archivos", "*.*")),
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
        messagebox.showinfo("Salida guardada", f"Archivo guardado en:\n{path}")

    def _show_commands(self) -> None:
        messagebox.showinfo(
            "Comandos",
            "GUI:\n"
            "python transcriber_gui.py\n\n"
            "Consola:\n"
            "python transcriber_dia.py audio.m4a -m base -l es -o output/audio.txt\n"
            "python transcriber_dia.py audio.m4a -m base -l es --diarize --hf-token TU_TOKEN --num-speakers 2",
        )

    def _show_about(self) -> None:
        messagebox.showinfo(
            "Acerca de",
            "Speech2Text\n"
            "Licklider SpA\n"
            "Version 1.0",
        )

    def _start_processing(self) -> None:
        audio = self.audio_path.get().strip()
        if not audio:
            messagebox.showwarning("Falta audio", "Selecciona un archivo de audio antes de procesar.")
            return

        if self.enable_diarization.get() and not self.hf_token.get().strip():
            messagebox.showwarning(
                "Falta HF Token",
                "Para usar diarizacion debes ingresar --hf-token o configurar HF_TOKEN.",
            )
            return

        num_speakers = self._parse_num_speakers()
        if num_speakers is None and self.num_speakers.get().strip():
            messagebox.showwarning("Nro. hablantes invalido", "Ingresa un numero entero valido para hablantes.")
            return

        self._clear_output()
        self.running = True
        self.current_percent = 0
        self.current_stage_index = 0
        self.current_stage_name = ""
        self.start_button.configure(state=tk.DISABLED)
        self.stage_label.configure(text="Iniciando proceso...")
        self._set_progress(0)
        self.last_progress_at = time.time()

        output_path = str(self._build_output_path(audio))
        self.output_hint.configure(text=f"Salida: {Path(output_path).parent.name}/{Path(output_path).name}")
        language = self.language.get().strip() or None
        hf_token = self.hf_token.get().strip() or None

        self.worker = threading.Thread(
            target=self._run_transcription,
            kwargs={
                "audio_file_path": audio,
                "model_size": self.model.get(),
                "language": language,
                "output_file": output_path,
                "show_progress": False,
                "hf_token": hf_token,
                "enable_diarization": self.enable_diarization.get(),
                "num_speakers": num_speakers,
                "verbose": True,
                "progress_callback": self._progress_callback,
            },
            daemon=True,
        )
        self.worker.start()

    def _parse_num_speakers(self) -> Optional[int]:
        value = self.num_speakers.get().strip()
        if not value:
            return None
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None

    def _run_transcription(self, **kwargs: Any) -> None:
        try:
            from transcriber_dia import transcribe_audio_with_diarization

            result = transcribe_audio_with_diarization(**kwargs)
            self.events.put({"type": "result", "result": result})
        except Exception as exc:
            self.events.put({"type": "error", "message": str(exc)})

    def _progress_callback(self, event: Dict[str, Any]) -> None:
        self.events.put({"type": "progress", **event})

    def _process_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break

            event_type = event.get("type")
            if event_type == "progress":
                self._handle_progress(event)
            elif event_type == "result":
                self._handle_result(event["result"])
            elif event_type == "error":
                self._handle_error(event["message"])

        self.root.after(100, self._process_events)

    def _handle_progress(self, event: Dict[str, Any]) -> None:
        stage_name = event.get("stage_name") or "Procesando"
        stage_index = int(event.get("stage_index") or 0)
        total_stages = int(event.get("total_stages") or 0)
        percent = event.get("percent")
        message = event.get("message") or ""

        if stage_index and stage_index != self.current_stage_index:
            self.current_stage_index = stage_index
            self.current_stage_name = stage_name
            self._set_progress(0)
        else:
            self.current_stage_name = stage_name

        if stage_index and total_stages:
            self.stage_label.configure(text=f"{stage_name} ({stage_index}/{total_stages})")
        else:
            self.stage_label.configure(text=stage_name)

        if percent is not None:
            percent = int(percent)
            self._set_progress(percent)
            self.last_progress_at = time.time()

        if message:
            self._append_log(message)

    def _handle_result(self, result: Dict[str, Any]) -> None:
        self.running = False
        self.start_button.configure(state=tk.NORMAL)
        self.stage_label.configure(text="Proceso completado")
        self._set_progress(100)
        self._append_log("Proceso finalizado correctamente.")
        self.result_text.delete("1.0", tk.END)
        self.last_result_text = self._format_result(result)
        self.result_text.insert(tk.END, self.last_result_text)
        self.copy_button.configure(state=tk.NORMAL)

    def _handle_error(self, message: str) -> None:
        self.running = False
        self.start_button.configure(state=tk.NORMAL)
        self.stage_label.configure(text="Proceso detenido por error")
        self._append_log(f"ERROR: {message}")
        messagebox.showerror("Error", message)

    def _set_progress(self, percent: int) -> None:
        percent = max(0, min(100, int(percent)))
        self.current_percent = percent
        self.progress.configure(value=percent)
        self.percent_label.configure(text=f"{percent}%")

    def _append_log(self, message: str) -> None:
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)

    def _copy_result(self) -> None:
        text = self.result_text.get("1.0", tk.END).strip()
        if not text:
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        self._append_log("Transcripcion copiada al portapapeles.")

    def _format_result(self, result: Dict[str, Any]) -> str:
        if result.get("diarization"):
            lines = []
            current_speaker = None
            for segment in result.get("segments", []):
                speaker = segment.get("speaker", "SPEAKER_UNKNOWN")
                text = segment.get("text", "").strip()
                if speaker != current_speaker:
                    if lines:
                        lines.append("")
                    lines.append(f"{speaker}:")
                    current_speaker = speaker
                lines.append(text)
            return "\n".join(lines).strip()

        return result.get("text", "").strip()

    def _clear_output(self) -> None:
        self.log_text.delete("1.0", tk.END)
        self.result_text.delete("1.0", tk.END)
        self.last_result_text = ""
        self.stage_label.configure(text="Esperando audio...")
        self._set_progress(0)
        self.copy_button.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    TranscriberGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
