#!/usr/bin/env python3
"""Settings GUI using tkinter for LocalTranslate."""

import json
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path


CONFIG_FILE = Path.home() / ".localtranslate" / "config.json"


def get_audio_devices():
    """Get list of audio input devices."""
    try:
        import sounddevice as sd
        devices = ["default"]
        for idx, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                devices.append(device["name"])
        return devices
    except Exception:
        return ["default"]


def load_config():
    """Load config from file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_config(config):
    """Save config to file."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


class SettingsWindow:
    """Settings dialog window."""

    def __init__(self):
        self.config = load_config()
        self._vars = {}
        self.root = tk.Tk()
        self.root.title("LocalTranslate Settings")
        self.root.geometry("500x480")
        self.root.resizable(False, False)

        self._create_widgets()
        self._load_values()
        self._center_window()

    def _center_window(self):
        """Center window on screen."""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.root.winfo_width()) // 2
        y = (self.root.winfo_screenheight() - self.root.winfo_height()) // 2
        self.root.geometry(f"+{x}+{y}")

    def _create_widgets(self):
        """Create all widgets."""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        row = 0

        # Output folder
        ttk.Label(main_frame, text="Output Folder:").grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=row, column=0, sticky=tk.EW, pady=(0, 15))
        folder_frame.columnconfigure(0, weight=1)

        self._vars["output_folder"] = tk.StringVar()
        folder_entry = ttk.Entry(folder_frame, textvariable=self._vars["output_folder"], width=45)
        folder_entry.grid(row=0, column=0, sticky=tk.EW, padx=(0, 10))

        browse_btn = ttk.Button(folder_frame, text="Browse...", command=self._browse_folder)
        browse_btn.grid(row=0, column=1)
        row += 1

        # Whisper model
        ttk.Label(main_frame, text="Whisper Model:").grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        self._vars["whisper_model"] = tk.StringVar()
        model_combo = ttk.Combobox(
            main_frame,
            textvariable=self._vars["whisper_model"],
            values=["tiny", "base", "small", "medium", "large"],
            state="readonly",
            width=20,
        )
        model_combo.grid(row=row, column=0, sticky=tk.W, pady=(0, 15))
        row += 1

        # Microphone
        ttk.Label(main_frame, text="Microphone:").grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        self._vars["microphone"] = tk.StringVar()
        devices = get_audio_devices()
        mic_combo = ttk.Combobox(
            main_frame,
            textvariable=self._vars["microphone"],
            values=devices,
            state="readonly",
            width=40,
        )
        mic_combo.grid(row=row, column=0, sticky=tk.W, pady=(0, 15))
        row += 1

        # File format
        ttk.Label(main_frame, text="File Format:").grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        format_frame = ttk.Frame(main_frame)
        format_frame.grid(row=row, column=0, sticky=tk.W, pady=(0, 15))

        self._vars["file_format"] = tk.StringVar()
        ttk.Radiobutton(
            format_frame,
            text="Markdown (.md)",
            variable=self._vars["file_format"],
            value="md",
        ).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(
            format_frame,
            text="Plain Text (.txt)",
            variable=self._vars["file_format"],
            value="txt",
        ).pack(side=tk.LEFT)
        row += 1

        # Checkboxes
        self._vars["include_timestamps"] = tk.BooleanVar()
        ttk.Checkbutton(
            main_frame,
            text="Include timestamps",
            variable=self._vars["include_timestamps"],
        ).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1

        self._vars["enable_diarization"] = tk.BooleanVar()
        ttk.Checkbutton(
            main_frame,
            text="Enable speaker diarization (requires HuggingFace token)",
            variable=self._vars["enable_diarization"],
            command=self._on_diarization_toggle,
        ).grid(row=row, column=0, sticky=tk.W, pady=(0, 10))
        row += 1

        # HuggingFace token
        ttk.Label(main_frame, text="HuggingFace Token:").grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        row += 1

        self._vars["huggingface_token"] = tk.StringVar()
        self._token_entry = ttk.Entry(
            main_frame,
            textvariable=self._vars["huggingface_token"],
            width=50,
            show="*",
        )
        self._token_entry.grid(row=row, column=0, sticky=tk.W, pady=(0, 15))
        row += 1

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=row, column=0, sticky=tk.E, pady=(20, 0))

        ttk.Button(btn_frame, text="Cancel", command=self.root.destroy).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Button(btn_frame, text="Save", command=self._on_save).pack(side=tk.LEFT)

        main_frame.columnconfigure(0, weight=1)

    def _load_values(self):
        """Load current config values into widgets."""
        defaults = {
            "output_folder": str(Path.home() / "Documents" / "Transcriptions"),
            "whisper_model": "base",
            "microphone": "default",
            "file_format": "md",
            "include_timestamps": True,
            "enable_diarization": False,
            "huggingface_token": "",
        }

        for key, default in defaults.items():
            value = self.config.get(key, default)
            if key in self._vars:
                self._vars[key].set(value)

        self._on_diarization_toggle()

    def _browse_folder(self):
        """Open folder selection dialog."""
        folder = filedialog.askdirectory(
            initialdir=self._vars["output_folder"].get(),
            title="Select Output Folder",
        )
        if folder:
            self._vars["output_folder"].set(folder)

    def _on_diarization_toggle(self):
        """Handle diarization checkbox toggle."""
        if self._vars["enable_diarization"].get():
            self._token_entry.configure(state="normal")
        else:
            self._token_entry.configure(state="disabled")

    def _on_save(self):
        """Save settings and close."""
        if not self._vars["output_folder"].get():
            messagebox.showerror("Error", "Please select an output folder")
            return

        if self._vars["enable_diarization"].get() and not self._vars["huggingface_token"].get():
            messagebox.showerror("Error", "HuggingFace token is required for diarization")
            return

        new_config = {
            "output_folder": self._vars["output_folder"].get(),
            "whisper_model": self._vars["whisper_model"].get(),
            "microphone": self._vars["microphone"].get(),
            "file_format": self._vars["file_format"].get(),
            "include_timestamps": self._vars["include_timestamps"].get(),
            "enable_diarization": self._vars["enable_diarization"].get(),
            "huggingface_token": self._vars["huggingface_token"].get(),
        }

        save_config(new_config)
        print("Settings saved")
        self.root.destroy()

    def run(self):
        """Run the settings window."""
        self.root.mainloop()


if __name__ == "__main__":
    app = SettingsWindow()
    app.run()
