"""Settings management for LocalTranslate."""

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "output_folder": str(Path.home() / "Documents" / "Transcriptions"),
    "whisper_model": "base",
    "file_format": "md",
    "include_timestamps": True,
    "enable_diarization": False,
    "huggingface_token": "",
    "microphone": "default",
}

CONFIG_DIR = Path.home() / ".localtranslate"
CONFIG_FILE = CONFIG_DIR / "config.json"


class Config:
    """Manages application configuration."""

    def __init__(self):
        self._config: dict[str, Any] = {}
        self._ensure_config_dir()
        self.load()

    def _ensure_config_dir(self) -> None:
        """Create config directory if it doesn't exist."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        """Load configuration from file, creating defaults if needed."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    self._config = json.load(f)
                # Merge with defaults to ensure all keys exist
                for key, value in DEFAULT_CONFIG.items():
                    if key not in self._config:
                        self._config[key] = value
            except (json.JSONDecodeError, IOError):
                self._config = DEFAULT_CONFIG.copy()
                self.save()
        else:
            self._config = DEFAULT_CONFIG.copy()
            self.save()

    def save(self) -> None:
        """Save current configuration to file."""
        self._ensure_config_dir()
        with open(CONFIG_FILE, "w") as f:
            json.dump(self._config, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value and save."""
        self._config[key] = value
        self.save()

    def update(self, values: dict[str, Any]) -> None:
        """Update multiple configuration values and save."""
        self._config.update(values)
        self.save()

    @property
    def output_folder(self) -> str:
        return self._config.get("output_folder", DEFAULT_CONFIG["output_folder"])

    @property
    def whisper_model(self) -> str:
        return self._config.get("whisper_model", DEFAULT_CONFIG["whisper_model"])

    @property
    def file_format(self) -> str:
        return self._config.get("file_format", DEFAULT_CONFIG["file_format"])

    @property
    def include_timestamps(self) -> bool:
        return self._config.get("include_timestamps", DEFAULT_CONFIG["include_timestamps"])

    @property
    def enable_diarization(self) -> bool:
        return self._config.get("enable_diarization", DEFAULT_CONFIG["enable_diarization"])

    @property
    def huggingface_token(self) -> str:
        return self._config.get("huggingface_token", DEFAULT_CONFIG["huggingface_token"])

    @property
    def microphone(self) -> str:
        return self._config.get("microphone", DEFAULT_CONFIG["microphone"])

    def to_dict(self) -> dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()


# Global config instance
config = Config()
