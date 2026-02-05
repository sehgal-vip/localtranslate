"""Output file handling for LocalTranslate."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class FileManager:
    """Manages transcription output files."""

    def __init__(self, output_folder: str, file_format: str = "md"):
        self.output_folder = Path(output_folder)
        self.file_format = file_format
        self.current_file: Optional[Path] = None
        self._session_timestamp: str = ""
        self._ensure_output_folder()

    def _ensure_output_folder(self) -> None:
        """Create output folder if it doesn't exist."""
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def set_output_folder(self, folder: str) -> None:
        """Update the output folder."""
        self.output_folder = Path(folder)
        self._ensure_output_folder()

    def set_file_format(self, fmt: str) -> None:
        """Update the file format."""
        self.file_format = fmt

    def start_new_session(self) -> Path:
        """Create new transcription file for this session."""
        self._ensure_output_folder()
        self._session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filename = f"transcription_{self._session_timestamp}.{self.file_format}"
        self.current_file = self.output_folder / filename

        return self.current_file

    def _create_header(self) -> str:
        """Create file header based on format."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.file_format == "md":
            return f"# Transcription\n\n**Date:** {timestamp}\n\n"
        else:
            return f"Transcription - {timestamp}\n{'=' * 40}\n\n"

    def write_transcription(
        self,
        text: str,
        duration: float = 0.0,
    ) -> Path:
        """Write the transcription to file."""
        if self.current_file is None:
            self.start_new_session()

        duration_str = self.format_timestamp(duration) if duration > 0 else "N/A"

        with open(self.current_file, "w", encoding="utf-8") as f:
            f.write(self._create_header())
            f.write(f"**Duration:** {duration_str}\n\n")
            if self.file_format == "md":
                f.write("---\n\n")
            else:
                f.write("-" * 40 + "\n\n")
            f.write(text.strip() + "\n")

        return self.current_file

    def get_current_file(self) -> Optional[Path]:
        """Get the path to the current transcription file."""
        return self.current_file

    def close_session(self) -> Optional[Path]:
        """Close the current session and return file path."""
        file_path = self.current_file
        self.current_file = None
        return file_path

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
