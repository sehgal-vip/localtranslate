"""Menu bar interface using rumps for LocalTranslate."""

import os
import subprocess
import threading
from typing import Callable, Optional

import rumps


class MenuBarApp(rumps.App):
    """Menu bar application for LocalTranslate."""

    ICON_IDLE = "🎙"
    ICON_RECORDING = "🔴"
    TITLE_IDLE = "LocalTranslate"
    TITLE_RECORDING = "Recording..."
    TITLE_PROCESSING = "Processing..."

    def __init__(
        self,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_settings: Optional[Callable[[], None]] = None,
        on_quit: Optional[Callable[[], None]] = None,
        output_folder: str = "",
    ):
        super().__init__(self.TITLE_IDLE, quit_button=None)

        self.on_start = on_start
        self.on_stop = on_stop
        self.on_settings = on_settings
        self.on_quit_callback = on_quit
        self.output_folder = output_folder

        self._is_recording = False
        self._setup_menu()

    def _setup_menu(self) -> None:
        """Set up menu items."""
        self.record_button = rumps.MenuItem(
            "Start Recording",
            callback=self._toggle_recording,
        )

        self.open_folder_button = rumps.MenuItem(
            "Open Output Folder",
            callback=self._open_output_folder,
        )

        self.settings_button = rumps.MenuItem(
            "Settings...",
            callback=self._open_settings,
        )

        self._quit_item = rumps.MenuItem(
            "Quit LocalTranslate",
            callback=self._quit_app,
        )

        self.menu = [
            self.record_button,
            None,  # Separator
            self.open_folder_button,
            self.settings_button,
            None,  # Separator
            self._quit_item,
        ]

    def _toggle_recording(self, _) -> None:
        """Toggle recording state."""
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Start recording."""
        self._is_recording = True
        self.title = self.TITLE_RECORDING
        self.record_button.title = "Stop Recording"

        if self.on_start:
            # Run in thread to not block UI
            threading.Thread(target=self.on_start, daemon=True).start()

    def _stop_recording(self) -> None:
        """Stop recording."""
        self._is_recording = False
        self.record_button.title = "Start Recording"
        self.title = self.TITLE_PROCESSING

        if self.on_stop:
            # Run in thread so UI can update
            threading.Thread(target=self._run_stop_callback, daemon=True).start()

    def _run_stop_callback(self) -> None:
        """Run stop callback and reset UI when done."""
        try:
            self.on_stop()
        finally:
            self.title = self.TITLE_IDLE

    def _open_output_folder(self, _) -> None:
        """Open the output folder in Finder."""
        if self.output_folder and os.path.exists(self.output_folder):
            subprocess.run(["open", self.output_folder])
        else:
            rumps.alert(
                title="Output Folder",
                message="Output folder not found. Please check settings.",
            )

    def _open_settings(self, _) -> None:
        """Open settings window."""
        if self.on_settings:
            # Run in separate thread to avoid blocking rumps
            threading.Thread(target=self.on_settings, daemon=True).start()

    def _quit_app(self, _) -> None:
        """Quit the application."""
        if self._is_recording:
            self._stop_recording()

        if self.on_quit_callback:
            self.on_quit_callback()

        rumps.quit_application()

    def set_output_folder(self, folder: str) -> None:
        """Update the output folder path."""
        self.output_folder = folder

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    def set_processing(self, processing: bool) -> None:
        """Set processing state in menu bar."""
        if processing:
            self.title = self.TITLE_PROCESSING
        else:
            self.title = self.TITLE_IDLE

    def update_status(self, message: str) -> None:
        """Update the menu bar title with a status message."""
        if self._is_recording:
            self.title = f"🔴 {message}"
        else:
            self.title = message

    def show_notification(self, title: str, message: str) -> None:
        """Show a notification."""
        try:
            rumps.notification(
                title=title,
                subtitle="",
                message=message,
            )
        except Exception as e:
            # Notification may fail without proper Info.plist
            print(f"[Notification] {title}: {message}")
