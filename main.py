#!/usr/bin/env python3
"""Entry point for LocalTranslate - Real-Time Speech-to-Text App."""

import os
# Fix OpenMP conflict between torch and other libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import subprocess
import sys
import threading
import time
from pathlib import Path

from audio_processor import AudioProcessor
from audio_recorder import AudioRecorder
from config import config
from file_manager import FileManager
from transcriber import Transcriber
from ui import MenuBarApp


class LocalTranslate:
    """Main application class."""

    def __init__(self):
        self.recorder = AudioRecorder(
            device=config.microphone,
            system_device=config.system_audio_device if config.include_system_audio else None,
        )
        self.transcriber = Transcriber(config.whisper_model)
        self.file_manager = FileManager(config.output_folder, config.file_format)
        self.audio_processor = AudioProcessor(sample_rate=16000)

        # Lazy load speaker identifier only if needed
        self._speaker_identifier = None
        self._diarization_available = False

        self._recording_start_time: float = 0

        # Create menu bar app
        self.app = MenuBarApp(
            on_start=self._on_start_recording,
            on_stop=self._on_stop_recording,
            on_settings=self._on_settings,
            on_quit=self._on_quit,
            output_folder=config.output_folder,
        )

    def run(self) -> None:
        """Run the application."""
        print("Starting LocalTranslate...")
        print(f"Output folder: {config.output_folder}")
        print(f"Whisper model: {config.whisper_model}")
        print(f"Diarization: {'enabled' if config.enable_diarization else 'disabled'}")
        print(f"System audio: {'enabled (' + config.system_audio_device + ')' if config.include_system_audio else 'disabled'}")

        # Ensure output folder exists
        Path(config.output_folder).mkdir(parents=True, exist_ok=True)

        # Pre-load model in background
        threading.Thread(target=self._preload_model, daemon=True).start()

        # Start menu bar app (blocks)
        self.app.run()

    def _preload_model(self) -> None:
        """Pre-load the Whisper model in background."""
        try:
            print("Pre-loading Whisper model...")
            self.transcriber.load_model()
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not pre-load model: {e}")

    def _init_diarization(self) -> bool:
        """Initialize diarization if enabled. Returns True if available."""
        if not config.enable_diarization or not config.huggingface_token:
            return False

        try:
            if self._speaker_identifier is None:
                from speaker_identifier import SpeakerIdentifier
                self._speaker_identifier = SpeakerIdentifier(config.huggingface_token)

            self._speaker_identifier.set_token(config.huggingface_token)
            self._speaker_identifier.reset_speakers()

            if self._speaker_identifier.load_pipeline():
                self._diarization_available = True
                return True
            else:
                print("Diarization pipeline failed to load - continuing without it")
                self._diarization_available = False
                return False
        except Exception as e:
            print(f"Diarization init error: {e}")
            self._diarization_available = False
            return False

    def _reload_config(self) -> None:
        """Reload config from file and update components."""
        config.load()

        self.recorder.set_device(config.microphone)
        self.recorder.set_system_device(
            config.system_audio_device if config.include_system_audio else None
        )
        self.file_manager.set_output_folder(config.output_folder)
        self.file_manager.set_file_format(config.file_format)
        self.app.set_output_folder(config.output_folder)

        if config.whisper_model != self.transcriber.get_model_name():
            threading.Thread(
                target=lambda: self.transcriber.change_model(config.whisper_model),
                daemon=True,
            ).start()

    def _on_start_recording(self) -> None:
        """Handle start recording event."""
        print("\n" + "=" * 50)
        print("Starting recording...")

        # Reload config in case settings changed
        self._reload_config()

        # Start new session
        self.file_manager.start_new_session()

        # Start audio recording
        self.recorder.set_device(config.microphone)
        self.recorder.set_system_device(
            config.system_audio_device if config.include_system_audio else None
        )
        try:
            self.recorder.start(include_system_audio=config.include_system_audio)
            self._recording_start_time = time.time()
            print("Recording started...")
            if config.include_system_audio:
                print(f"  System audio: {config.system_audio_device}")
        except Exception as e:
            print(f"Error starting recording: {e}")
            self._safe_notification("Recording Error", f"Could not start recording: {e}")
            return

        self._safe_notification("Recording Started", "Listening for speech...")

    def _on_stop_recording(self) -> None:
        """Handle stop recording event."""
        print("\nStopping recording...")

        # Calculate duration
        duration = time.time() - self._recording_start_time if self._recording_start_time > 0 else 0

        # Stop recorder and get full audio
        full_audio = self.recorder.stop()
        print("Processing...")

        # Process full audio
        if full_audio is not None and len(full_audio) > 0:
            print(f"Transcribing {len(full_audio) / 16000:.1f} seconds of audio...")
            self._process_audio(full_audio, duration)
        else:
            print("No audio recorded")
            self.file_manager.write_transcription("[No audio recorded]", duration)

        # Close session
        file_path = self.file_manager.close_session()

        if file_path:
            self._safe_notification("Saved", file_path.name)
            print(f"\nSaved: {file_path}")
        print("=" * 50)

    def _process_audio(self, audio, duration: float) -> None:
        """Process audio and save transcription."""
        try:
            # Ensure model is loaded
            if not self.transcriber.is_loaded():
                print("Loading Whisper model...")
                self.transcriber.load_model()

            # Pre-process audio for better quality
            print("Optimizing audio...")
            raw_rms = self.audio_processor.compute_rms(audio)
            raw_peak = self.audio_processor.compute_peak(audio)
            print(f"  Raw: RMS={raw_rms:.4f}, Peak={raw_peak:.4f}")

            processed_audio = self.audio_processor.process(audio)

            proc_rms = self.audio_processor.compute_rms(processed_audio)
            proc_peak = self.audio_processor.compute_peak(processed_audio)
            print(f"  Processed: RMS={proc_rms:.4f}, Peak={proc_peak:.4f}")
            print(f"  Length: {len(processed_audio) / 16000:.1f}s")

            # Try speaker diarization if enabled
            # IMPORTANT: Run diarization on PROCESSED audio (same as transcription)
            # to ensure timing alignment between speaker segments and transcribed text
            speaker_segments = []
            if config.enable_diarization:
                try:
                    if self._init_diarization():
                        print("Running speaker diarization...")
                        speaker_segments = self._speaker_identifier.identify_speakers(
                            processed_audio, sample_rate=16000
                        )
                        if speaker_segments:
                            print(f"  Found {len(speaker_segments)} speaker segments:")
                            for seg in speaker_segments[:3]:
                                print(f"    {seg.speaker}: {seg.start_time:.2f}s - {seg.end_time:.2f}s")
                            if len(speaker_segments) > 3:
                                print(f"    ... and {len(speaker_segments) - 3} more")
                        else:
                            print("  WARNING: No speaker segments detected")
                except Exception as e:
                    print(f"  Diarization failed: {e}")
                    import traceback
                    traceback.print_exc()
                    speaker_segments = []

            # Transcribe
            print("Transcribing...")
            if speaker_segments:
                # Get segments with timing for speaker alignment
                text, segments = self.transcriber.transcribe_with_segments(
                    processed_audio, sample_rate=16000
                )
                if text and segments:
                    final_text = self._merge_speakers_with_text(speaker_segments, segments)
                else:
                    final_text = text or "[No speech detected]"
            else:
                # Simple transcription without diarization
                result = self.transcriber.transcribe(processed_audio, sample_rate=16000)
                final_text = result.text if result else "[No speech detected]"

            if final_text and final_text.strip() and final_text != "[No speech detected]":
                self.file_manager.write_transcription(final_text, duration)
                print(f"\nTranscription:\n{'-' * 30}")
                print(final_text)
                print(f"{'-' * 30}")
            else:
                self.file_manager.write_transcription("[No speech detected]", duration)
                print("No speech detected")

        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            self.file_manager.write_transcription(f"[Error: {e}]", duration)

    def _merge_speakers_with_text(self, speaker_segments, text_segments) -> str:
        """Merge speaker diarization with transcribed text segments."""
        if not speaker_segments or not text_segments:
            return " ".join(seg.get("text", "") for seg in text_segments)

        lines = []
        current_speaker = None

        for seg in text_segments:
            # Find speaker for this segment's midpoint
            seg_mid = (seg.get("start", 0) + seg.get("end", 0)) / 2
            speaker = None

            for sp in speaker_segments:
                # Use exclusive end bound to avoid double-matching at boundaries
                if sp.start_time <= seg_mid < sp.end_time:
                    speaker = sp.speaker
                    break

            if speaker is None:
                speaker = "Speaker"

            text = seg.get("text", "").strip()
            if not text:
                continue

            # Group consecutive segments from same speaker
            if speaker != current_speaker:
                current_speaker = speaker
                lines.append(f"\n**{speaker}:**")

            lines.append(text)

        return " ".join(lines).strip()

    def _safe_notification(self, title: str, message: str) -> None:
        """Show notification with error handling."""
        try:
            self.app.show_notification(title, message)
        except Exception:
            print(f"[{title}] {message}")

    def _on_settings(self) -> None:
        """Handle settings button click."""
        settings_script = Path(__file__).parent / "settings_window.py"
        subprocess.Popen([sys.executable, str(settings_script)])

    def _on_quit(self) -> None:
        """Handle quit event."""
        print("Quitting LocalTranslate...")
        if self.recorder.is_recording():
            self.recorder.stop()


def main():
    """Main entry point."""
    app = LocalTranslate()
    app.run()


if __name__ == "__main__":
    main()
