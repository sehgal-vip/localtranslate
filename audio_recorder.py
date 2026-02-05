"""Audio capture using sounddevice for LocalTranslate."""

import queue
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioRecorder:
    """Handles microphone audio capture."""

    SAMPLE_RATE = 16000  # Whisper requirement
    CHANNELS = 1
    DTYPE = np.float32
    CHUNK_DURATION = 4  # seconds per chunk for real-time

    def __init__(self, device: Optional[str] = None):
        self.device = device if device != "default" else None
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._recording = False
        self._paused = False
        self._stream: Optional[sd.InputStream] = None
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._chunk_samples = int(self.SAMPLE_RATE * self.CHUNK_DURATION)
        self._lock = threading.Lock()
        # Store full recording for end-of-session transcription
        self._full_recording: list[np.ndarray] = []

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags
    ) -> None:
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")

        if self._paused:
            return

        audio_data = indata.copy().flatten()

        with self._lock:
            # Store for full recording
            self._full_recording.append(audio_data.copy())

            # Buffer for real-time chunks
            self._buffer.append(audio_data)
            self._buffer_samples += len(audio_data)

            # When we have enough samples, push to queue for real-time processing
            if self._buffer_samples >= self._chunk_samples:
                chunk = np.concatenate(self._buffer)
                self.audio_queue.put(chunk[: self._chunk_samples])

                # Keep remainder in buffer
                remainder = chunk[self._chunk_samples :]
                self._buffer = [remainder] if len(remainder) > 0 else []
                self._buffer_samples = len(remainder)

    def start(self) -> None:
        """Start recording audio."""
        if self._recording:
            return

        self._recording = True
        self._paused = False
        self._buffer = []
        self._buffer_samples = 0
        self._full_recording = []

        # Clear any old data from queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        try:
            device_idx = self._get_device_index()
            print(f"Opening audio device: {device_idx if device_idx else 'default'}")

            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=self.DTYPE,
                device=device_idx,
                callback=self._audio_callback,
                blocksize=int(self.SAMPLE_RATE * 0.1),  # 100ms blocks
            )
            self._stream.start()
            print("Audio stream started successfully")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self._recording = False
            raise

    def stop(self) -> Optional[np.ndarray]:
        """Stop recording and return the full recording."""
        if not self._recording:
            return None

        self._recording = False

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                print(f"Error stopping stream: {e}")
            self._stream = None

        # Return full recording
        with self._lock:
            if self._full_recording:
                full_audio = np.concatenate(self._full_recording)
                self._full_recording = []
                self._buffer = []
                self._buffer_samples = 0
                return full_audio

        return None

    def pause(self) -> None:
        """Pause recording."""
        self._paused = True

    def resume(self) -> None:
        """Resume recording."""
        self._paused = False

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def is_paused(self) -> bool:
        """Check if recording is paused."""
        return self._paused

    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get the next audio chunk from the queue for real-time processing."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def set_device(self, device: Optional[str]) -> None:
        """Set the recording device."""
        self.device = device if device != "default" else None

    def _get_device_index(self) -> Optional[int]:
        """Get the device index for the current device setting."""
        if self.device is None:
            return None

        devices = self.list_devices()
        for idx, name in devices:
            if name == self.device:
                return idx
        return None

    @staticmethod
    def list_devices() -> list[tuple[int, str]]:
        """List available input devices."""
        devices = []
        try:
            device_list = sd.query_devices()
            for idx, device in enumerate(device_list):
                if device["max_input_channels"] > 0:
                    devices.append((idx, device["name"]))
        except Exception as e:
            print(f"Error listing devices: {e}")
        return devices

    @staticmethod
    def get_default_device() -> Optional[str]:
        """Get the name of the default input device."""
        try:
            default_idx = sd.default.device[0]
            if default_idx is not None:
                device = sd.query_devices(default_idx)
                return device["name"]
        except Exception:
            pass
        return None
