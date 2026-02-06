"""Audio capture using sounddevice for LocalTranslate."""

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioRecorder:
    """Handles microphone and system audio capture."""

    SAMPLE_RATE = 16000  # Whisper requirement
    CHANNELS = 1
    DTYPE = np.float32
    CHUNK_DURATION = 4  # seconds per chunk for real-time

    def __init__(self, device: Optional[str] = None, system_device: Optional[str] = None):
        self.device = device if device != "default" else None
        self.system_device = system_device if system_device else None
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._recording = False
        self._paused = False
        self._stream: Optional[sd.InputStream] = None
        self._system_stream: Optional[sd.InputStream] = None
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._chunk_samples = int(self.SAMPLE_RATE * self.CHUNK_DURATION)
        self._lock = threading.Lock()
        # Store full recording for end-of-session transcription
        self._full_recording: list[np.ndarray] = []
        # Store system audio separately for mixing
        self._system_recording: list[np.ndarray] = []
        self._include_system_audio = False

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

    def _system_audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags
    ) -> None:
        """Callback for system audio stream."""
        if status:
            print(f"System audio status: {status}")

        if self._paused:
            return

        audio_data = indata.copy().flatten()

        with self._lock:
            self._system_recording.append(audio_data.copy())

    def start(self, include_system_audio: bool = False) -> None:
        """Start recording audio."""
        if self._recording:
            return

        self._recording = True
        self._paused = False
        self._buffer = []
        self._buffer_samples = 0
        self._full_recording = []
        self._system_recording = []
        self._include_system_audio = include_system_audio and self.system_device

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

            # Start system audio stream if enabled
            if self._include_system_audio:
                try:
                    system_idx = self._get_system_device_index()
                    if system_idx is not None:
                        print(f"Opening system audio device: {self.system_device}")
                        self._system_stream = sd.InputStream(
                            samplerate=self.SAMPLE_RATE,
                            channels=self.CHANNELS,
                            dtype=self.DTYPE,
                            device=system_idx,
                            callback=self._system_audio_callback,
                            blocksize=int(self.SAMPLE_RATE * 0.1),
                        )
                        self._system_stream.start()
                        print("System audio stream started successfully")
                    else:
                        print(f"Warning: System audio device '{self.system_device}' not found")
                        self._include_system_audio = False
                except Exception as e:
                    print(f"Error starting system audio stream: {e}")
                    self._include_system_audio = False
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self._recording = False
            raise

    def stop(self) -> Optional[np.ndarray]:
        """Stop recording and return the full recording (mixed with system audio if enabled)."""
        if not self._recording:
            return None

        self._recording = False

        # Stop microphone stream
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                print(f"Error stopping stream: {e}")
            self._stream = None

        # Stop system audio stream
        if self._system_stream:
            try:
                self._system_stream.stop()
                self._system_stream.close()
            except Exception as e:
                print(f"Error stopping system audio stream: {e}")
            self._system_stream = None

        # Return full recording (mixed if system audio was captured)
        with self._lock:
            if self._full_recording:
                mic_audio = np.concatenate(self._full_recording)

                # Mix with system audio if available
                if self._include_system_audio and self._system_recording:
                    system_audio = np.concatenate(self._system_recording)
                    mixed_audio = self._mix_audio(mic_audio, system_audio)
                    print(f"Mixed mic ({len(mic_audio)/self.SAMPLE_RATE:.1f}s) + system ({len(system_audio)/self.SAMPLE_RATE:.1f}s) audio")
                else:
                    mixed_audio = mic_audio

                self._full_recording = []
                self._system_recording = []
                self._buffer = []
                self._buffer_samples = 0
                return mixed_audio

        return None

    def _mix_audio(self, mic_audio: np.ndarray, system_audio: np.ndarray) -> np.ndarray:
        """Mix microphone and system audio together."""
        # Match lengths by padding the shorter one
        max_len = max(len(mic_audio), len(system_audio))

        if len(mic_audio) < max_len:
            mic_audio = np.pad(mic_audio, (0, max_len - len(mic_audio)), mode='constant')
        if len(system_audio) < max_len:
            system_audio = np.pad(system_audio, (0, max_len - len(system_audio)), mode='constant')

        # Mix by averaging (prevents clipping)
        mixed = (mic_audio + system_audio) / 2.0

        # Normalize to prevent clipping while preserving dynamics
        peak = np.max(np.abs(mixed))
        if peak > 0.95:
            mixed = mixed * (0.95 / peak)

        return mixed.astype(self.DTYPE)

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

    def set_system_device(self, device: Optional[str]) -> None:
        """Set the system audio device (e.g., BlackHole 2ch)."""
        self.system_device = device if device else None

    def _get_device_index(self) -> Optional[int]:
        """Get the device index for the current device setting."""
        if self.device is None:
            return None

        devices = self.list_devices()
        for idx, name in devices:
            if name == self.device:
                return idx
        return None

    def _get_system_device_index(self) -> Optional[int]:
        """Get the device index for the system audio device."""
        if self.system_device is None:
            return None

        devices = self.list_devices()
        for idx, name in devices:
            if name == self.system_device:
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
