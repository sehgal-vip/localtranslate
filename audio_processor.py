"""Audio processing and optimization for better transcription."""

import numpy as np
from typing import Optional


class AudioProcessor:
    """Processes audio to improve transcription quality."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply all processing steps to audio."""
        if len(audio) == 0:
            return audio

        # Ensure float32
        audio = audio.astype(np.float32)

        # Remove DC offset
        audio = self.remove_dc_offset(audio)

        # Apply high-pass filter to remove low-frequency noise
        audio = self.high_pass_filter(audio, cutoff=80)

        # Normalize audio levels
        audio = self.normalize(audio)

        # Apply noise gate to reduce background noise
        audio = self.noise_gate(audio, threshold=0.01)

        # Trim silence from start and end
        audio = self.trim_silence(audio)

        # Final normalization
        audio = self.normalize(audio)

        # Pad to minimum length for Whisper (0.5 seconds)
        min_samples = int(self.sample_rate * 0.5)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        return audio

    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio."""
        return audio - np.mean(audio)

    def normalize(self, audio: np.ndarray, target_level: float = 0.9) -> np.ndarray:
        """Normalize audio to target level."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * (target_level / max_val)
        return audio

    def high_pass_filter(self, audio: np.ndarray, cutoff: float = 80) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise."""
        # Simple first-order high-pass filter
        rc = 1.0 / (2.0 * np.pi * cutoff)
        dt = 1.0 / self.sample_rate
        alpha = rc / (rc + dt)

        filtered = np.zeros_like(audio)
        filtered[0] = audio[0]

        for i in range(1, len(audio)):
            filtered[i] = alpha * (filtered[i-1] + audio[i] - audio[i-1])

        return filtered

    def noise_gate(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Apply noise gate to reduce background noise."""
        # Calculate envelope using RMS in small windows
        window_size = int(self.sample_rate * 0.02)  # 20ms windows
        envelope = np.zeros_like(audio)

        for i in range(0, len(audio), window_size):
            end = min(i + window_size, len(audio))
            rms = np.sqrt(np.mean(audio[i:end] ** 2))
            envelope[i:end] = rms

        # Apply gate with smooth transition
        gate = np.clip(envelope / threshold, 0, 1)

        # Smooth the gate to avoid clicks
        smooth_window = int(self.sample_rate * 0.01)  # 10ms smoothing
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            gate = np.convolve(gate, kernel, mode='same')

        return audio * gate

    def trim_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.01,
        min_silence_duration: float = 0.1
    ) -> np.ndarray:
        """Trim silence from start and end of audio."""
        min_samples = int(min_silence_duration * self.sample_rate)

        # Find start (first sample above threshold)
        start = 0
        window_size = int(self.sample_rate * 0.02)
        for i in range(0, len(audio) - window_size, window_size):
            rms = np.sqrt(np.mean(audio[i:i+window_size] ** 2))
            if rms > threshold:
                start = max(0, i - min_samples)
                break

        # Find end (last sample above threshold)
        end = len(audio)
        for i in range(len(audio) - window_size, 0, -window_size):
            rms = np.sqrt(np.mean(audio[i:i+window_size] ** 2))
            if rms > threshold:
                end = min(len(audio), i + window_size + min_samples)
                break

        if start >= end:
            return audio

        return audio[start:end]

    def apply_preemphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter to boost high frequencies."""
        return np.append(audio[0], audio[1:] - coef * audio[:-1])

    def compute_rms(self, audio: np.ndarray) -> float:
        """Compute RMS level of audio."""
        return float(np.sqrt(np.mean(audio ** 2)))

    def compute_peak(self, audio: np.ndarray) -> float:
        """Compute peak level of audio."""
        return float(np.max(np.abs(audio)))
