"""Whisper transcription logic for LocalTranslate."""

import threading
from typing import Optional

import numpy as np

# Whisper is imported lazily to speed up app startup
_whisper = None
_whisper_lock = threading.Lock()


def _get_whisper():
    """Lazy load whisper module."""
    global _whisper
    if _whisper is None:
        with _whisper_lock:
            if _whisper is None:
                import whisper

                _whisper = whisper
    return _whisper


class TranscriptionResult:
    """Result from a transcription."""

    def __init__(self, text: str, start_time: float = 0.0, end_time: float = 0.0):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time

    def __repr__(self) -> str:
        return f"TranscriptionResult(text='{self.text[:50]}...', start={self.start_time:.2f}, end={self.end_time:.2f})"


class Transcriber:
    """Handles Whisper model loading and transcription."""

    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None
        self._model_lock = threading.Lock()
        self._loading = False
        self._cumulative_time: float = 0.0

    def load_model(self, model_name: Optional[str] = None) -> None:
        """Load the Whisper model (downloads if needed)."""
        if model_name:
            self.model_name = model_name

        with self._model_lock:
            if self._model is not None and model_name is None:
                return  # Already loaded

            self._loading = True
            try:
                whisper = _get_whisper()
                print(f"Loading Whisper model: {self.model_name}")
                self._model = whisper.load_model(self.model_name)
                print(f"Model {self.model_name} loaded successfully")
            finally:
                self._loading = False

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def is_loading(self) -> bool:
        """Check if model is currently loading."""
        return self._loading

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Optional[TranscriptionResult]:
        """Transcribe an audio chunk.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of the audio (should be 16000 for Whisper)

        Returns:
            TranscriptionResult or None if transcription failed
        """
        if self._model is None:
            self.load_model()

        if self._model is None:
            return None

        try:
            # Ensure audio is the right format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize if needed
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            # Transcribe with optimized parameters
            result = self._model.transcribe(
                audio,
                language="en",
                fp16=False,  # Use fp32 for CPU compatibility
                temperature=0,  # Deterministic output, less hallucination
                compression_ratio_threshold=2.4,  # Filter out bad segments
                logprob_threshold=-1.0,  # Filter low confidence
                no_speech_threshold=0.6,  # Better silence detection
                condition_on_previous_text=False,  # Avoid error propagation
                initial_prompt="This is a clear speech recording.",  # Hint for better decoding
            )

            text = result.get("text", "").strip()

            if not text:
                return None

            # Calculate timing
            duration = len(audio) / sample_rate
            start_time = self._cumulative_time
            end_time = start_time + duration
            self._cumulative_time = end_time

            return TranscriptionResult(
                text=text,
                start_time=start_time,
                end_time=end_time,
            )

        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def reset_timing(self) -> None:
        """Reset the cumulative timing counter."""
        self._cumulative_time = 0.0

    def transcribe_with_segments(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[str, list[dict]]:
        """Transcribe audio and return text with segment timing.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of the audio

        Returns:
            Tuple of (full_text, list of segment dicts with start/end/text)
        """
        if self._model is None:
            self.load_model()

        if self._model is None:
            return "", []

        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            # Transcribe with segment info
            result = self._model.transcribe(
                audio,
                language="en",
                fp16=False,
                temperature=0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                initial_prompt="This is a clear speech recording.",
                word_timestamps=True,  # Enable word-level timestamps
            )

            text = result.get("text", "").strip()
            segments = result.get("segments", [])

            # Extract segment info
            segment_list = []
            for seg in segments:
                segment_list.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", ""),
                })

            return text, segment_list

        except Exception as e:
            print(f"Transcription error: {e}")
            return "", []

    def change_model(self, model_name: str) -> None:
        """Change to a different Whisper model."""
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {self.AVAILABLE_MODELS}")

        if model_name == self.model_name and self._model is not None:
            return

        # Unload current model
        with self._model_lock:
            self._model = None

        # Load new model
        self.model_name = model_name
        self.load_model()

    def get_model_name(self) -> str:
        """Get the current model name."""
        return self.model_name
