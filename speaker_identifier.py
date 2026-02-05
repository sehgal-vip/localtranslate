"""Speaker diarization using pyannote.audio for LocalTranslate."""

import threading
from typing import Optional

import numpy as np

# Pyannote is imported lazily
_pipeline = None
_pipeline_lock = threading.Lock()


class SpeakerSegment:
    """A segment of audio with speaker identification."""

    def __init__(
        self,
        speaker: str,
        start_time: float,
        end_time: float,
        text: str = "",
    ):
        self.speaker = speaker
        self.start_time = start_time
        self.end_time = end_time
        self.text = text


class SpeakerIdentifier:
    """Handles speaker diarization using pyannote.audio."""

    def __init__(self, huggingface_token: str = ""):
        self.huggingface_token = huggingface_token
        self._pipeline = None
        self._pipeline_lock = threading.Lock()
        self._loading = False
        self._available = False
        self._speaker_map: dict[str, str] = {}  # Maps pyannote labels to "Speaker N"
        self._speaker_count = 0

    def set_token(self, token: str) -> None:
        """Set the HuggingFace token."""
        self.huggingface_token = token
        # Reset pipeline if token changes
        with self._pipeline_lock:
            self._pipeline = None
            self._available = False

    def load_pipeline(self) -> bool:
        """Load the diarization pipeline.

        Returns:
            True if pipeline loaded successfully, False otherwise.
        """
        if not self.huggingface_token:
            print("No HuggingFace token provided for diarization")
            return False

        with self._pipeline_lock:
            if self._pipeline is not None:
                return True

            self._loading = True
            try:
                from pyannote.audio import Pipeline

                print("Loading speaker diarization pipeline...")
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self.huggingface_token,
                )
                self._available = True
                print("Diarization pipeline loaded successfully")
                return True
            except Exception as e:
                print(f"Failed to load diarization pipeline: {e}")
                self._available = False
                return False
            finally:
                self._loading = False

    def is_available(self) -> bool:
        """Check if diarization is available."""
        return self._available and self._pipeline is not None

    def is_loading(self) -> bool:
        """Check if pipeline is currently loading."""
        return self._loading

    def identify_speakers(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[SpeakerSegment]:
        """Identify speakers in an audio segment.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of the audio

        Returns:
            List of SpeakerSegments, empty if diarization not available
        """
        if not self.is_available():
            return []

        try:
            import torch

            # Convert to tensor format expected by pyannote
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Create waveform tensor (batch, channels, samples)
            waveform = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)

            # Run diarization
            diarization = self._pipeline(
                {"waveform": waveform, "sample_rate": sample_rate}
            )

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Map speaker label to friendly name
                if speaker not in self._speaker_map:
                    self._speaker_count += 1
                    self._speaker_map[speaker] = f"Speaker {self._speaker_count}"

                segments.append(
                    SpeakerSegment(
                        speaker=self._speaker_map[speaker],
                        start_time=turn.start,
                        end_time=turn.end,
                    )
                )

            return segments

        except Exception as e:
            print(f"Diarization error: {e}")
            return []

    def get_speaker_for_time(
        self,
        segments: list[SpeakerSegment],
        time: float,
    ) -> Optional[str]:
        """Get the speaker at a specific time.

        Args:
            segments: List of speaker segments
            time: Time in seconds

        Returns:
            Speaker label or None if no speaker found
        """
        for segment in segments:
            if segment.start_time <= time <= segment.end_time:
                return segment.speaker
        return None

    def reset_speakers(self) -> None:
        """Reset speaker mapping (start fresh for new session)."""
        self._speaker_map = {}
        self._speaker_count = 0
