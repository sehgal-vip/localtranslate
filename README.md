# LocalTranslate

A macOS menu bar application that captures microphone audio and transcribes it locally using OpenAI's Whisper model, with optional speaker diarization.

## Features

- **Local transcription**: Records audio sessions and transcribes them using OpenAI's Whisper model running entirely on-device
- **Menu bar integration**: Runs as a lightweight macOS menu bar app
- **Speaker diarization**: Optional feature to identify and label different speakers (requires HuggingFace token)
- **Flexible output**: Saves transcriptions to Markdown or plain text files with session date and duration
- **System audio capture**: Record and transcribe system audio (video calls, podcasts, YouTube) alongside your microphone using BlackHole
- **Configurable settings**: Choose Whisper model size, output format, microphone, and more

## Requirements

- macOS
- Python 3.10+
- Microphone access

## Installation

1. Clone or download this repository:
   ```bash
   cd /Users/apple/localtranslate
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Grant microphone permissions when prompted on first run.

## Usage

### Starting the App

```bash
python main.py
```

The app will appear in your menu bar.

### Menu Bar Controls

- **Start Recording / Stop Recording**: Toggle audio capture and transcription
- **Open Output Folder**: Open the folder containing transcription files
- **Settings...**: Configure the app
- **Quit**: Exit the application

### Settings

Access settings through the menu bar to configure:

| Setting | Description | Default |
|---------|-------------|---------|
| Output Folder | Where transcription files are saved | `~/Documents/Transcriptions` |
| Whisper Model | Model size (tiny/base/small/medium/large) | `base` |
| Microphone | Input device selection | System default |
| Include System Audio | Capture system audio (requires BlackHole) | Disabled |
| System Audio Device | Device for system audio capture | (empty) |
| File Format | Output format (.md or .txt) | Markdown |
| Include Timestamps | Add timestamps to transcription lines (planned) | Enabled |
| Enable Diarization | Identify different speakers | Disabled |
| HuggingFace Token | Required for speaker diarization | (empty) |

### Whisper Model Selection

| Model | Size | Speed | Accuracy | Recommended For |
|-------|------|-------|----------|-----------------|
| tiny | ~39MB | Fastest | Basic | Quick testing |
| base | ~74MB | Fast | Good | **Daily use** |
| small | ~244MB | Moderate | Better | Higher accuracy |
| medium | ~769MB | Slow | High | Quality transcription |
| large | ~1.5GB | Slowest | Best | Maximum accuracy |

The model will be downloaded automatically on first use.

### Speaker Diarization (Optional)

To enable speaker identification:

1. Create a HuggingFace account at https://huggingface.co
2. Accept the terms for `pyannote/speaker-diarization-3.1`
3. Generate an access token at https://huggingface.co/settings/tokens
4. Enter the token in Settings and enable diarization

Note: Diarization adds processing latency and requires additional model downloads.

### System Audio Capture (Optional)

Capture both microphone AND system audio simultaneously. This enables transcribing video calls, podcasts, YouTube videos, or any audio playing on your Mac alongside your voice.

#### Step 1: Install BlackHole

```bash
brew install blackhole-2ch
```

#### Step 2: Create Multi-Output Device

1. Open **Audio MIDI Setup** (`/Applications/Utilities/Audio MIDI Setup.app`)
2. Click the **+** button at the bottom left → **Create Multi-Output Device**
3. Check both your speakers/headphones AND "BlackHole 2ch"
4. Right-click the new Multi-Output Device → **Use This Device For Sound Output**

This setup allows you to hear audio normally while BlackHole captures it for transcription.

#### Step 3: Enable in Settings

1. Open LocalTranslate Settings from the menu bar
2. Check "Include System Audio"
3. Select "BlackHole 2ch" from the System Audio Device dropdown
4. Save settings

Now when you record, both your microphone input and system audio will be captured and transcribed together.

## Output Format

Transcriptions are saved with a header containing the date and session duration:

**Markdown (.md)**:
```markdown
# Transcription

**Date:** 2024-01-15 14:30:00

**Duration:** 00:00:15

---

Hello, this is a test recording. The transcription appears after recording stops.
```

**With speaker diarization**:
```markdown
**Speaker 1:** Hello, this is a test recording. **Speaker 2:** Yes, I can see the transcription working.
```

## Configuration File

Settings are stored in `~/.localtranslate/config.json`. You can edit this file directly or use the Settings window.

## Troubleshooting

### Microphone not working
- Ensure microphone permissions are granted in System Preferences > Security & Privacy > Privacy > Microphone
- Check that the correct microphone is selected in Settings

### Model download issues
- First run requires internet connection to download Whisper model
- Models are cached in `~/.cache/whisper/`

### Slow transcription
- Try a smaller model (tiny or base)
- Disable diarization if not needed
- Close other resource-intensive applications

### tkinter issues on macOS
If Settings window doesn't appear:
```bash
brew install python-tk
```

## Dependencies

- `openai-whisper`: Local speech recognition
- `sounddevice`: Cross-platform audio capture
- `numpy`: Audio data processing
- `rumps`: macOS menu bar integration
- `torch`: Neural network backend
- `pyannote.audio`: Speaker diarization (optional feature)

## Recent Changes

### System Audio Capture (Feb 2026)

- **BlackHole integration**: Capture system audio alongside microphone input using BlackHole as a virtual audio loopback device
- **Audio mixing**: Mic and system audio streams are mixed together with normalization to prevent clipping
- **Settings UI**: Added checkbox and device dropdown in the Settings window to enable and configure system audio capture
- **Documentation**: Added setup instructions for installing BlackHole and creating a Multi-Output Device

### Diarization & Audio Processing Fixes (Feb 2026)

**Diarization Fixes:**

- **Fixed timing alignment**: Diarization now runs on the same processed audio as transcription, fixing speaker label misalignment issues
- **Updated pyannote 3.x API**: Fixed compatibility with pyannote.audio 3.x which changed the output format from `Annotation` to `DiarizeOutput`
- **Fixed tensor format**: Corrected waveform tensor shape from `(batch, channel, time)` to `(channel, time)` as required by pyannote
- **Improved error reporting**: Added detailed logging showing detected speaker segments with timestamps, warnings when no segments found, and full tracebacks on errors
- **Fixed boundary matching**: Speaker segment matching now uses exclusive end bounds to prevent double-matching at segment boundaries

**Audio Processing Improvements:**

- **Lowered high-pass filter**: Changed from 80Hz to 60Hz to preserve male voice fundamentals (85-180Hz range)
- **Reduced noise gate threshold**: Lowered from 0.01 to 0.005 RMS to avoid cutting quiet speech
- **Enabled pre-emphasis filter**: Boosts high frequencies/consonants for improved speech recognition accuracy
- **Improved padding**: Changed from zero-padding to edge-padding for short audio clips to avoid confusing Whisper

## License

MIT License
