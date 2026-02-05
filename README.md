# LocalTranslate

A macOS menu bar application that captures microphone audio and transcribes it in real-time using a local Whisper model, with optional speaker diarization.

## Features

- **Real-time transcription**: Captures audio and transcribes using OpenAI's Whisper model running locally
- **Menu bar integration**: Runs as a lightweight macOS menu bar app
- **Speaker diarization**: Optional feature to identify and label different speakers (requires HuggingFace token)
- **Timestamped output**: Saves transcriptions with timestamps to Markdown or plain text files
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
| File Format | Output format (.md or .txt) | Markdown |
| Include Timestamps | Add timestamps to transcription lines | Enabled |
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

## Output Format

Transcriptions are saved with timestamps:

**Markdown (.md)**:
```markdown
# Transcription

**Date:** 2024-01-15 14:30:00

---

[00:00:03] Hello, this is a test recording.
[00:00:08] The transcription appears in real-time.
```

**With speaker diarization**:
```markdown
[00:00:03] Speaker 1: Hello, this is a test recording.
[00:00:08] Speaker 2: Yes, I can see the transcription working.
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

## License

MIT License
