# Scribe

Scribe is a local-first meeting workbench for turning audio into usable notes.

It transcribes audio, applies speaker diarization when available, keeps the raw transcript visible early, and can stream local LLM analysis through Ollama. The UI is built in Gradio and is designed for power users who want fast access to raw text instead of waiting for the entire pipeline to finish.

## What Scribe Does

- Transcribes uploaded audio files with [mlx-whisper](https://pypi.org/project/mlx-whisper/) on Apple Silicon Macs
- Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on other platforms by default
- Runs speaker diarization with `pyannote.audio` when `HF_TOKEN` is configured
- Falls back to a single-speaker transcript when diarization is unavailable
- Lets you upload existing diarized `.txt` transcripts for analysis-only runs
- Streams three optional analysis views through Ollama:
  - Executive summary
  - Detailed summary
  - Action items
- Keeps a live raw transcript available before diarization and LLM analysis finish
- Optionally saves diarized transcripts to disk and exposes them for download in the UI

## Why This App Exists

Most meeting tools force you to wait for the "nice" version of the output. Scribe is built around the opposite constraint: raw text is often the thing you need first.

That is why the app surfaces the raw transcript as soon as transcription completes. If diarization is slow, you can still copy the text and move on.

## How The Pipeline Works

1. Audio is normalized to mono 16 kHz WAV with `ffmpeg` when needed.
2. The app transcribes with `mlx-whisper` on Apple Silicon Macs or `faster-whisper` elsewhere.
3. `pyannote.audio` assigns speaker segments if diarization is enabled.
4. Timestamped words are aligned back onto speaker turns.
5. The diarized transcript is formatted for reading and downstream prompts.
6. If Ollama is enabled, the diarized transcript is sent through three analysis prompts.

For transcript-only runs, Scribe skips the audio pipeline and analyzes uploaded `.txt` files directly.

## Requirements

- Python `3.11`
- `ffmpeg` available on `PATH`
- Ollama installed and running if you want summaries
- A Hugging Face token if you want true multi-speaker diarization
- Acceptance of the `pyannote/speaker-diarization-3.1` model terms on Hugging Face
- Apple Silicon is optional, not required

## Installation

Create a virtual environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install system dependencies separately if they are not already present:

- `ffmpeg`
- `ollama` if you want local summaries

## Configuration

This project does **not** auto-load `.env` files. `.env.example` is a reference file only.

You can either export variables manually in your shell or source a file yourself.

Example:

```bash
cp .env.example .env
set -a
source .env
set +a
```

Minimum useful configuration:

```bash
export HF_TOKEN=hf_your_token_here
export SCRIBE_DEFAULT_LLM=gemma3:27b  # example default
```

Important environment variables:

- `HF_TOKEN`
  - Enables `pyannote.audio` speaker diarization. Without it, audio still transcribes, but diarization falls back to a single speaker.
- `SCRIBE_TRANSCRIPTION_BACKEND`
  - `auto`, `mlx`, or `faster-whisper`. `auto` uses MLX on Apple Silicon Macs and Faster-Whisper elsewhere.
- `SCRIBE_MLX_MODEL_NAME`
  - Whisper model used by `mlx-whisper` on Apple Silicon Macs.
- `SCRIBE_FASTER_WHISPER_MODEL`
  - Whisper model used by `faster-whisper` on non-Apple-Silicon systems.
- `SCRIBE_MODEL_NAME`
  - Legacy alias for `SCRIBE_MLX_MODEL_NAME`.
- `SCRIBE_DEFAULT_LLM`
  - Preferred Ollama model shown in the UI. This is just the default selection, not a hard requirement.
- `SCRIBE_DEFAULT_SAVE_DIR`
  - Save fallback for uploaded files that do not have a stable original path.
- `SCRIBE_MAX_SPEAKERS`
  - Upper bound for automatic diarization speaker search.
- `SCRIBE_ENABLE_REMOTE_TOKENIZER_LOOKUP`
  - Off by default. Enables more accurate token counting for supported Ollama model families.
- `GRADIO_SERVER_NAME`
- `GRADIO_SERVER_PORT`
- `PORT`
- `GRADIO_SHARE`
- `GRADIO_DEBUG`

## Prepare Local Models

Scribe is not tied to a specific Ollama model. Any local chat-capable Ollama model should work for the analysis step. `gemma3:27b` is used here only as the repo default and example value.

For transcription, Scribe uses different backends by platform:

- Apple Silicon macOS: `mlx-whisper`
- Other systems: `faster-whisper`

If you want LLM analysis, make sure Ollama is running and pull whichever model you want to use:

```bash
ollama serve
ollama pull gemma3:27b
```

If you want to warm the transcription and diarization model caches before first use, run:

```bash
python download_models.py
```

That script will:

- Trigger the active transcription backend model download
- Download the pyannote diarization pipeline if `HF_TOKEN` is set

## Run The App

```bash
python app.py
```

By default the Gradio app listens on `0.0.0.0:7860`. Override that with `PORT` or `GRADIO_SERVER_PORT` if needed.

## Input Modes

- `Audio`
  - Upload one or more audio files for transcription, diarization, and optional LLM analysis.
- `Transcript Only`
  - Upload diarized `.txt` files for analysis without running the speech pipeline.
- `Audio + Transcript`
  - Process audio files and transcript files in one run.

## Output Surfaces

- `Live Raw Transcript`
  - Updates as soon as transcription finishes. This is the fastest usable output.
- `Diarized Transcript`
  - Speaker-attributed transcript, plus downloadable `.txt` files when available.
- `AI Analysis`
  - Executive summary, detailed summary, and action items.

## Operational Notes

- Raw transcript is intentionally first-class. It is available before diarization completes.
- `SCRIBE_TRANSCRIPTION_BACKEND=auto` picks MLX only on Apple Silicon Macs. Intel Macs, Linux, and Windows fall back to Faster-Whisper.
- Saving diarized transcripts only applies to audio inputs.
- Prompt editing supports both `{{transcript}}` and `{transcript}` placeholders.
- If diarization cannot initialize, Scribe still produces a transcript and labels it as a single speaker.
- If Ollama is unavailable, the app can still run transcription and diarization without summaries.

## Testing

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

Basic syntax verification:

```bash
python -m py_compile app.py scribe/*.py tests/*.py download_models.py
```

## Project Layout

```text
app.py
download_models.py
scribe/
  analysis.py
  config.py
  files.py
  transcription.py
  ui.py
tests/
```

## Troubleshooting

- `ffmpeg` not found
  - Install `ffmpeg` and make sure it is on your shell `PATH`.
- No Ollama models appear in the UI
  - Start Ollama with `ollama serve` and pull at least one model.
- Diarization does not separate speakers
  - Confirm `HF_TOKEN` is exported and that you have accepted the pyannote model terms.
- First run is slow
  - Model downloads and lazy initialization happen on first use unless you pre-warm them with `download_models.py`.
- Transcription backend confusion
  - `gemma3:27b` is only the default Ollama example. Speech transcription uses `mlx-whisper` or `faster-whisper`, not Ollama.
