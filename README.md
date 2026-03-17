# Scribe

Scribe is a local-first meeting workbench for turning audio into usable notes.

It transcribes audio, applies speaker diarization when available, keeps the raw transcript visible early, and can stream local LLM analysis through Ollama. The UI is built in Gradio and is designed for power users who want fast access to raw text instead of waiting for the entire pipeline to finish.

## What Scribe Does

- Transcribes uploaded audio files with `mlx-whisper`
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
2. `mlx-whisper` generates a transcript with word timestamps.
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
export SCRIBE_DEFAULT_LLM=gemma3:27b
```

Important environment variables:

- `HF_TOKEN`
  - Enables `pyannote.audio` speaker diarization. Without it, audio still transcribes, but diarization falls back to a single speaker.
- `SCRIBE_MODEL_NAME`
  - Whisper model used by `mlx-whisper`.
- `SCRIBE_DEFAULT_LLM`
  - Preferred Ollama model shown in the UI.
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

If you want LLM analysis, make sure Ollama is running and the model is pulled:

```bash
ollama serve
ollama pull gemma3:27b
```

If you want to warm model caches before first use, run:

```bash
python download_models.py
```

That script will:

- Trigger the Whisper model download
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
