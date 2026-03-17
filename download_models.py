import os

from pyannote.audio import Pipeline

from scribe.config import load_config
from scribe.transcription import resolve_transcription_backend


CONFIG = load_config()
HF_TOKEN = CONFIG.hf_token or os.getenv("HF_TOKEN")


def download_whisper_model_mlx():
    import mlx_whisper

    model_name = CONFIG.mlx_model_name
    print(f"--- Downloading MLX Whisper Model: {model_name} ---")
    print("This is optimized for Apple Silicon Macs.")
    print("The download is triggered by a dummy transcription call.")

    try:
        mlx_whisper.transcribe("dummy.wav", path_or_hf_repo=model_name, word_timestamps=True)
    except FileNotFoundError:
        print(f"\n✅ MLX Whisper model '{model_name}' downloaded successfully.\n")
    except Exception as exc:
        print(f"\n❌ An error occurred during MLX Whisper model download: {exc}")


def download_whisper_model_faster_whisper():
    from faster_whisper import WhisperModel

    model_name = CONFIG.faster_whisper_model
    print(f"--- Downloading Faster-Whisper Model: {model_name} ---")
    print("This backend is used on non-Apple-Silicon systems by default.")

    try:
        WhisperModel(model_name, device="cpu", compute_type="int8")
        print(f"\n✅ Faster-Whisper model '{model_name}' downloaded successfully.\n")
    except Exception as exc:
        print(f"\n❌ An error occurred during Faster-Whisper model download: {exc}")


def download_transcription_model():
    backend = resolve_transcription_backend(CONFIG.transcription_backend)
    print(f"Selected transcription backend: {backend}")
    if backend == "mlx":
        download_whisper_model_mlx()
        return
    download_whisper_model_faster_whisper()


def download_pyannote_models():
    print("--- Downloading Pyannote.audio Models ---")
    print("This will download the diarization pipeline used for speaker separation.")

    if not HF_TOKEN:
        print("\n❌ Hugging Face token not set. Cannot download Pyannote models.")
        print("Please set the HF_TOKEN environment variable before running.")
        return

    try:
        try:
            Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=HF_TOKEN,
            )
        except TypeError:
            Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN,
            )
        print("\n✅ Pyannote.audio models downloaded successfully.\n")
    except Exception as exc:
        print(f"\n❌ An error occurred during Pyannote model download: {exc}")
        print("Please ensure you accepted the model terms on Hugging Face.")


if __name__ == "__main__":
    print("Starting model pre-download for Scribe...")
    print("This only needs to be run once per environment.\n")

    download_transcription_model()
    download_pyannote_models()

    print("Model download step complete.")
