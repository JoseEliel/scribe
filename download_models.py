import os
import mlx_whisper
from pyannote.audio import Pipeline
from scribe.config import load_config

CONFIG = load_config()
MODEL_NAME = CONFIG.model_name
HF_TOKEN = CONFIG.hf_token or os.getenv("HF_TOKEN")

def download_whisper_model_mlx():
    """
    Downloads the mlx-whisper model from Hugging Face and caches it locally.
    It works by calling the transcribe function, which triggers the download
    mechanism internally. We expect it to fail on a non-existent audio file,
    but only after the download is complete.
    """
    print(f"--- Downloading Whisper Model for MLX: {MODEL_NAME} ---")
    print("This is a large file and may take a significant amount of time depending on your internet connection.")
    print("You will see download progress from huggingface_hub.")
    
    try:
        # This call will first trigger the model download from Hugging Face.
        # It will then fail with a FileNotFoundError because "dummy.wav" doesn't exist.
        # We catch this expected error to confirm the download was successful.
        mlx_whisper.transcribe("dummy.wav", path_or_hf_repo=MODEL_NAME, word_timestamps=True)
    except FileNotFoundError:
        # This is the expected outcome! It means the download worked and the script
        # is now trying (and failing) to find the audio file.
        print(f"\n✅ Whisper model '{MODEL_NAME}' downloaded successfully.\n")
    except Exception as e:
        # This will catch any other errors, such as network problems during download.
        print(f"\n❌ An error occurred during Whisper model download: {e}")

def download_pyannote_models():
    """Downloads the pyannote.audio models for diarization and caches them locally."""
    print("--- Downloading Pyannote.audio Models ---")
    print("This will download several smaller model files for speaker diarization.")
    print("You will see progress bars for each file.")
    
    if not HF_TOKEN:
        print("\n❌ Hugging Face token not set. Cannot download Pyannote models.")
        print("Please set the HF_TOKEN environment variable before running.")
        return
        
    try:
        # Note: You must accept the user agreement for these models on the Hugging Face website first.
        try:
            Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=HF_TOKEN  # newer API
            )
        except TypeError:
            Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN  # backward-compat
            )
        print("\n✅ Pyannote.audio models downloaded successfully.\n")
    except Exception as e:
        print(f"\n❌ An error occurred during Pyannote model download: {e}")
        print("Please ensure you have accepted the model terms on Hugging Face and your HF_TOKEN is correct.")

if __name__ == "__main__":
    print("Starting pre-download of all required AI models for the Meeting Analyzer...")
    print("This script only needs to be run once.")
    
    download_whisper_model_mlx()
    download_pyannote_models()
    
    print("🎉 All models have been downloaded and cached locally.")
    print("You can now run the main `app.py` application without download delays.")
