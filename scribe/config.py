from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass


MODEL_NAME = "mlx-community/whisper-large-v3-turbo"
DEFAULT_LLM = "gemma3:27b"
DEFAULT_MAX_SPEAKERS = 8


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppConfig:
    hf_token: str
    model_name: str
    default_llm: str
    default_save_dir: str
    enable_remote_tokenizer_lookup: bool
    max_speakers: int


def cpu_thread_count() -> int:
    return max(1, mp.cpu_count() - 1)


def load_config() -> AppConfig:
    return AppConfig(
        hf_token=os.getenv("HF_TOKEN", "").strip(),
        model_name=os.getenv("SCRIBE_MODEL_NAME", MODEL_NAME).strip() or MODEL_NAME,
        default_llm=os.getenv("SCRIBE_DEFAULT_LLM", DEFAULT_LLM).strip() or DEFAULT_LLM,
        default_save_dir=os.path.abspath(
            os.path.expanduser(
                os.getenv("SCRIBE_DEFAULT_SAVE_DIR", "~/Downloads/ScribeTranscripts")
            )
        ),
        enable_remote_tokenizer_lookup=env_flag(
            "SCRIBE_ENABLE_REMOTE_TOKENIZER_LOOKUP", False
        ),
        max_speakers=env_int("SCRIBE_MAX_SPEAKERS", DEFAULT_MAX_SPEAKERS),
    )


def launch_settings() -> dict[str, object]:
    return {
        "server_name": os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        "server_port": env_int("PORT", env_int("GRADIO_SERVER_PORT", 7860)),
        "share": env_flag("GRADIO_SHARE", False),
        "debug": env_flag("GRADIO_DEBUG", False),
    }

