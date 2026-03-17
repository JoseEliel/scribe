from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from typing import Any

from .config import AppConfig, cpu_thread_count
from .files import format_hms


_UNSET = object()


@dataclass
class TranscriptionResult:
    raw_text: str
    word_chunks: list[dict[str, Any]]


class SpeechPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.thread_count = cpu_thread_count()
        self._torch = None
        self._mlx_whisper = None
        self._pyannote_pipeline = _UNSET
        self._diar_device: str | None = None
        self._diarization_warning: str = ""

    def startup_note(self) -> str:
        if not self.config.hf_token:
            return (
                "⚠️ Speaker diarization is in fallback mode. "
                "Set `HF_TOKEN` to enable pyannote speaker separation."
            )
        return (
            "ℹ️ Speaker diarization loads on first audio run. "
            "Apple Silicon will prefer MPS when available."
        )

    def _ensure_torch(self):
        if self._torch is None:
            import torch

            os.environ["OMP_NUM_THREADS"] = str(self.thread_count)
            os.environ["MKL_NUM_THREADS"] = str(self.thread_count)
            torch.set_num_threads(self.thread_count)
            try:
                torch.set_num_interop_threads(self.thread_count)
            except RuntimeError:
                pass
            self._torch = torch
        return self._torch

    def _detect_diar_device(self) -> str:
        if self._diar_device:
            return self._diar_device
        torch = self._ensure_torch()
        if torch.backends.mps.is_available():
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            self._diar_device = "mps"
        else:
            self._diar_device = "cpu"
        return self._diar_device

    def _ensure_diarization_pipeline(self):
        if self._pyannote_pipeline is not _UNSET:
            return self._pyannote_pipeline

        self._detect_diar_device()
        if not self.config.hf_token:
            self._diarization_warning = "HF_TOKEN not set."
            self._pyannote_pipeline = None
            return self._pyannote_pipeline

        try:
            from pyannote.audio import Pipeline

            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self.config.hf_token,
                )
            except TypeError:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.config.hf_token,
                )
        except Exception as exc:
            self._diarization_warning = str(exc)
            self._pyannote_pipeline = None
            return self._pyannote_pipeline

        try:
            torch = self._ensure_torch()
            pipeline.to(torch.device(self._diar_device or "cpu"))
            self._pyannote_pipeline = pipeline
        except Exception as exc:
            self._diarization_warning = str(exc)
            self._pyannote_pipeline = None
        return self._pyannote_pipeline

    def empty_accelerator_cache(self) -> None:
        if self._detect_diar_device() != "mps":
            return
        try:
            torch = self._ensure_torch()
            torch.mps.empty_cache()
        except Exception:
            pass

    def get_audio_duration_seconds(self, path: str) -> float:
        try:
            if path.lower().endswith(".wav"):
                with contextlib.closing(wave.open(path, "rb")) as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    return frames / float(rate) if rate else 0.0
            output = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    path,
                ],
                stderr=subprocess.DEVNULL,
            )
            return float(output.strip())
        except Exception:
            return 0.0

    def is_target_wav(self, path: str) -> bool:
        try:
            with contextlib.closing(wave.open(path, "rb")) as wav_file:
                return (
                    wav_file.getframerate() == 16000
                    and wav_file.getnchannels() == 1
                    and wav_file.getsampwidth() == 2
                    and wav_file.getcomptype() == "NONE"
                )
        except Exception:
            return False

    def preprocess_audio_to_wav(self, input_path: str) -> tuple[str, bool]:
        if input_path.lower().endswith(".wav") and self.is_target_wav(input_path):
            return input_path, False

        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    tmp_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return tmp_path, True
        except subprocess.CalledProcessError:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return input_path, False

    def _ensure_mlx_whisper(self):
        if self._mlx_whisper is None:
            import mlx_whisper

            self._mlx_whisper = mlx_whisper
        return self._mlx_whisper

    def transcribe_audio(self, audio_path: str) -> TranscriptionResult:
        mlx_whisper = self._ensure_mlx_whisper()
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self.config.model_name,
            word_timestamps=True,
        )
        segments = result.get("segments", []) or []
        raw_parts: list[str] = []
        words: list[dict[str, Any]] = []
        for segment in segments:
            segment_text = (segment.get("text") or "").strip()
            if segment_text:
                raw_parts.append(segment_text)
            for word in segment.get("words", []) or []:
                if isinstance(word, dict) and {"word", "start", "end"} <= set(word):
                    text = (word.get("word") or "").strip()
                    if text:
                        words.append(
                            {
                                "text": text,
                                "timestamp": (word["start"], word["end"]),
                            }
                        )
        raw_text = "\n\n".join(raw_parts).strip()
        if not raw_text:
            raw_text = " ".join(word["text"] for word in words)
        return TranscriptionResult(raw_text=raw_text, word_chunks=words)

    def diarize_audio(
        self,
        audio_path: str,
        num_speakers_hint: int | float | None = 0,
        duration: float | None = None,
    ) -> list[dict[str, float | str]]:
        try:
            requested_speakers = int(num_speakers_hint or 0)
        except (TypeError, ValueError):
            requested_speakers = 0

        pipeline = self._ensure_diarization_pipeline()
        if pipeline is None:
            end = float(duration) if duration else 0.0
            if end <= 0:
                end = 9999.0
            return [{"speaker": "SPEAKER_00", "start": 0.0, "end": end}]

        params = (
            {"num_speakers": requested_speakers}
            if requested_speakers > 0
            else {"min_speakers": 1, "max_speakers": self.config.max_speakers}
        )
        try:
            diarization = pipeline({"audio": audio_path}, **params)
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    {
                        "speaker": speaker,
                        "start": float(turn.start),
                        "end": float(turn.end),
                    }
                )
            return segments
        except Exception:
            end = float(duration) if duration else 0.0
            if end <= 0:
                end = 9999.0
            return [{"speaker": "SPEAKER_00", "start": 0.0, "end": end}]

    def align_and_reassemble(
        self,
        word_chunks: list[dict[str, Any]],
        speaker_segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not word_chunks:
            return []

        segments = sorted(
            speaker_segments,
            key=lambda segment: (segment.get("start", 0.0), segment.get("end", 0.0)),
        )
        segment_index = 0
        for word in word_chunks:
            start, end = word["timestamp"]
            if start is None or end is None:
                word["speaker"] = "UNKNOWN"
                continue

            while segment_index < len(segments) and segments[segment_index]["end"] <= start:
                segment_index += 1

            best_speaker = "UNKNOWN"
            best_overlap = 0.0
            probe = segment_index
            while probe < len(segments) and segments[probe]["start"] < end:
                overlap = max(
                    0.0,
                    min(end, segments[probe]["end"]) - max(start, segments[probe]["start"]),
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = str(segments[probe]["speaker"])
                probe += 1
            word["speaker"] = best_speaker

        current = {
            "speaker": word_chunks[0].get("speaker", "UNKNOWN"),
            "start_time": word_chunks[0]["timestamp"][0],
            "end_time": word_chunks[0]["timestamp"][1],
            "text": word_chunks[0]["text"],
        }
        turns = []
        for word in word_chunks[1:]:
            if not word.get("text"):
                continue
            start, end = word["timestamp"]
            if start is None or end is None:
                continue
            if word["speaker"] == current["speaker"]:
                current["text"] += " " + word["text"]
                current["end_time"] = end
            else:
                turns.append(current)
                current = {
                    "speaker": word["speaker"],
                    "start_time": start,
                    "end_time": end,
                    "text": word["text"],
                }
        turns.append(current)

        merge_gap_sec = 0.3
        smoothed = []
        for turn in turns:
            if (
                smoothed
                and turn["speaker"] == smoothed[-1]["speaker"]
                and (turn["start_time"] - smoothed[-1]["end_time"]) <= merge_gap_sec
            ):
                smoothed[-1]["text"] += " " + turn["text"]
                smoothed[-1]["end_time"] = turn["end_time"]
            else:
                smoothed.append(turn)
        return smoothed

    def format_transcript_for_llm(self, turns: list[dict[str, Any]]) -> str:
        lines = []
        for turn in turns:
            timestamp = format_hms(turn.get("start_time"))
            speaker = turn.get("speaker", "UNKNOWN")
            text = " ".join(str(turn.get("text", "")).split())
            lines.append(f"[{timestamp}] {speaker}: {text}")
        return "\n".join(lines)
