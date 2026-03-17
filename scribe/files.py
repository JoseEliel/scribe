from __future__ import annotations

import os
import tempfile
from typing import Any


def format_hms(seconds: float | int | None) -> str:
    total = 0.0 if seconds is None else float(seconds)
    hours, remainder = divmod(int(total), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _extract_existing_original_path(candidate: Any) -> str:
    if not isinstance(candidate, str):
        return ""
    expanded = os.path.expanduser(candidate)
    if not os.path.isabs(expanded):
        return ""
    normalized = os.path.abspath(expanded)
    if not os.path.exists(normalized):
        return ""
    if not os.path.isfile(normalized):
        return ""
    return normalized


def _normalize_single_file_item(item: Any) -> dict[str, str] | None:
    path = ""
    original_candidates: list[Any] = []
    if isinstance(item, str):
        path = item
    elif hasattr(item, "name"):
        path = getattr(item, "name", "") or ""
        original_candidates.extend(
            [
                getattr(item, "orig_name", None),
                getattr(item, "original_path", None),
            ]
        )
    elif isinstance(item, dict):
        path = item.get("name") or item.get("path") or ""
        original_candidates.extend(
            [
                item.get("orig_path"),
                item.get("orig_name"),
                item.get("original_path"),
                item.get("original_file_path"),
            ]
        )
    if not isinstance(path, str) or not path:
        return None

    original_path = ""
    for candidate in original_candidates:
        original_path = _extract_existing_original_path(candidate)
        if original_path:
            break

    return {"path": path, "original_path": original_path}


def normalize_file_items(file_input: Any) -> list[dict[str, str]]:
    if file_input is None:
        return []
    if isinstance(file_input, list):
        items = []
        for raw in file_input:
            item = _normalize_single_file_item(raw)
            if item:
                items.append(item)
        return items
    item = _normalize_single_file_item(file_input)
    return [item] if item else []


def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as handle:
            return handle.read()
    except OSError:
        return ""


def append_section_text(existing: str, title: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return existing

    sections: list[tuple[str, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    for line in existing.splitlines():
        if line.startswith("=== ") and line.endswith(" ==="):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[4:-4].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))

    for index, (section_title, _) in enumerate(sections):
        if section_title == title:
            sections[index] = (title, body)
            break
    else:
        sections.append((title, body))

    return "\n\n".join(f"=== {name} ===\n{content}" for name, content in sections if content)


def append_markdown_section(existing: str, title: str, body: str, level: int = 2) -> str:
    body = (body or "").strip()
    if not body:
        return existing

    marker_prefix = "<!-- SECTION:"
    sections: list[tuple[str, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    for line in existing.splitlines():
        if line.startswith(marker_prefix) and line.endswith("-->"):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[len(marker_prefix):-3].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))

    for index, (section_title, _) in enumerate(sections):
        if section_title == title:
            sections[index] = (title, body)
            break
    else:
        sections.append((title, body))

    heading = "#" * max(1, level)
    blocks = []
    for name, content in sections:
        if content:
            blocks.append(f"<!-- SECTION: {name} -->\n{heading} {name}\n\n{content}")
    return "\n\n".join(blocks)


def is_gradio_temp_path(path: str) -> bool:
    if not path:
        return False
    real_path = os.path.realpath(path)
    parts = [part.lower() for part in real_path.split(os.sep) if part]
    if "gradio" not in parts:
        return False
    tmp_root = os.path.realpath(tempfile.gettempdir())
    try:
        return os.path.commonpath([real_path, tmp_root]) == tmp_root
    except ValueError:
        return False


def resolve_output_directory(
    audio_path: str,
    default_save_dir: str,
    output_dir: str = "",
    original_path: str = "",
) -> tuple[str, str]:
    requested = (output_dir or "").strip()
    if requested:
        directory = os.path.abspath(os.path.expanduser(requested))
        try:
            os.makedirs(directory, exist_ok=True)
            return directory, "custom"
        except OSError:
            pass

    original_dir = os.path.dirname(original_path) if original_path else ""
    if original_dir and os.path.isdir(original_dir):
        return original_dir, "original_audio_dir"

    audio_dir = os.path.dirname(os.path.abspath(audio_path)) if audio_path else ""
    if audio_dir and os.path.isdir(audio_dir):
        if is_gradio_temp_path(audio_dir):
            try:
                os.makedirs(default_save_dir, exist_ok=True)
                return default_save_dir, "default_save_dir_gradio_upload"
            except OSError:
                return os.getcwd(), "cwd_fallback_gradio_temp"
        return audio_dir, "audio_dir"

    try:
        os.makedirs(default_save_dir, exist_ok=True)
        return default_save_dir, "default_save_dir"
    except OSError:
        return os.getcwd(), "cwd_fallback"


def build_diar_save_path(
    audio_path: str,
    idx: int,
    default_save_dir: str,
    output_dir: str = "",
    original_path: str = "",
) -> tuple[str, str]:
    directory, mode = resolve_output_directory(
        audio_path,
        default_save_dir,
        output_dir=output_dir,
        original_path=original_path,
    )
    name_source = original_path or audio_path
    base = os.path.splitext(os.path.basename(name_source))[0] or f"file_{idx}"
    safe_base = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in base)
    candidate = os.path.join(directory, f"{safe_base}_diarized.txt")
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{safe_base}_diarized_{counter}.txt")
        counter += 1
    return candidate, mode

