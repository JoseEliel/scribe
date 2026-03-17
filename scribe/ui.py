from __future__ import annotations

import html
import json
import os
import shutil
import tempfile

import gradio as gr

from .analysis import (
    ACTION_ITEMS_PROMPT,
    DETAILED_SUMMARY_PROMPT,
    EXECUTIVE_SUMMARY_PROMPT,
    build_llm_status_message,
    build_prompt,
    choose_default_model,
    fetch_ollama_models,
    stream_ollama_analysis,
)
from .config import AppConfig, load_config
from .files import (
    append_markdown_section,
    append_section_text,
    build_diar_save_path,
    normalize_file_items,
    read_text_file,
)
from .transcription import SpeechPipeline


COPY_HELPERS_HEAD = """
<script>
window.scribeGetTextboxValue = function (elemId) {
  const root = document.getElementById(elemId);
  if (!root) {
    return "";
  }
  const field = root.querySelector("textarea, input");
  return field ? (field.value || "") : "";
};

window.scribeCopyText = async function (text, label) {
  const value = typeof text === "string" ? text : "";
  if (!value.trim()) {
    return { ok: false, message: `Nothing to copy for ${label}.` };
  }

  const fallbackCopy = () => {
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    textarea.style.pointerEvents = "none";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    textarea.setSelectionRange(0, textarea.value.length);
    const copied = document.execCommand("copy");
    document.body.removeChild(textarea);
    if (!copied) {
      throw new Error("execCommand('copy') returned false");
    }
  };

  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try {
        await navigator.clipboard.writeText(value);
      } catch (err) {
        fallbackCopy();
      }
    } else {
      fallbackCopy();
    }
    return { ok: true, message: `Copied ${label}.` };
  } catch (err) {
    console.error(err);
    return {
      ok: false,
      message: `Could not copy ${label}. Select the text and use Cmd/Ctrl+C.`
    };
  }
};

window.scribeCopyFromTextbox = async function (elemId, statusId, label) {
  const text = window.scribeGetTextboxValue(elemId);
  const result = await window.scribeCopyText(text, label);
  const status = document.getElementById(statusId);
  if (status) {
    status.textContent = result.message;
    status.dataset.state = result.ok ? "success" : "error";
  }
};
</script>
"""


APP_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
<script>
window.scribeForceLightTheme = function () {
  const html = document.documentElement;
  const body = document.body;
  if (html) {
    if (html.dataset.theme !== "light") {
      html.dataset.theme = "light";
    }
    html.classList.remove("dark");
    html.classList.add("light");
    html.style.colorScheme = "light";
  }
  if (body) {
    body.classList.remove("dark");
    body.classList.add("light");
    body.style.colorScheme = "light";
  }
};
window.addEventListener("DOMContentLoaded", window.scribeForceLightTheme);
window.addEventListener("load", window.scribeForceLightTheme);
</script>
""" + COPY_HELPERS_HEAD


APP_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
).set(
    background_fill_primary="#f6efe5",
    background_fill_primary_dark="#f6efe5",
    background_fill_secondary="#fbf6ee",
    background_fill_secondary_dark="#fbf6ee",
    body_background_fill="#f6efe5",
    body_background_fill_dark="#f6efe5",
    body_text_color="#111111",
    body_text_color_dark="#111111",
    body_text_color_subdued="#4f4338",
    body_text_color_subdued_dark="#4f4338",
    block_background_fill="#fbf6ee",
    block_background_fill_dark="#fbf6ee",
    block_border_color="#111111",
    block_border_color_dark="#111111",
    block_label_background_fill="#fbf6ee",
    block_label_background_fill_dark="#fbf6ee",
    block_label_text_color="#111111",
    block_label_text_color_dark="#111111",
    input_background_fill="#fffaf2",
    input_background_fill_dark="#fffaf2",
    input_border_color="#111111",
    input_border_color_dark="#111111",
    input_placeholder_color="#6b5f54",
    input_placeholder_color_dark="#6b5f54",
    button_primary_background_fill="#f18a45",
    button_primary_background_fill_dark="#f18a45",
    button_primary_background_fill_hover="#ff9b59",
    button_primary_background_fill_hover_dark="#ff9b59",
    button_primary_text_color="#111111",
    button_primary_text_color_dark="#111111",
    button_primary_border_color="#111111",
    button_primary_border_color_dark="#111111",
    button_secondary_background_fill="#f7dcc2",
    button_secondary_background_fill_dark="#f7dcc2",
    button_secondary_background_fill_hover="#f2c79e",
    button_secondary_background_fill_hover_dark="#f2c79e",
    button_secondary_text_color="#111111",
    button_secondary_text_color_dark="#111111",
    button_secondary_border_color="#111111",
    button_secondary_border_color_dark="#111111",
)


CUSTOM_CSS = """
:root {
  --scribe-bg: #f6efe5;
  --scribe-paper: #fbf6ee;
  --scribe-panel: #f7dcc2;
  --scribe-panel-alt: #f18a45;
  --scribe-panel-muted: #eadfce;
  --scribe-panel-soft: #fffaf2;
  --scribe-ink: #111111;
  --scribe-danger: #e61b00;
  --scribe-shadow: 8px 8px 0 0 #111111;
  --scribe-border: 3px solid #111111;
  --body-text-color: #111111;
  --body-text-color-subdued: #111111;
  --block-label-text-color: #111111;
  --block-title-text-color: #111111;
  --input-text-color: #111111;
  color-scheme: light;
}

html, body {
  background:
    linear-gradient(90deg, rgba(241, 138, 69, 0.08) 1px, transparent 1px),
    linear-gradient(rgba(17, 17, 17, 0.04) 1px, transparent 1px),
    var(--scribe-bg);
  background-size: 28px 28px, 28px 28px, auto;
  color-scheme: light;
}

.gradio-container,
.gradio-container * {
  font-family: "Space Grotesk", "Helvetica Neue", Arial, sans-serif !important;
}

.gradio-container {
  background: transparent !important;
  color: var(--scribe-ink) !important;
  max-width: 1480px !important;
  padding-top: 18px !important;
  color-scheme: light !important;
}

div[data-testid="progress-bar"], .progress-bar {
  position: static !important;
  inset: auto !important;
  margin-top: 8px !important;
  z-index: 0 !important;
}

#raw_txt_out [data-testid="loader"],
#diar_txt_out [data-testid="loader"],
#exec_md_out [data-testid="loader"],
#exec_txt_out [data-testid="loader"],
#det_md_out [data-testid="loader"],
#det_txt_out [data-testid="loader"],
#act_md_out [data-testid="loader"],
#act_txt_out [data-testid="loader"] {
  opacity: 0 !important;
  pointer-events: none !important;
  z-index: -1 !important;
}

.scribe-header {
  border: var(--scribe-border);
  box-shadow: var(--scribe-shadow);
  background: linear-gradient(135deg, #f7e3bf 0 60%, var(--scribe-panel-alt) 60% 100%);
  color: var(--scribe-ink) !important;
  padding: 24px;
  margin-bottom: 18px;
}

.scribe-header p,
.scribe-header span,
.scribe-header strong,
.scribe-header__card,
.scribe-header__card * {
  color: var(--scribe-ink) !important;
}

.scribe-header__eyebrow {
  display: inline-block;
  border: var(--scribe-border);
  background: var(--scribe-paper);
  padding: 6px 10px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.scribe-header h1 {
  margin: 16px 0 10px;
  font-size: clamp(2.4rem, 6vw, 5rem);
  line-height: 0.94;
  letter-spacing: -0.05em;
  text-transform: uppercase;
  color: var(--scribe-ink) !important;
}

.scribe-header__grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
  margin-top: 18px;
}

.scribe-header__card {
  border: var(--scribe-border);
  background: rgba(255, 250, 242, 0.96);
  padding: 14px;
  box-shadow: 4px 4px 0 0 #111111;
}

.scribe-header__card strong {
  display: block;
  margin-bottom: 6px;
  text-transform: uppercase;
  font-size: 0.8rem;
  letter-spacing: 0.09em;
}

.gradio-container h2,
.gradio-container h3 {
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--scribe-ink) !important;
}

.gradio-container label,
.gradio-container legend,
.gradio-container [data-testid="block-info"],
.gradio-container [data-testid="file-upload-dropzone"],
.gradio-container [data-testid="file-upload-dropzone"] *,
#input_mode,
#input_mode *,
#run_llm_toggle,
#run_llm_toggle *,
#save_diar_toggle,
#save_diar_toggle *,
#audio_drop,
#audio_drop *,
#text_drop,
#text_drop * {
  color: var(--scribe-ink) !important;
}

.gradio-container .block,
.gradio-container .gr-block,
.gradio-container .gr-panel,
.gradio-container .gr-group,
.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-accordion,
.gradio-container fieldset {
  border-radius: 0 !important;
}

.gradio-container .gr-group,
.gradio-container .gr-panel,
.gradio-container .gr-form,
.gradio-container fieldset,
.gradio-container .gr-box,
.gradio-container .gr-accordion {
  border: var(--scribe-border) !important;
  box-shadow: var(--scribe-shadow);
  background: linear-gradient(180deg, var(--scribe-paper), #f7f0e6) !important;
  color: var(--scribe-ink) !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select,
.gradio-container [data-testid="textbox"],
.gradio-container [data-testid="number-input"],
.gradio-container [data-testid="dropdown"] {
  border: var(--scribe-border) !important;
  border-radius: 0 !important;
  box-shadow: 5px 5px 0 0 #111111;
  background: var(--scribe-panel-soft) !important;
  color: var(--scribe-ink) !important;
}

.gradio-container textarea:focus,
.gradio-container input:focus,
.gradio-container select:focus {
  outline: none !important;
  box-shadow: 7px 7px 0 0 #111111 !important;
}

.gradio-container button,
.gradio-container .lg,
.gradio-container .md,
.gradio-container .sm {
  border: var(--scribe-border) !important;
  border-radius: 0 !important;
  box-shadow: 6px 6px 0 0 #111111;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700 !important;
  transition: transform 0.08s ease;
}

.gradio-container button:hover {
  transform: translate(-2px, -2px);
}

.gradio-container button:active {
  transform: translate(2px, 2px);
  box-shadow: 2px 2px 0 0 #111111 !important;
}

.gradio-container button.primary,
.gradio-container .primary {
  background: var(--scribe-panel-alt) !important;
  color: var(--scribe-ink) !important;
}

.gradio-container button.secondary,
.gradio-container .secondary,
.copy-action {
  background: var(--scribe-panel) !important;
  color: var(--scribe-ink) !important;
}

.gradio-container [role="tablist"] {
  gap: 10px;
  background: transparent !important;
  margin-bottom: 10px;
}

.gradio-container [role="tab"] {
  border: var(--scribe-border) !important;
  border-radius: 0 !important;
  box-shadow: 5px 5px 0 0 #111111;
  background: #efe5d7 !important;
  color: var(--scribe-ink) !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700 !important;
}

.gradio-container [role="tab"][aria-selected="true"] {
  background: var(--scribe-panel) !important;
}

.gradio-container [role="tabpanel"] {
  border: var(--scribe-border) !important;
  box-shadow: var(--scribe-shadow);
  background: linear-gradient(180deg, var(--scribe-paper), #f8f1e7) !important;
  padding: 16px !important;
  color: var(--scribe-ink) !important;
}

.gradio-container button[aria-expanded] {
  border: var(--scribe-border) !important;
  border-radius: 0 !important;
  background: #efe5d7 !important;
  color: var(--scribe-ink) !important;
}

.gradio-container .prose {
  line-height: 1.5;
  color: var(--scribe-ink) !important;
}

.gradio-container .prose *,
.gradio-container [data-testid="markdown"],
.gradio-container [data-testid="markdown"] * {
  color: var(--scribe-ink) !important;
}

.gradio-container .prose blockquote {
  border-left: 8px solid #111111;
  padding-left: 12px;
  background: #fff5c5;
}

.gradio-container .prose code,
.gradio-container code {
  border: 2px solid #111111;
  border-radius: 0;
  background: #fff5c5;
  padding: 0.1rem 0.35rem;
}

.gradio-container .prose pre {
  border: var(--scribe-border);
  border-radius: 0 !important;
  background: #181818 !important;
  color: #fff4e6 !important;
  box-shadow: var(--scribe-shadow);
}

.gradio-container .prose pre *,
.gradio-container .prose code {
  color: inherit !important;
}

.gradio-container .prose th,
.gradio-container .prose td {
  border: 2px solid #111111;
  padding: 8px;
}

.gradio-container .prose th {
  background: var(--scribe-panel);
}

.scribe-status-note {
  border: var(--scribe-border);
  background: linear-gradient(90deg, #fff4d7, #f9e7c8);
  padding: 10px 12px;
  box-shadow: 4px 4px 0 0 #111111;
  font-size: 0.92rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--scribe-ink) !important;
}

.scribe-ribbon {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  border: var(--scribe-border);
  background: linear-gradient(90deg, #f6d5b9, #f8e4cf);
  color: var(--scribe-ink);
  padding: 12px 14px;
  margin: 16px 0 12px;
  box-shadow: var(--scribe-shadow);
}

.scribe-ribbon strong {
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.scribe-ribbon span:last-child {
  font-size: 0.95rem;
}

.scribe-ribbon--hot {
  background: var(--scribe-panel-alt);
  color: var(--scribe-ink);
}

.scribe-ribbon--hot .scribe-kicker {
  background: var(--scribe-paper);
  color: var(--scribe-ink);
}

.scribe-kicker {
  display: inline-block;
  border: var(--scribe-border);
  background: var(--scribe-panel);
  color: var(--scribe-ink);
  padding: 4px 8px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.scribe-panel-note {
  margin: 0 0 10px;
  font-size: 0.95rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--scribe-ink) !important;
}

.scribe-status-block {
  border: var(--scribe-border);
  background: linear-gradient(180deg, #fffaf2, #f6ead9);
  color: var(--scribe-ink) !important;
  padding: 12px 14px !important;
  box-shadow: var(--scribe-shadow);
}

.scribe-status-block p,
.scribe-status-block ul,
.scribe-status-block li {
  margin: 0 !important;
  color: inherit !important;
}

.scribe-status-block p,
.scribe-status-block li {
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 700;
}

.copy-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}

.copy-action {
  appearance: none;
  border: var(--scribe-border) !important;
  border-radius: 0 !important;
  background: var(--scribe-panel) !important;
  color: var(--scribe-ink) !important;
  box-shadow: 6px 6px 0 0 #111111;
  padding: 0.45rem 0.8rem;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
}

.copy-action:hover {
  background: var(--scribe-panel-alt) !important;
  color: var(--scribe-ink) !important;
}

.copy-status {
  min-height: 1.25rem;
  font-size: 0.92rem;
  color: rgba(17, 17, 17, 0.72);
}

.copy-status[data-state="success"] {
  color: var(--scribe-panel-alt);
}

.copy-status[data-state="error"] {
  color: var(--scribe-danger);
}

#audio_drop,
#text_drop {
  border: var(--scribe-border) !important;
  box-shadow: var(--scribe-shadow);
  background:
    linear-gradient(135deg, rgba(248, 218, 194, 0.72), rgba(241, 138, 69, 0.18)),
    var(--scribe-paper) !important;
  color: var(--scribe-ink) !important;
}

#run_btn {
  min-height: 72px;
  font-size: 1.05rem;
  letter-spacing: 0.14em;
  background: var(--scribe-panel-alt) !important;
  color: var(--scribe-ink) !important;
}

.gradio-container input[type="checkbox"],
.gradio-container input[type="radio"] {
  accent-color: var(--scribe-panel-alt);
}

.gradio-container [data-testid="checkbox"],
.gradio-container [data-testid="radio"],
.gradio-container [role="radiogroup"] {
  background: transparent !important;
  color: var(--scribe-ink) !important;
}

.gradio-container footer,
.gradio-container footer * {
  background: transparent !important;
  color: var(--scribe-ink) !important;
}

.gradio-container footer {
  border-top: var(--scribe-border);
  margin-top: 28px;
  padding-top: 12px;
}

#refresh_models_btn {
  width: 100%;
}

@media (max-width: 720px) {
  .scribe-header {
    padding: 18px;
  }

  .scribe-header h1 {
    font-size: 2.5rem;
  }
}
"""


def build_copy_toolbar(target_elem_id: str, status_id: str, button_text: str, label: str) -> str:
    onclick = (
        f"window.scribeCopyFromTextbox("
        f"{json.dumps(target_elem_id)}, {json.dumps(status_id)}, {json.dumps(label)})"
    )
    escaped_onclick = html.escape(onclick, quote=True)
    return (
        '<div class="copy-toolbar">'
        f'<button type="button" class="copy-action" onclick="{escaped_onclick}">'
        f"{html.escape(button_text)}</button>"
        f'<span id="{html.escape(status_id, quote=True)}" class="copy-status" '
        'role="status" aria-live="polite"></span>'
        "</div>"
    )


def build_input_help_markdown() -> str:
    return (
        '<section class="scribe-header">'
        '<div class="scribe-header__eyebrow">Local / Raw First / Brutalist</div>'
        "<h1>Scribe</h1>"
        "<p>Batch audio transcription, diarization, transcript review, and local LLM analysis without hiding the raw text behind a finished pipeline.</p>"
        '<div class="scribe-header__grid">'
        '<div class="scribe-header__card"><strong>Raw First</strong><span>The live raw transcript lands before diarization finishes so you can copy and ship it during long runs.</span></div>'
        '<div class="scribe-header__card"><strong>Power User Modes</strong><span>Switch between audio, transcript-only, or mixed ingestion without changing apps.</span></div>'
        '<div class="scribe-header__card"><strong>Local Model Stack</strong><span>Whisper + pyannote + Ollama, with clear failure states instead of silent soft-fallback UI.</span></div>'
        "</div>"
        "</section>"
    )


def build_section_banner(title: str, subtitle: str, tone: str = "") -> str:
    classes = "scribe-ribbon"
    if tone:
        classes += f" scribe-ribbon--{tone}"
    return (
        f'<div class="{html.escape(classes, quote=True)}">'
        '<span class="scribe-kicker">Interface</span>'
        f"<strong>{html.escape(title)}</strong>"
        f"<span>{html.escape(subtitle)}</span>"
        "</div>"
    )


def refresh_ollama_models_ui(run_llm_enabled: bool):
    models, error = fetch_ollama_models()
    default_model = choose_default_model(models)
    status_text = build_llm_status_message(models, error)
    if models:
        dropdown_update = gr.update(choices=models, value=default_model, interactive=True)
        run_llm_update = gr.update(value=run_llm_enabled)
        prompt_group_update = gr.update(visible=bool(run_llm_enabled))
    else:
        dropdown_update = gr.update(choices=[], value=None, interactive=False)
        run_llm_update = gr.update(value=False)
        prompt_group_update = gr.update(visible=False)
    return dropdown_update, run_llm_update, status_text, prompt_group_update


def update_visibility(input_mode: str, run_llm_enabled: bool, save_diar_enabled: bool):
    wants_audio = input_mode in {"Audio", "Audio + Transcript"}
    wants_text = input_mode in {"Transcript Only", "Audio + Transcript"}
    return (
        gr.update(visible=wants_audio),
        gr.update(visible=wants_text),
        gr.update(visible=wants_audio),
        gr.update(visible=bool(save_diar_enabled and wants_audio)),
        gr.update(visible=bool(run_llm_enabled)),
    )


def build_demo(config: AppConfig | None = None) -> gr.Blocks:
    config = config or load_config()
    pipeline = SpeechPipeline(config)
    ollama_models, ollama_poll_error = fetch_ollama_models()
    default_llm = choose_default_model(ollama_models, config.default_llm)
    initial_llm_status = build_llm_status_message(ollama_models, ollama_poll_error)

    def process_meeting_audio_streaming(
        input_mode,
        audio_input,
        diarized_txt_input,
        llm_model,
        num_speakers,
        run_llm,
        save_diar_txt,
        save_diar_dir,
        exec_summary_prompt,
        detailed_summary_prompt,
        action_items_prompt,
        progress=gr.Progress(track_tqdm=True),
    ):
        status = "Starting…"
        raw_txt = ""
        diar_txt = ""
        exec_md = exec_txt = ""
        det_md = det_txt = ""
        act_md = act_txt = ""
        saved_diar_files: list[str] = []
        download_diar_files: list[str] = []
        download_stage_dir = ""
        save_modes: list[str] = []
        item_errors: list[str] = []
        section_name_counts: dict[str, int] = {}

        def yield_all():
            yield (
                status,
                raw_txt,
                diar_txt,
                exec_md,
                exec_txt,
                det_md,
                det_txt,
                act_md,
                act_txt,
                download_diar_files,
            )

        def stage_download_file(path: str):
            nonlocal download_stage_dir
            try:
                if not download_stage_dir:
                    download_stage_dir = tempfile.mkdtemp(prefix="scribe_gradio_downloads_")
                name = os.path.basename(path) or "diarized.txt"
                stem, ext = os.path.splitext(name)
                candidate = os.path.join(download_stage_dir, name)
                counter = 1
                while os.path.exists(candidate):
                    candidate = os.path.join(download_stage_dir, f"{stem}_{counter}{ext}")
                    counter += 1
                shutil.copy2(path, candidate)
                download_diar_files.append(candidate)
            except OSError:
                save_modes.append("download_stage_error")

        def append_llm_section(section: str, label: str, content: str):
            nonlocal exec_md, exec_txt, det_md, det_txt, act_md, act_txt
            if section == "exec":
                exec_md = append_markdown_section(exec_md, label, content)
                exec_txt = append_section_text(exec_txt, label, content)
                return
            if section == "det":
                det_md = append_markdown_section(det_md, label, content)
                det_txt = append_section_text(det_txt, label, content)
                return
            act_md = append_markdown_section(act_md, label, content)
            act_txt = append_section_text(act_txt, label, content)

        def append_llm_skipped(prefix: str, label: str):
            skip_message = f"{prefix}: LLM analysis skipped."
            append_llm_section("exec", label, skip_message)
            append_llm_section("det", label, skip_message)
            append_llm_section("act", label, skip_message)

        def make_unique_label(base_name: str, fallback: str) -> str:
            name = (base_name or fallback).strip()
            count = section_name_counts.get(name, 0) + 1
            section_name_counts[name] = count
            return name if count == 1 else f"{name} ({count})"

        def record_item_error(prefix: str, label: str, message: str, step_progress):
            nonlocal status, raw_txt, diar_txt
            status = f"{prefix}: ❌ {message}"
            item_errors.append(f"{label}: {message}")
            step_progress(1.0, desc=status)
            raw_txt = append_section_text(raw_txt, label, f"ERROR: {message}")
            diar_txt = append_section_text(diar_txt, label, f"ERROR: {message}")
            append_llm_section("exec", label, f"ERROR: {message}")
            append_llm_section("det", label, f"ERROR: {message}")
            append_llm_section("act", label, f"ERROR: {message}")
            yield from yield_all()

        def stream_llm_sections(
            prefix: str,
            label: str,
            transcript: str,
            step_progress,
            section_progress: tuple[float, float, float],
        ) -> bool:
            nonlocal status
            sections = [
                ("exec", "Generating executive summary…", exec_summary_prompt, section_progress[0]),
                ("det", "Generating detailed summary…", detailed_summary_prompt, section_progress[1]),
                ("act", "Extracting action items…", action_items_prompt, section_progress[2]),
            ]
            had_error = False
            for key, label_status, prompt_template, fraction in sections:
                status = f"{prefix}: {label_status}"
                step_progress(fraction, desc=status)
                yield from yield_all()
                prompt = build_prompt(transcript, prompt_template)
                buffer: list[str] = []
                try:
                    for chunk in stream_ollama_analysis(
                        prompt,
                        llm_model,
                        enable_remote_tokenizer_lookup=config.enable_remote_tokenizer_lookup,
                    ):
                        buffer.append(chunk)
                        append_llm_section(key, label, "".join(buffer))
                        yield from yield_all()
                except Exception as exc:
                    had_error = True
                    error_text = f"ERROR: {exc}"
                    item_errors.append(f"{label}: {label_status.rstrip('…')} failed: {exc}")
                    append_llm_section(key, label, error_text)
                    status = f"{prefix}: ⚠️ {label_status.rstrip('…')} failed."
                    step_progress(fraction, desc=status)
                    yield from yield_all()
            return had_error

        audio_items = normalize_file_items(audio_input) if input_mode != "Transcript Only" else []
        text_items = normalize_file_items(diarized_txt_input) if input_mode != "Audio" else []
        effective_run_llm = bool(run_llm and llm_model)

        if not audio_items and not text_items:
            status = "Please provide at least one audio file or diarized transcript."
            yield from yield_all()
            return

        if run_llm and not llm_model:
            status = "No Ollama model is currently available. Continuing without LLM analysis."
            yield from yield_all()

        total_items = len(audio_items) + len(text_items)
        file_fraction = 1.0 / total_items if total_items else 1.0
        wav_path = None
        is_temp = False
        index = 0

        try:
            for audio_item in audio_items:
                index += 1
                wav_path = None
                is_temp = False
                audio_path = audio_item.get("path", "")
                original_path = audio_item.get("original_path", "")
                display_path = original_path or audio_path
                label = make_unique_label(os.path.basename(display_path), f"Audio File {index}")
                prefix = f"[{index}/{total_items}] {label}"

                def step_progress(fraction, desc):
                    base = (index - 1) * file_fraction
                    progress(min(1.0, base + fraction * file_fraction), desc=desc)

                try:
                    if not audio_path or not os.path.exists(audio_path):
                        status = f"{prefix}: ❌ Audio file could not be read."
                        step_progress(1.0, desc=status)
                        yield from yield_all()
                        continue

                    status = f"{prefix}: Optimizing audio…"
                    step_progress(0.05, desc=status)
                    yield from yield_all()
                    wav_path, is_temp = pipeline.preprocess_audio_to_wav(audio_path)

                    pipeline.empty_accelerator_cache()
                    status = f"{prefix}: Transcribing…"
                    step_progress(0.15, desc=status)
                    yield from yield_all()
                    transcription = pipeline.transcribe_audio(wav_path)
                    raw_txt = append_section_text(raw_txt, label, transcription.raw_text)
                    status = f"{prefix}: ✅ Raw transcript ready."
                    step_progress(0.4, desc=status)
                    yield from yield_all()

                    pipeline.empty_accelerator_cache()
                    status = f"{prefix}: Identifying speakers…"
                    step_progress(0.48, desc=status)
                    yield from yield_all()
                    duration = pipeline.get_audio_duration_seconds(wav_path)
                    segments = pipeline.diarize_audio(wav_path, num_speakers, duration=duration)
                    status = f"{prefix}: ✅ Diarization complete."
                    step_progress(0.62, desc=status)
                    yield from yield_all()

                    pipeline.empty_accelerator_cache()
                    status = f"{prefix}: Aligning transcript…"
                    step_progress(0.72, desc=status)
                    yield from yield_all()
                    turns = pipeline.align_and_reassemble(transcription.word_chunks, segments)
                    diarized_text = pipeline.format_transcript_for_llm(turns)
                    diar_txt = append_section_text(diar_txt, label, diarized_text)
                    status = f"{prefix}: ✅ Diarized transcript ready."
                    step_progress(0.8, desc=status)
                    yield from yield_all()

                    if save_diar_txt and diarized_text.strip():
                        path, save_mode = build_diar_save_path(
                            audio_path,
                            index,
                            config.default_save_dir,
                            output_dir=save_diar_dir,
                            original_path=original_path,
                        )
                        try:
                            with open(path, "w", encoding="utf-8") as handle:
                                handle.write(diarized_text)
                            saved_diar_files.append(path)
                            stage_download_file(path)
                            save_modes.append(save_mode)
                        except OSError as exc:
                            save_modes.append("save_error")
                            item_errors.append(f"{label}: Could not save diarized transcript: {exc}")
                            status = f"{prefix}: ⚠️ Could not save diarized transcript."
                        yield from yield_all()

                    if not diarized_text.strip():
                        status = f"{prefix}: Could not generate a diarized transcript."
                        step_progress(1.0, desc=status)
                        yield from yield_all()
                        continue

                    if not effective_run_llm:
                        append_llm_skipped(prefix, label)
                        status = f"{prefix}: ✅ Audio processing complete (LLM skipped)."
                        step_progress(1.0, desc=status)
                        yield from yield_all()
                        continue

                    llm_had_error = yield from stream_llm_sections(
                        prefix=prefix,
                        label=label,
                        transcript=diarized_text,
                        step_progress=step_progress,
                        section_progress=(0.84, 0.92, 0.98),
                    )
                    status = (
                        f"{prefix}: ✅ Done with LLM warnings."
                        if llm_had_error
                        else f"{prefix}: ✅ Done."
                    )
                    step_progress(1.0, desc=status)
                    yield from yield_all()
                except Exception as exc:
                    yield from record_item_error(prefix, label, f"Processing failed: {exc}", step_progress)
                finally:
                    if is_temp and wav_path and os.path.exists(wav_path):
                        try:
                            os.remove(wav_path)
                        except OSError:
                            pass

            for text_item in text_items:
                index += 1
                text_path = text_item.get("path", "")
                original_path = text_item.get("original_path", "")
                display_path = original_path or text_path
                label = make_unique_label(os.path.basename(display_path), f"Transcript {index}")
                prefix = f"[{index}/{total_items}] {label}"

                def step_progress(fraction, desc):
                    base = (index - 1) * file_fraction
                    progress(min(1.0, base + fraction * file_fraction), desc=desc)

                try:
                    status = f"{prefix}: Loading transcript…"
                    step_progress(0.12, desc=status)
                    yield from yield_all()

                    transcript_text = (read_text_file(text_path) or "").strip()
                    if not transcript_text:
                        status = f"{prefix}: ❌ Could not read transcript."
                        step_progress(1.0, desc=status)
                        yield from yield_all()
                        continue

                    raw_txt = append_section_text(raw_txt, label, transcript_text)
                    diar_txt = append_section_text(diar_txt, label, transcript_text)
                    status = f"{prefix}: ✅ Transcript loaded."
                    step_progress(0.28, desc=status)
                    yield from yield_all()

                    if not effective_run_llm:
                        append_llm_skipped(prefix, label)
                        status = f"{prefix}: ✅ Transcript review complete (LLM skipped)."
                        step_progress(1.0, desc=status)
                        yield from yield_all()
                        continue

                    llm_had_error = yield from stream_llm_sections(
                        prefix=prefix,
                        label=label,
                        transcript=transcript_text,
                        step_progress=step_progress,
                        section_progress=(0.5, 0.74, 0.92),
                    )
                    status = (
                        f"{prefix}: ✅ Done with LLM warnings."
                        if llm_had_error
                        else f"{prefix}: ✅ Done."
                    )
                    step_progress(1.0, desc=status)
                    yield from yield_all()
                except Exception as exc:
                    yield from record_item_error(prefix, label, f"Processing failed: {exc}", step_progress)

            if save_diar_txt:
                if saved_diar_files:
                    saved_dirs = ", ".join(
                        sorted({os.path.dirname(path) or os.getcwd() for path in saved_diar_files})
                    )
                    status += f" Saved diarized transcript files to: {saved_dirs}."
                    if "default_save_dir_gradio_upload" in save_modes:
                        status += f" Uploaded audio files were saved to default directory: {config.default_save_dir}."
                    if "default_save_dir" in save_modes:
                        status += f" Files were saved to default directory: {config.default_save_dir}."
                    if "save_error" in save_modes:
                        status += " Some files could not be written to disk."
                    if "download_stage_error" in save_modes:
                        status += " Some files could not be prepared for in-app download."
                else:
                    status += " No diarized transcript files were saved."
            if item_errors:
                preview = "; ".join(item_errors[:3])
                more = f" (+{len(item_errors) - 3} more)" if len(item_errors) > 3 else ""
                status += f" Completed with {len(item_errors)} file error(s): {preview}{more}"
            progress(1.0, desc=status)
            yield from yield_all()
        finally:
            if download_stage_dir and os.path.isdir(download_stage_dir):
                try:
                    shutil.rmtree(download_stage_dir)
                except OSError:
                    pass
            if is_temp and wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    with gr.Blocks(
        theme=APP_THEME,
        css=CUSTOM_CSS,
        head=APP_HEAD,
    ) as demo:
        gr.HTML(build_input_help_markdown())
        gr.HTML(f'<div class="scribe-status-note">{html.escape(pipeline.startup_note())}</div>')
        gr.HTML(
            build_section_banner(
                "Control Deck",
                "Expose ingestion, speaker routing, save behavior, and model controls up front.",
                tone="hot",
            )
        )

        with gr.Row():
            input_mode = gr.Radio(
                choices=["Audio", "Transcript Only", "Audio + Transcript"],
                value="Audio",
                label="Input Mode",
                elem_id="input_mode",
            )
            run_llm_input = gr.Checkbox(
                value=bool(ollama_models),
                label="Run LLM analysis",
                elem_id="run_llm_toggle",
            )
            save_diar_checkbox = gr.Checkbox(
                value=False,
                label="Save diarized transcripts",
                elem_id="save_diar_toggle",
            )

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group(visible=True) as audio_group:
                    audio_input = gr.File(
                        file_count="multiple",
                        file_types=["audio"],
                        type="filepath",
                        label="Audio Files",
                        elem_id="audio_drop",
                    )
                with gr.Group(visible=False) as text_group:
                    diarized_txt_upload = gr.File(
                        file_count="multiple",
                        file_types=[".txt"],
                        type="filepath",
                        label="Transcript Files (.txt)",
                        elem_id="text_drop",
                    )
                num_speakers_input = gr.Number(
                    label="Number of Speakers (Optional)",
                    value=0,
                    info="Leave at 0 for auto-detect. Only applies to audio input.",
                    precision=0,
                    visible=True,
                    elem_id="speaker_count",
                )
                save_dir_input = gr.Textbox(
                    label="Save Directory (optional)",
                    value="",
                    placeholder="Example: ~/Documents/Transcripts",
                    info=f"If blank, saves next to the source when possible. Browser uploads fall back to: {config.default_save_dir}.",
                    visible=False,
                    elem_id="save_dir_input",
                )
            with gr.Column():
                llm_model_input = gr.Dropdown(
                    label="Ollama Model",
                    choices=ollama_models,
                    value=default_llm if ollama_models else None,
                    info="Select a local Ollama model for analysis.",
                    elem_id="llm_model_input",
                )
                refresh_models_btn = gr.Button("Refresh Ollama Models", elem_id="refresh_models_btn")
                llm_status_md = gr.Markdown(initial_llm_status, elem_classes=["scribe-status-block"])

        run_btn = gr.Button("Run", variant="primary", elem_id="run_btn")

        with gr.Group(visible=bool(ollama_models)) as prompt_editor_group:
            with gr.Accordion("Advanced LLM Prompts", open=False):
                prompt_exec_summary = gr.Textbox(
                    label="Executive Summary Prompt",
                    value=EXECUTIVE_SUMMARY_PROMPT,
                    lines=8,
                )
                prompt_detailed_summary = gr.Textbox(
                    label="Detailed Summary Prompt",
                    value=DETAILED_SUMMARY_PROMPT,
                    lines=9,
                )
                prompt_action_items = gr.Textbox(
                    label="Action Items Prompt",
                    value=ACTION_ITEMS_PROMPT,
                    lines=9,
                )

        gr.HTML(
            build_section_banner(
                "Output Wall",
                "Raw transcript lands first, diarization follows, analysis stays separate and copyable.",
            )
        )
        status_md = gr.Markdown("Status: idle.", elem_classes=["scribe-status-block"])

        with gr.Tabs():
            with gr.Tab("Live Raw Transcript"):
                gr.HTML(
                    '<p class="scribe-panel-note">This updates as soon as transcription finishes, before diarization and summaries complete.</p>'
                )
                gr.HTML(
                    build_copy_toolbar(
                        "raw_txt_out",
                        "raw_copy_status",
                        "Copy Raw Transcript",
                        "raw transcript",
                    )
                )
                raw_txt_out = gr.Textbox(
                    label="Raw Transcript",
                    lines=18,
                    interactive=False,
                    show_copy_button=False,
                    elem_id="raw_txt_out",
                )
            with gr.Tab("Diarized Transcript"):
                gr.HTML(
                    '<p class="scribe-panel-note">Speaker-attributed transcript plus optional file downloads for each processed item.</p>'
                )
                gr.HTML(
                    build_copy_toolbar(
                        "diar_txt_out",
                        "diar_copy_status",
                        "Copy Diarized Transcript",
                        "diarized transcript",
                    )
                )
                diar_txt_out = gr.Textbox(
                    label="Diarized Transcript",
                    lines=20,
                    interactive=False,
                    show_copy_button=False,
                    elem_id="diar_txt_out",
                )
                diar_file_out = gr.File(
                    label="Download diarized transcripts (.txt per file)",
                    file_count="multiple",
                )
            with gr.Tab("AI Analysis"):
                gr.HTML(
                    '<p class="scribe-panel-note">Executive summary, detailed summary, and action items stay isolated so you can copy each output cleanly.</p>'
                )
                gr.Markdown("Executive Summary")
                exec_md_out = gr.Markdown(elem_id="exec_md_out")
                with gr.Accordion("Copy Executive Summary Text", open=False):
                    gr.HTML(
                        build_copy_toolbar(
                            "exec_txt_out",
                            "exec_copy_status",
                            "Copy Executive Summary",
                            "executive summary",
                        )
                    )
                    exec_txt_out = gr.Textbox(
                        label="Executive Summary Text",
                        lines=10,
                        interactive=False,
                        show_label=False,
                        show_copy_button=False,
                        elem_id="exec_txt_out",
                    )

                gr.Markdown("Detailed Summary")
                det_md_out = gr.Markdown(elem_id="det_md_out")
                with gr.Accordion("Copy Detailed Summary Text", open=False):
                    gr.HTML(
                        build_copy_toolbar(
                            "det_txt_out",
                            "det_copy_status",
                            "Copy Detailed Summary",
                            "detailed summary",
                        )
                    )
                    det_txt_out = gr.Textbox(
                        label="Detailed Summary Text",
                        lines=16,
                        interactive=False,
                        show_label=False,
                        show_copy_button=False,
                        elem_id="det_txt_out",
                    )

                gr.Markdown("Action Items")
                act_md_out = gr.Markdown(elem_id="act_md_out")
                with gr.Accordion("Copy Action Items Text", open=False):
                    gr.HTML(
                        build_copy_toolbar(
                            "act_txt_out",
                            "act_copy_status",
                            "Copy Action Items",
                            "action items",
                        )
                    )
                    act_txt_out = gr.Textbox(
                        label="Action Items Text",
                        lines=12,
                        interactive=False,
                        show_label=False,
                        show_copy_button=False,
                        elem_id="act_txt_out",
                    )

        run_btn.click(
            fn=process_meeting_audio_streaming,
            inputs=[
                input_mode,
                audio_input,
                diarized_txt_upload,
                llm_model_input,
                num_speakers_input,
                run_llm_input,
                save_diar_checkbox,
                save_dir_input,
                prompt_exec_summary,
                prompt_detailed_summary,
                prompt_action_items,
            ],
            outputs=[
                status_md,
                raw_txt_out,
                diar_txt_out,
                exec_md_out,
                exec_txt_out,
                det_md_out,
                det_txt_out,
                act_md_out,
                act_txt_out,
                diar_file_out,
            ],
            show_progress="minimal",
        )
        refresh_models_btn.click(
            fn=refresh_ollama_models_ui,
            inputs=[run_llm_input],
            outputs=[llm_model_input, run_llm_input, llm_status_md, prompt_editor_group],
        )
        input_mode.change(
            fn=update_visibility,
            inputs=[input_mode, run_llm_input, save_diar_checkbox],
            outputs=[audio_group, text_group, num_speakers_input, save_dir_input, prompt_editor_group],
        )
        run_llm_input.change(
            fn=update_visibility,
            inputs=[input_mode, run_llm_input, save_diar_checkbox],
            outputs=[audio_group, text_group, num_speakers_input, save_dir_input, prompt_editor_group],
        )
        save_diar_checkbox.change(
            fn=update_visibility,
            inputs=[input_mode, run_llm_input, save_diar_checkbox],
            outputs=[audio_group, text_group, num_speakers_input, save_dir_input, prompt_editor_group],
        )

    return demo
