from __future__ import annotations

from typing import Any, Iterable

from .config import DEFAULT_LLM


OLLAMA_TO_HF_MODEL_MAP = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma3": "google/gemma-2-27b-it",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
}

TOKENIZER_CACHE: dict[str, Any] = {}


EXECUTIVE_SUMMARY_PROMPT = """You are an expert meeting analyst.

Summarize the meeting for a busy stakeholder using markdown with these sections:
- Overview
- Key Decisions
- Risks or Blockers
- Next Steps

Keep it concise and specific.

Transcript:
{{transcript}}"""


DETAILED_SUMMARY_PROMPT = """You are a meticulous meeting analyst.

Write a detailed markdown summary organized by major discussion topics. For each topic include:
- What was discussed
- Important viewpoints or tradeoffs
- Decisions or unresolved questions

Transcript:
{{transcript}}"""


ACTION_ITEMS_PROMPT = """You are a project manager extracting follow-up work from a meeting.

Produce markdown grouped by owner. For each action item include:
- Task
- Owner
- Due date if mentioned
- Notes if the owner or due date is unclear

If ownership is ambiguous, place the item under "Unassigned".

Transcript:
{{transcript}}"""


def _extract_model_name(model_info: Any) -> str:
    if isinstance(model_info, dict):
        return model_info.get("model") or model_info.get("name") or ""
    return getattr(model_info, "model", "") or getattr(model_info, "name", "") or ""


def fetch_ollama_models() -> tuple[list[str], str]:
    try:
        import ollama
    except Exception as exc:
        return [], str(exc)

    try:
        model_data = ollama.list()
        models_data: Iterable[Any]
        if isinstance(model_data, dict):
            models_data = model_data.get("models", []) or []
        else:
            models_data = getattr(model_data, "models", []) or []
        names = []
        for item in models_data:
            name = _extract_model_name(item).strip()
            if name:
                names.append(name)
        return list(dict.fromkeys(names)), ""
    except Exception as exc:
        return [], str(exc)


def choose_default_model(models: list[str], preferred: str = DEFAULT_LLM) -> str | None:
    if not models:
        return None
    if preferred in models:
        return preferred
    return models[0]


def build_llm_status_message(models: list[str], error: str) -> str:
    if models:
        return f"LLM status: ready ({len(models)} model(s) detected)."
    if error:
        return f"LLM status: unavailable ({error})."
    return "LLM status: unavailable (no local Ollama models found)."


def _get_tokenizer(hf_model_name: str):
    if hf_model_name not in TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        TOKENIZER_CACHE[hf_model_name] = AutoTokenizer.from_pretrained(hf_model_name)
    return TOKENIZER_CACHE[hf_model_name]


def count_tokens_for_model(
    text: str,
    model_name: str,
    enable_remote_tokenizer_lookup: bool = False,
) -> int:
    if not enable_remote_tokenizer_lookup:
        return max(1, len(text) // 4)
    base = model_name.split(":")[0]
    hf_model_name = OLLAMA_TO_HF_MODEL_MAP.get(base)
    if not hf_model_name:
        return max(1, len(text) // 4)
    try:
        tokenizer = _get_tokenizer(hf_model_name)
        return len(tokenizer.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def build_prompt(transcript_text: str, template: str) -> str:
    text = template or ""
    if "{{transcript}}" in text:
        return text.replace("{{transcript}}", transcript_text)
    if "{transcript}" in text:
        return text.replace("{transcript}", transcript_text)
    text = text.rstrip()
    if not text:
        return transcript_text
    return f"{text}\n\nTranscript:\n{transcript_text}"


def stream_ollama_analysis(
    prompt: str,
    model: str,
    enable_remote_tokenizer_lookup: bool = False,
):
    import ollama

    prompt_tokens = count_tokens_for_model(
        prompt,
        model,
        enable_remote_tokenizer_lookup=enable_remote_tokenizer_lookup,
    )
    options = {"temperature": 0.3, "num_ctx": min(131072, prompt_tokens + 4096)}
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
            stream=True,
        )
        buffer: list[str] = []
        for chunk in response:
            delta = ""
            try:
                delta = chunk.get("message", {}).get("content", "") or ""
            except Exception:
                pass
            if delta:
                buffer.append(delta)
                if len(buffer) >= 16:
                    yield "".join(buffer)
                    buffer = []
        if buffer:
            yield "".join(buffer)
    except Exception as exc:
        message = (
            f"Error connecting to Ollama: {exc}. "
            f"Ensure Ollama is running and the model '{model}' is pulled."
        )
        raise RuntimeError(message) from exc

