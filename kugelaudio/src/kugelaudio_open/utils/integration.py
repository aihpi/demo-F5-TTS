"""Kugel INT4 integration helpers for FastAPI and other runtimes."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any

import numpy as np
import torch

from kugelaudio_open.utils.generation import chunk_and_generate

# Deterministic defaults for Kugel INT4 behavior.
KUGEL_AUDIO_DDPM_STEPS = 10
KUGEL_AUDIO_CFG_SCALE = 2.4
KUGEL_AUDIO_SPEECH_END_PENALTY = 4.5
KUGEL_AUDIO_MIN_DIFFUSION_TOKENS = 12
KUGEL_AUDIO_SINGLE_PASS_MAX_CHARS = 320
KUGEL_AUDIO_ENABLE_TEXT_NORMALIZATION = os.getenv("KUGEL_AUDIO_ENABLE_TEXT_NORMALIZATION", "true").strip().lower() == "true"


def clone_voice_cache(voice_cache: dict[str, Any]) -> dict[str, Any]:
    """Deep-clone tensors and containers to isolate each generation call."""

    def _deep_clone(value: Any) -> Any:
        if hasattr(value, "clone"):
            try:
                return value.clone()
            except Exception:
                return value
        if isinstance(value, dict):
            return {k: _deep_clone(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_deep_clone(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_deep_clone(v) for v in value)
        return value

    return _deep_clone(voice_cache)


def compute_style_seed(ref_audio: bytes, text: str) -> int:
    """Derive style seed from voice+text for stable, expressive defaults."""
    payload = ref_audio + b"||" + (text or "").encode("utf-8", errors="ignore")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def normalize_text(text: str) -> str:
    """Normalize text and soften very short leading exclamations."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return normalized
    return re.sub(r"^([A-Za-zÄÖÜäöüß]{2,6})!\s+", r"\1, ", normalized)


def select_runtime_params(text: str, cfg_strength: float, f5_ui_default_cfg: float = 2.0) -> dict[str, float | int]:
    """Pick Kugel runtime params based on text length and cfg input."""
    text_len = len((text or "").strip())

    if text_len <= 220:
        base = {"max_chars": 300, "max_new_tokens": 2400}
    elif text_len <= 420:
        base = {"max_chars": 260, "max_new_tokens": 3400}
    else:
        base = {"max_chars": 240, "max_new_tokens": 4200}

    try:
        parsed_cfg = float(cfg_strength)
        requested_cfg = KUGEL_AUDIO_CFG_SCALE if abs(parsed_cfg - f5_ui_default_cfg) < 1e-6 else parsed_cfg
    except (TypeError, ValueError):
        requested_cfg = KUGEL_AUDIO_CFG_SCALE

    requested_cfg = max(0.1, min(5.0, requested_cfg))

    return {
        "ddpm_steps": KUGEL_AUDIO_DDPM_STEPS,
        "cfg_scale": requested_cfg,
        "max_chars": base["max_chars"],
        "max_new_tokens": base["max_new_tokens"],
    }


def generate_with_policy(
    *,
    model: Any,
    processor: Any,
    ref_audio: bytes,
    text: str,
    voice_cache: dict[str, Any],
    scenario_key: str,
    cfg_strength: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate Kugel audio with single/chunk policy, fallback and trimming."""
    raw_text = re.sub(r"\s+", " ", (text or "").strip())
    if not raw_text:
        raise RuntimeError("No text provided for KugelAudio generation")

    normalized_text = normalize_text(raw_text) if KUGEL_AUDIO_ENABLE_TEXT_NORMALIZATION else raw_text

    runtime_params = select_runtime_params(normalized_text, cfg_strength)
    ddpm_steps = int(runtime_params["ddpm_steps"])
    cfg_scale = float(runtime_params["cfg_scale"])
    max_chars = int(runtime_params["max_chars"])
    max_new_tokens = int(runtime_params["max_new_tokens"])
    model.set_ddpm_inference_steps(ddpm_steps)

    style_seed = compute_style_seed(ref_audio, normalized_text)
    text_len = len(normalized_text)

    use_single_pass_first = text_len <= KUGEL_AUDIO_SINGLE_PASS_MAX_CHARS
    primary = "single" if use_single_pass_first else "chunk"
    selected_mode = primary

    def _generate_single() -> Any:
        device = next(model.parameters()).device
        inputs = processor(text=normalized_text, voice_cache=voice_cache, model=model, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                cfg_scale=cfg_scale,
                max_new_tokens=max_new_tokens,
                speech_end_penalty=KUGEL_AUDIO_SPEECH_END_PENALTY,
                min_speech_diffusion_tokens=KUGEL_AUDIO_MIN_DIFFUSION_TOKENS,
            )
        return outputs.speech_outputs[0] if outputs.speech_outputs else None

    def _generate_chunked() -> Any:
        return chunk_and_generate(
            model=model,
            processor=processor,
            text=normalized_text,
            voice_cache=voice_cache,
            max_chars=max_chars,
            cfg_scale=cfg_scale,
            max_new_tokens=max_new_tokens,
            speech_end_penalty=KUGEL_AUDIO_SPEECH_END_PENALTY,
            min_speech_diffusion_tokens=KUGEL_AUDIO_MIN_DIFFUSION_TOKENS,
            silence_duration=0.10,
            sample_rate=24000,
        )

    try:
        torch.manual_seed(style_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(style_seed)
        audio_tensor = _generate_single() if primary == "single" else _generate_chunked()
    except Exception:
        if primary == "single":
            selected_mode = "chunk"
            audio_tensor = _generate_chunked()
        else:
            selected_mode = "single"
            audio_tensor = _generate_single()

    if audio_tensor is None:
        raise RuntimeError("KugelAudio generation failed - no audio output")

    audio_np = audio_tensor.cpu().float().numpy()
    audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)

    meta = {
        "selected_mode": selected_mode,
        "output_samples": int(audio_np.shape[-1]),
        "output_duration_s": round(float(audio_np.shape[-1]) / 24000.0, 3),
    }
    return audio_np, meta
