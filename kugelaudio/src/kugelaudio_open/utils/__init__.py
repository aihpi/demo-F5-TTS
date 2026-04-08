"""Utility functions for KugelAudio."""

from kugelaudio_open.utils.generation import (
    chunk_and_generate,
    load_model_int4,
)
from kugelaudio_open.utils.integration import (
    KUGEL_AUDIO_CFG_SCALE,
    KUGEL_AUDIO_DDPM_STEPS,
    KUGEL_AUDIO_MIN_DIFFUSION_TOKENS,
    KUGEL_AUDIO_SINGLE_PASS_MAX_CHARS,
    KUGEL_AUDIO_SPEECH_END_PENALTY,
    clone_voice_cache,
    compute_style_seed,
    generate_with_policy,
    normalize_text,
    select_runtime_params,
)

__all__ = [
    "chunk_and_generate",
    "load_model_int4",
    "KUGEL_AUDIO_CFG_SCALE",
    "KUGEL_AUDIO_DDPM_STEPS",
    "KUGEL_AUDIO_MIN_DIFFUSION_TOKENS",
    "KUGEL_AUDIO_SINGLE_PASS_MAX_CHARS",
    "KUGEL_AUDIO_SPEECH_END_PENALTY",
    "clone_voice_cache",
    "compute_style_seed",
    "generate_with_policy",
    "normalize_text",
    "select_runtime_params",
]
