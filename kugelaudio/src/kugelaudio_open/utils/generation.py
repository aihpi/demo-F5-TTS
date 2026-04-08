"""High-level generation utilities for KugelAudio."""

import os
import re
from typing import Optional, Union

import torch


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_model_int4(
    model_name_or_path: str = "kugelaudio/kugelaudio-0-open",
    processor_path: Optional[str] = None,
    device_map: Optional[dict] = None,
):
    """Load KugelAudio model with INT4 quantisation.

    The LLM backbone (Qwen2 transformer) is quantised to NF4 INT4 with double
    quantisation. All other modules are kept in bfloat16 for quality and
    compatibility.

    Args:
        model_name_or_path: HuggingFace model ID or local path for model weights
        processor_path: Path for processor config. Defaults to the local repo root.
        device_map: Device placement map. Defaults to {'model': 'cuda', 'lm_head': 'cuda'}

    Returns:
        Tuple of (model, processor)
    """
    from transformers import BitsAndBytesConfig
    from kugelaudio_open.models import KugelAudioForConditionalGenerationInference
    from kugelaudio_open.processors import KugelAudioProcessor

    if processor_path is None:
        processor_path = _REPO_ROOT

    if device_map is None:
        device_map = {"model": "cuda", "lm_head": "cuda"}

    skip_modules = [
        "acoustic_tokenizer",
        "semantic_tokenizer",
        "prediction_head",
        "acoustic_connector",
        "semantic_connector",
        "lm_head",
    ]

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=skip_modules,
    )

    model = KugelAudioForConditionalGenerationInference.from_pretrained(
        model_name_or_path,
        quantization_config=quant_config,
        device_map=device_map,
    )
    model.eval()
    processor = KugelAudioProcessor.from_pretrained(processor_path)

    return model, processor


def chunk_and_generate(
    model,
    processor,
    text: str,
    voice: Optional[str] = None,
    voice_prompt: Optional[Union[str, torch.Tensor]] = None,
    voice_cache: Optional[dict] = None,
    max_chars: int = 260,
    cfg_scale: float = 3.0,
    max_new_tokens: int = 2800,
    speech_end_penalty: float = 5.0,
    min_speech_diffusion_tokens: int = 8,
    silence_duration: float = 0.2,
    sample_rate: int = 24000,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Generate speech for long text by chunking at prosody-aware boundaries.

    Args:
        model: KugelAudio model
        processor: KugelAudio processor
        text: Full input text (any length)
        voice: Named pre-encoded voice
        voice_prompt: Path to .wav or audio tensor for on-the-fly cloning
        voice_cache: Pre-encoded voice dict
        max_chars: Maximum characters per chunk (default 260)
        cfg_scale: Classifier-free guidance scale
        max_new_tokens: Max tokens per chunk
        speech_end_penalty: Logit penalty on speech_end token
        min_speech_diffusion_tokens: Minimum diffusion tokens before speech_end is allowed
        silence_duration: Seconds of silence inserted between chunks
        sample_rate: Audio sample rate in Hz (default 24000)
        device: Device for generation (auto-detected from model if None)

    Returns:
        Stitched audio tensor of the full text
    """
    if device is None:
        device = next(model.parameters()).device

    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        raise RuntimeError("No text provided for chunked generation")

    text_len = len(normalized)

    # Relax chunking for short/medium inputs so scenario lines stay cohesive.
    if text_len <= max_chars + 60:
        effective_max_chars = min(420, max(max_chars, text_len))
    elif text_len <= (2 * max_chars) + 80:
        effective_max_chars = min(420, max(max_chars, (text_len // 2) + 24))
    else:
        effective_max_chars = max_chars

    def _append_with_word_fallback(target_chunks, fragment: str, char_limit: int):
        """Append a fragment, splitting by words only if strictly necessary."""
        fragment = fragment.strip()
        if not fragment:
            return
        if len(fragment) <= char_limit:
            target_chunks.append(fragment)
            return
        words = fragment.split()
        word_acc = ""
        for word in words:
            candidate = f"{word_acc} {word}".strip()
            if len(candidate) <= char_limit:
                word_acc = candidate
            else:
                if word_acc:
                    target_chunks.append(word_acc)
                word_acc = word
        if word_acc:
            target_chunks.append(word_acc)

    # Hard boundaries; fall back to soft (comma/colon) then word splitting.
    hard_units = re.split(r'(?<=[.!?;...])\s+|\n+', normalized)
    hard_units = [s.strip() for s in hard_units if s.strip()]
    if not hard_units:
        hard_units = [normalized]

    chunks = []
    current = ''
    for s in hard_units:
        if not current:
            current = s
            continue
        if len(current) + 1 + len(s) <= effective_max_chars:
            current = f'{current} {s}'
        else:
            if len(s) > effective_max_chars:
                chunks.append(current.strip())
                current = ''
                soft_parts = re.split(r'(?<=[,:])\s+', s)
                for part in soft_parts:
                    part = part.strip()
                    if not part:
                        continue
                    if len(part) <= effective_max_chars:
                        if not current:
                            current = part
                        elif len(current) + 1 + len(part) <= effective_max_chars:
                            current = f'{current} {part}'
                        else:
                            chunks.append(current.strip())
                            current = part
                        continue
                    _append_with_word_fallback(chunks, part, effective_max_chars)
            else:
                chunks.append(current.strip())
                current = s
    if current:
        chunks.append(current.strip())

    audio_segments = []
    for chunk in chunks:
        chunk_max_new_tokens = min(max_new_tokens, max(640, int(len(chunk) * 18)))
        inputs = processor(
            text=chunk,
            voice=voice,
            voice_prompt=voice_prompt,
            voice_cache=voice_cache,
            model=model,
            return_tensors='pt',
        )
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                cfg_scale=cfg_scale,
                max_new_tokens=chunk_max_new_tokens,
                speech_end_penalty=speech_end_penalty,
                min_speech_diffusion_tokens=min_speech_diffusion_tokens,
            )
        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio_segments.append(outputs.speech_outputs[0])

    if not audio_segments:
        raise RuntimeError("No audio generated for any chunk")

    silence = torch.zeros(
        int(silence_duration * sample_rate),
        device=audio_segments[0].device,
        dtype=audio_segments[0].dtype,
    )
    stitched = []
    for i, seg in enumerate(audio_segments):
        stitched.append(seg)
        if i < len(audio_segments) - 1:
            stitched.append(silence)

    return torch.cat(stitched, dim=-1)
