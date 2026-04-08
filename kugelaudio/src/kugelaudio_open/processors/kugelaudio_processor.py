"""Main processor for KugelAudio combining text and audio processing."""

import json
import math
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
)
from transformers.utils import TensorType, cached_file, logging

from kugelaudio_open.processors.audio_processor import AudioNormalizer, AudioProcessor



class KugelAudioProcessor:
    """Combined processor for KugelAudio text and audio.

    Wraps a text tokenizer and audio processor into a single interface
    for preparing inputs for KugelAudio models.

    Voice cloning from raw audio is not supported. Instead, pre-encoded
    voices are loaded from .pt files referenced in a voices.json registry.

    Example:
        >>> processor = KugelAudioProcessor.from_pretrained("kugelaudio/kugelaudio-0-open")
        >>> inputs = processor(text="Hello world", voice="default")
    """

    def __init__(
        self,
        tokenizer=None,
        audio_processor: Optional[AudioProcessor] = None,
        speech_compression_ratio: int = 3200,
        db_normalize: bool = True,
        voices_registry: Optional[Dict[str, Any]] = None,
        voices_dir: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor or AudioProcessor()
        self.speech_compression_ratio = speech_compression_ratio
        self.db_normalize = db_normalize
        self.audio_normalizer = AudioNormalizer() if db_normalize else None
        self.voices_registry = voices_registry or {}
        self.voices_dir = voices_dir
        self._model_name_or_path = model_name_or_path

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load processor from pretrained model.

        Args:
            pretrained_model_name_or_path: Model ID or local path

        Returns:
            KugelAudioProcessor instance
        """
        from kugelaudio_open.processors.text_tokenizer import KugelAudioTextTokenizer

        # Try to load config
        config_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
        config = None

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            try:
                config_file = cached_file(
                    pretrained_model_name_or_path, "preprocessor_config.json", **kwargs
                )
                with open(config_file, "r") as f:
                    config = json.load(f)
            except Exception:
                config = {
                    "speech_compression_ratio": 3200,
                    "db_normalize": True,
                }

        # Extract parameters
        speech_compression_ratio = config.get("speech_compression_ratio", 3200)
        db_normalize = config.get("db_normalize", True)

        # Load tokenizer
        lm_name = config.get("language_model_pretrained_name") or kwargs.pop(
            "language_model_pretrained_name", "Qwen/Qwen2.5-7B"
        )
        tokenizer = KugelAudioTextTokenizer.from_pretrained(lm_name, **kwargs)

        # Load audio processor
        if "audio_processor" in config:
            audio_config = config["audio_processor"]
            audio_processor = AudioProcessor(
                sampling_rate=audio_config.get("sampling_rate", 24000),
                normalize_audio=audio_config.get("normalize_audio", True),
                target_dB_FS=audio_config.get("target_dB_FS", -25),
            )
        else:
            audio_processor = AudioProcessor()

        # Load voices registry (voices.json)
        voices_registry = {}
        voices_dir = None
        voices_json_path = os.path.join(pretrained_model_name_or_path, "voices", "voices.json")
        if os.path.exists(voices_json_path):
            with open(voices_json_path, "r") as f:
                voices_registry = json.load(f)
            voices_dir = os.path.join(pretrained_model_name_or_path, "voices")
        else:
            try:
                voices_file = cached_file(
                    pretrained_model_name_or_path, "voices/voices.json", **kwargs
                )
                if voices_file:
                    with open(voices_file, "r") as f:
                        voices_registry = json.load(f)
                    voices_dir = os.path.dirname(voices_file)
            except Exception:
                pass

        return cls(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            speech_compression_ratio=speech_compression_ratio,
            db_normalize=db_normalize,
            voices_registry=voices_registry,
            voices_dir=voices_dir,
            model_name_or_path=pretrained_model_name_or_path,
        )

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """Save processor to directory."""
        os.makedirs(save_directory, exist_ok=True)

        config = {
            "processor_class": "KugelAudioProcessor",
            "speech_compression_ratio": self.speech_compression_ratio,
            "db_normalize": self.db_normalize,
            "audio_processor": {
                "feature_extractor_type": "AudioProcessor",
                "sampling_rate": getattr(self.audio_processor, "sampling_rate", 24000),
                "normalize_audio": getattr(self.audio_processor, "normalize_audio", True),
                "target_dB_FS": getattr(self.audio_processor, "target_dB_FS", -25),
            },
        }

        config_path = os.path.join(save_directory, "preprocessor_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def get_available_voices(self) -> List[str]:
        """Return list of available pre-encoded voice names."""
        return list(self.voices_registry.keys())

    # PATCH: load_voice_cache() updated — Option B fallback added.
    # Original behaviour: raise ValueError if .pt not found.
    # New behaviour: if .pt is missing but a .wav exists alongside it in voices_dir,
    # and a model is provided, encode on the fly via model.encode_voice_prompt(),
    # save the result as .pt for next time, and return it.
    # This means the first call with voice="Lasse_Kohlmeyer" transparently encodes
    # from the .wav — no separate setup step required. The model parameter is optional
    # so existing callers that don't pass it still get the original raise behaviour.
    def load_voice_cache(self, voice_name: str, model=None) -> dict:
        """Load a pre-encoded voice by name, encoding from .wav if .pt is missing.

        Lookup order:
          1. Local voices_dir — load .pt if present
          2. HuggingFace Hub — download .pt if model was loaded from HF
          3. Encode from .wav — if .pt not found but a matching .wav exists in
             voices_dir and a model is provided, encodes on the fly and saves
             the .pt for next time (Option B fallback)

        Args:
            voice_name: Name of the voice (must be in voices.json registry).
            model: Optional KugelAudio inference model. Required for the
                   encode-from-.wav fallback. If None and .pt is missing,
                   raises ValueError as before.

        Returns:
            Dict with acoustic_mean, acoustic_std, semantic_mean tensors.

        Raises:
            ValueError: If voice_name is not in registry, or .pt not found
                        and no .wav fallback is available.
        """
        if voice_name not in self.voices_registry:
            available = ", ".join(self.voices_registry.keys()) or "(none)"
            raise ValueError(f"Voice '{voice_name}' not found. Available voices: {available}")

        voice_info = self.voices_registry[voice_name]
        voice_file = voice_info["file"]
        voice_path = None

        # Strategy 1: Try local voices_dir
        if self.voices_dir:
            candidate = os.path.join(self.voices_dir, voice_file)
            if os.path.exists(candidate):
                voice_path = candidate

        # Strategy 2: Download from HuggingFace Hub if not found locally.
        # Skip if _model_name_or_path is a local directory — passing a local path
        # to cached_file() produces a nonsense HF URL and a useless warning.
        is_local = self._model_name_or_path and os.path.isdir(self._model_name_or_path)
        if voice_path is None and self._model_name_or_path and not is_local:
            try:
                voice_path = cached_file(
                    self._model_name_or_path,
                    f"voices/{voice_file}",
                )
            except Exception as e:
                pass

        # Strategy 3: Encode from .wav on the fly if .pt missing and model provided
        if voice_path is None and model is not None and self.voices_dir:
            # Use the 'sample' field from voices.json if present (points to samples/ subdir),
            # otherwise fall back to deriving .wav name from the .pt filename in voices/
            sample_rel = voice_info.get("sample")
            if sample_rel:
                wav_path = os.path.join(self.voices_dir, sample_rel)
            else:
                wav_filename = os.path.splitext(voice_file)[0] + ".wav"
                wav_path = os.path.join(self.voices_dir, wav_filename)
            if os.path.exists(wav_path):
                voice_cache = model.encode_voice_prompt(wav_path)
                # Save for next time so encoding only happens once
                pt_save_path = os.path.join(self.voices_dir, voice_file)
                torch.save(voice_cache, pt_save_path)
                return voice_cache

        if voice_path is None:
            raise ValueError(
                f"Could not find voice file '{voice_file}' for voice '{voice_name}'. "
                f"Checked local dir: {self.voices_dir}, "
                f"HuggingFace repo: {self._model_name_or_path}. "
                f"Tip: provide a model to encode from .wav automatically."
            )

        try:
            voice_cache = torch.load(voice_path, map_location="cpu", weights_only=True)
        except Exception:
            voice_cache = torch.load(voice_path, map_location="cpu", weights_only=False)

        if "acoustic_mean" not in voice_cache:
            raise ValueError(
                f"Voice file '{voice_file}' does not contain 'acoustic_mean'. "
                f"Available keys: {list(voice_cache.keys())}"
            )
        return voice_cache

    def __call__(
        self,
        text: Optional[str] = None,
        voice: Optional[str] = None,
        voice_cache: Optional[dict] = None,
        # PATCH: voice_prompt parameter added. This allows raw audio to be passed directly for on-the-fly voice cloning without needing a pre-encoded .pt file.
        voice_prompt: Optional[Union[str, torch.Tensor]] = None,
        # PATCH: model parameter added — passed to load_voice_cache() so it can
        # encode from .wav on the fly if the .pt file is missing (Option B fallback).
        model=None,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """Process text and optional voice input.

        Priority order: voice_cache > voice (named) > voice_prompt (raw audio)

        Args:
            text: Input text to synthesize
            voice: Name of a pre-encoded voice (from voices.json registry).
                   If the corresponding .pt file is missing, and model is provided,
                   it will be encoded from the matching .wav automatically (Option B).
            voice_cache: Pre-encoded voice features dict — highest priority
            voice_prompt: Path to a .wav file (str) or raw audio tensor for
                          on-the-fly voice cloning — lowest priority
            model: Optional KugelAudio inference model. Required for the
                   encode-from-.wav fallback when voice= is used and .pt is missing.
            padding: Padding strategy
            truncation: Truncation strategy
            max_length: Maximum sequence length
            return_tensors: Return format

        Returns:
            BatchEncoding with text_ids, speech_input_mask, and either
            voice_cache (for pre-encoded) or speech_tensors/speech_masks
            (for on-the-fly encoding).
        """
        if text is None:
            raise ValueError("Text input is required")

        # Special token IDs
        speech_start_id = 151652  # <|vision_start|> repurposed for speech
        speech_diffusion_id = 151654  # VAE token used as placeholder

        # Format text with proper template
        formatted_text = text.strip()
        if not formatted_text.startswith("Speaker"):
            formatted_text = f"Speaker 0: {formatted_text}"

        # Build the full prompt template matching the training format
        system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"

        # Start building tokens and speech_input_mask
        full_tokens = []
        speech_input_mask = []

        # System prompt tokens
        system_tokens = self.tokenizer.encode(system_prompt, add_special_tokens=False)
        full_tokens.extend(system_tokens)
        speech_input_mask.extend([False] * len(system_tokens))

        # Priority: voice_cache > voice (named) > voice_prompt (raw audio)
        loaded_voice_cache = voice_cache
        if voice is not None and loaded_voice_cache is None:
            # model passed here enables the .wav fallback if .pt is missing
            loaded_voice_cache = self.load_voice_cache(voice, model=model)

        # Handle voice_prompt (raw audio path) — only when no pre-encoded voice available
        speech_tensors = None
        speech_masks_out = None
        if voice_prompt is not None and loaded_voice_cache is None:
            if isinstance(voice_prompt, str):
                if not os.path.exists(voice_prompt):
                    raise FileNotFoundError(f"Voice prompt file not found: {voice_prompt}")
                audio_result = self.audio_processor(voice_prompt, return_tensors="pt")
                speech_tensors = audio_result["audio"]
            elif isinstance(voice_prompt, torch.Tensor):
                speech_tensors = voice_prompt
            else:
                raise ValueError(
                    f"voice_prompt must be a file path (str) or torch.Tensor, "
                    f"got {type(voice_prompt)}"
                )
            # Normalise to [batch, channels, samples]
            if speech_tensors.dim() == 1:
                speech_tensors = speech_tensors.unsqueeze(0).unsqueeze(0)
            elif speech_tensors.dim() == 2:
                speech_tensors = speech_tensors.unsqueeze(1)

        # Process pre-encoded voice if available
        if loaded_voice_cache is not None:
            acoustic_mean = loaded_voice_cache["acoustic_mean"]
            # Number of tokens is determined by the acoustic_mean time dimension
            if acoustic_mean.dim() == 3:
                num_voice_tokens = acoustic_mean.shape[1]
            elif acoustic_mean.dim() == 2:
                num_voice_tokens = acoustic_mean.shape[0]
            else:
                num_voice_tokens = 1

            # Voice input section with placeholder tokens
            voice_input_tokens = self.tokenizer.encode(" Voice input:\n", add_special_tokens=False)
            full_tokens.extend(voice_input_tokens)
            speech_input_mask.extend([False] * len(voice_input_tokens))

            # Speaker prefix for voice
            speaker_prefix = self.tokenizer.encode(" Speaker 0:", add_special_tokens=False)
            full_tokens.extend(speaker_prefix)
            speech_input_mask.extend([False] * len(speaker_prefix))

            # Add placeholder VAE tokens that will be replaced with speech embeddings
            full_tokens.extend([speech_diffusion_id] * num_voice_tokens)
            speech_input_mask.extend([True] * num_voice_tokens)

            # Newline after voice
            newline_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
            full_tokens.extend(newline_tokens)
            speech_input_mask.extend([False] * len(newline_tokens))

        elif speech_tensors is not None:
            # voice_prompt path: compute number of placeholder tokens from audio length
            n_samples = speech_tensors.shape[-1]
            num_voice_tokens = max(math.ceil(n_samples / self.speech_compression_ratio), 1)

            # Build speech_masks [1, num_voice_tokens] — all frames valid
            speech_masks_out = torch.ones(1, num_voice_tokens, dtype=torch.bool)

            # Voice input section with placeholder tokens
            voice_input_tokens = self.tokenizer.encode(" Voice input:\n", add_special_tokens=False)
            full_tokens.extend(voice_input_tokens)
            speech_input_mask.extend([False] * len(voice_input_tokens))

            speaker_prefix = self.tokenizer.encode(" Speaker 0:", add_special_tokens=False)
            full_tokens.extend(speaker_prefix)
            speech_input_mask.extend([False] * len(speaker_prefix))

            full_tokens.extend([speech_diffusion_id] * num_voice_tokens)
            speech_input_mask.extend([True] * num_voice_tokens)

            newline_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
            full_tokens.extend(newline_tokens)
            speech_input_mask.extend([False] * len(newline_tokens))

        # Text input section
        text_input_tokens = self.tokenizer.encode(" Text input:\n", add_special_tokens=False)
        full_tokens.extend(text_input_tokens)
        speech_input_mask.extend([False] * len(text_input_tokens))

        # Speaker text
        speaker_text_tokens = self.tokenizer.encode(
            f" {formatted_text}\n", add_special_tokens=False
        )
        full_tokens.extend(speaker_text_tokens)
        speech_input_mask.extend([False] * len(speaker_text_tokens))

        # Speech output section
        speech_output_tokens = self.tokenizer.encode(" Speech output:\n", add_special_tokens=False)
        full_tokens.extend(speech_output_tokens)
        speech_input_mask.extend([False] * len(speech_output_tokens))

        # Add speech_start token
        full_tokens.append(speech_start_id)
        speech_input_mask.append(False)

        result = BatchEncoding()
        result["text_ids"] = full_tokens
        result["speech_input_mask"] = speech_input_mask

        if return_tensors == "pt":
            result["text_ids"] = torch.tensor([full_tokens], dtype=torch.long)
            result["speech_input_mask"] = torch.tensor([speech_input_mask], dtype=torch.bool)

        # Include voice data in result
        if loaded_voice_cache is not None:
            result["voice_cache"] = loaded_voice_cache
        elif speech_tensors is not None:
            result["speech_tensors"] = speech_tensors
            if speech_masks_out is not None:
                result["speech_masks"] = speech_masks_out

        return result

    def batch_decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.decode(*args, **kwargs)

    def save_audio(self, audio, output_path: str = "output.wav", **kwargs) -> List[str]:
        """Save generated audio to file."""
        return self.audio_processor.save_audio(audio, output_path, **kwargs)

    @property
    def model_input_names(self) -> List[str]:
        """Return list of model input names."""
        tokenizer_names = getattr(self.tokenizer, "model_input_names", [])
        audio_names = getattr(self.audio_processor, "model_input_names", [])
        return list(
            dict.fromkeys(tokenizer_names + audio_names + ["speech_inputs", "speech_input_mask"])
        )