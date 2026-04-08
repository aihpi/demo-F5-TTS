"""KugelAudio inference model for speech generation.

This is the open-source inference implementation without optimizations.
Based on the original VibeVoice model architecture.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import re
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import modeling_utils
from transformers.cache_utils import DynamicCache
from transformers.generation import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
)

from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel, AutoModelForCausalLM

from ..configs import KugelAudioConfig
from ..schedule.dpm_solver import DPMSolverMultistepScheduler
from .diffusion_head import KugelAudioDiffusionHead
from .kugelaudio_model import KugelAudioModel, KugelAudioPreTrainedModel
from .tokenizer import (
    KugelAudioTokenizerEncoderOutput,
    KugelAudioTokenizerStreamingCache,
)


if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]


def _get_cache_tensors(cache) -> Tuple[List, List]:
    """Get key and value cache tensors from a cache object."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache, cache.value_cache
    raise AttributeError(f"Cannot get cache tensors from {type(cache).__name__}")


@dataclass
class KugelAudioCausalLMOutputWithPast(BaseModelOutputWithPast):
    logits: Optional[torch.FloatTensor] = None


@dataclass
class KugelAudioGenerationOutput(ModelOutput):
    """Output type for KugelAudio generation."""

    sequences: torch.LongTensor = None
    speech_outputs: Optional[List[torch.FloatTensor]] = None


class KugelAudioTokenConstraintProcessor(LogitsProcessor):
    """Constrains token generation to only valid tokens during speech generation."""

    def __init__(self, valid_token_ids: List[int], device: torch.device = None):
        self.valid_token_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.valid_token_ids] = 0
        scores = scores + mask
        return scores


class KugelAudioForConditionalGenerationInference(KugelAudioPreTrainedModel, GenerationMixin):
    """KugelAudio model for inference with speech generation capabilities."""

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = KugelAudioModel(config)
        self.lm_head = nn.Linear(
            config.decoder_config.hidden_size,
            config.decoder_config.vocab_size,
            bias=False,
        )
        self.ddpm_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps
        self.post_init()

    @property
    def noise_scheduler(self):
        return self.model.noise_scheduler

    @property
    def prediction_head(self):
        return self.model.prediction_head

    @property
    def speech_scaling_factor(self):
        return self.model.speech_scaling_factor

    @property
    def speech_bias_factor(self):
        return self.model.speech_bias_factor

    @property
    def acoustic_tokenizer(self):
        return self.model.acoustic_tokenizer

    @property
    def acoustic_connector(self):
        return self.model.acoustic_connector

    # PATCH: expose semantic_tokenizer and semantic_connector as top-level properties for extracting psody/emotion embeddings in encode_voice_prompt() and _process_speech_inputs().
    @property
    def semantic_tokenizer(self):
        return self.model.semantic_tokenizer

    @property
    def semantic_connector(self):
        return self.model.semantic_connector

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = (
            num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps
        )

    # PATCH: encode_voice_prompt() added — encode raw audio into reusable voice_cache dict for on-the-fly cloning without pre-encoded .pt files.
    @torch.no_grad()
    def encode_voice_prompt(
        self,
        voice_audio: Union[torch.Tensor, str],
        sample_rate: int = 24000,
    ) -> dict:
        """Encode a wav file or audio tensor into a reusable voice_cache dict.

        This runs the audio through both the acoustic and semantic encoders and
        returns a dict that can be passed as voice_cache to generate(), or saved
        to disk as a .pt file for later reuse.

        Args:
            voice_audio: Path to a .wav file (str) or audio tensor
                         [batch, channels, samples] / [channels, samples] / [samples]
            sample_rate: Sample rate of the input audio (default 24000)

        Returns:
            Dict with keys: acoustic_mean, acoustic_std, semantic_mean,
                            audio_length, sample_rate
        """
        if getattr(self.acoustic_tokenizer, "encoder", None) is None:
            raise RuntimeError(
                "Cannot encode voice prompt: acoustic tokenizer encoder has been stripped. "
                "Do not call strip_encoders() if you need voice cloning."
            )

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Load from file if string path given
        if isinstance(voice_audio, str):
            from kugelaudio_open.processors.audio_processor import AudioProcessor
            audio_proc = AudioProcessor(sampling_rate=sample_rate)
            voice_audio = audio_proc(voice_audio, return_tensors="pt")["audio"]

        # Normalise shape to [batch, channels, samples]
        if voice_audio.dim() == 1:
            voice_audio = voice_audio.unsqueeze(0).unsqueeze(0)
        elif voice_audio.dim() == 2:
            voice_audio = voice_audio.unsqueeze(1)

        # Move to CPU float32 to avoid bfloat16/float16 dtype mismatch in conv layers for INT4 version.
        acoustic_enc = self.acoustic_tokenizer.encoder
        semantic_enc = self.semantic_tokenizer.encoder

        voice_cpu = voice_audio.cpu().float()
        acoustic_enc_cpu = acoustic_enc.to(device="cpu", dtype=torch.float32)
        semantic_enc_cpu = semantic_enc.to(device="cpu", dtype=torch.float32)

        acoustic_output = self.acoustic_tokenizer.encode(voice_cpu)
        semantic_output = self.semantic_tokenizer.encode(voice_cpu)

        # Restore encoders to original device/dtype
        acoustic_enc.to(device=device, dtype=dtype)
        semantic_enc.to(device=device, dtype=dtype)

        acoustic_std = getattr(acoustic_output, "std", self.acoustic_tokenizer.fix_std)
        if isinstance(acoustic_std, torch.Tensor):
            acoustic_std = acoustic_std.cpu()

        return {
            "acoustic_mean": acoustic_output.mean.cpu(),
            "acoustic_std": acoustic_std,
            "semantic_mean": semantic_output.mean.cpu(),
            "audio_length": voice_audio.shape[-1],
            "sample_rate": sample_rate,
        }

    # PATCH: _process_speech_inputs() added - processing for on-the-fly, voice-cache, or no voice inputs, with scaling and time alignment.
    def _process_speech_inputs(
        self,
        speech_tensors: Optional[torch.Tensor] = None,
        speech_masks: Optional[torch.Tensor] = None,
        voice_cache: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process voice inputs through acoustic + semantic encoders or from cache.

        Three modes (in priority order):
          1. voice_cache  — pre-encoded dict, encoders not needed
          2. speech_tensors — raw audio tensor, encode on the fly
          3. neither — return dummy zero features

        Args:
            speech_tensors: Raw audio tensor [batch, channels, samples]
            speech_masks:   Boolean mask [batch, time_frames]
            voice_cache:    Pre-encoded dict with acoustic_mean / semantic_mean

        Returns:
            Tuple of (acoustic_features [batch, time, dim], speech_embeds [valid_frames, hidden])
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if voice_cache is not None:
            # Mode 1: pre-encoded voice cache 
            acoustic_mean = voice_cache["acoustic_mean"].to(device=device, dtype=dtype)
            fix_std = voice_cache.get("acoustic_std", self.acoustic_tokenizer.fix_std)
            # PATCH: use acoustic_mean directly without random noise — randn_like
            # caused voice inconsistency between calls with identical voice_cache.
            acoustic_features = acoustic_mean

            if "semantic_mean" in voice_cache:
                semantic_mean = voice_cache["semantic_mean"].to(device=device, dtype=dtype)
            else:
                # Legacy .pt files without semantic_mean — fall back to zeros
                semantic_mean = torch.zeros_like(acoustic_mean[..., :self.config.semantic_vae_dim])

        elif speech_tensors is not None:
            # Mode 2: encode on the fly from raw audio tensor
            if getattr(self.acoustic_tokenizer, "encoder", None) is None:
                raise RuntimeError(
                    "Cannot encode voice_prompt: encoders have been stripped. "
                    "Do not call strip_encoders() if you need on-the-fly voice cloning."
                )
            # Cast both audio AND encoder to CPU float32 to avoid the INT4 dtype
            # mismatch. Under INT4 loading, encoder conv bias tensors land in float16
            # (encoders are in skip_modules so BitsAndBytes skips them). Running on
            # CPU float32 sidesteps the float/half mismatch entirely.
            # Same fix as encode_voice_prompt() — keep both in sync.
            audio_cpu = speech_tensors.cpu().float()

            acoustic_enc = self.acoustic_tokenizer.encoder
            semantic_enc = self.semantic_tokenizer.encoder
            acoustic_enc.to(device="cpu", dtype=torch.float32)
            semantic_enc.to(device="cpu", dtype=torch.float32)

            acoustic_output = self.acoustic_tokenizer.encode(audio_cpu)
            semantic_output = self.semantic_tokenizer.encode(audio_cpu)

            # Restore encoders to original device/dtype
            acoustic_enc.to(device=device, dtype=dtype)
            semantic_enc.to(device=device, dtype=dtype)

            acoustic_features = acoustic_output.mean.to(device=device, dtype=dtype)
            semantic_mean = semantic_output.mean.to(device=device, dtype=dtype)

        else:
            # Mode 3: no voice input — dummy features
            acoustic_features = torch.zeros(
                1, 1, self.config.acoustic_vae_dim, device=device, dtype=dtype
            )
            semantic_mean = torch.zeros(
                1, 1, self.config.semantic_vae_dim, device=device, dtype=dtype
            )

        # Apply scaling
        if not torch.isnan(self.speech_scaling_factor):
            acoustic_features = (
                acoustic_features + self.speech_bias_factor
            ) * self.speech_scaling_factor

        # Align time dimensions between acoustic and semantic
        acoustic_len = acoustic_features.shape[1]
        semantic_len = semantic_mean.shape[1]
        if semantic_len < acoustic_len:
            pad = acoustic_len - semantic_len
            semantic_mean = torch.nn.functional.pad(semantic_mean, (0, 0, 0, pad))
        elif semantic_len > acoustic_len:
            semantic_mean = semantic_mean[:, :acoustic_len, :]

        # Build speech_masks if not provided
        if speech_masks is None:
            batch_size = acoustic_features.shape[0]
            seq_len = acoustic_features.shape[1]
            speech_masks = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # Project through connectors and combine
        acoustic_embed = self.acoustic_connector(acoustic_features)
        semantic_embed = self.semantic_connector(semantic_mean)
        combined_embed = acoustic_embed + semantic_embed

        # Index by speech_masks for direct insertion into inputs_embeds
        speech_embeds = combined_embed[speech_masks.cpu()]

        return acoustic_features, speech_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        voice_cache: Optional[dict] = None,
        speech_tensors: Optional[torch.Tensor] = None,
        speech_masks: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, slice] = 0,
        **kwargs,
    ) -> Union[Tuple, KugelAudioCausalLMOutputWithPast]:
        """Forward pass for the model."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Process voice inputs if provided (voice_cache takes priority over speech_tensors)
        if voice_cache is not None or speech_tensors is not None:
            _, speech_embeds = self._process_speech_inputs(
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                voice_cache=voice_cache,
            )
            if speech_input_mask is not None:
                inputs_embeds[speech_input_mask] = speech_embeds

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return KugelAudioCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def sample_speech_tokens(
        self, condition: torch.Tensor, neg_condition: torch.Tensor, cfg_scale: float = 3.0
    ) -> torch.Tensor:
        """Sample speech latents using diffusion with classifier-free guidance."""
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)

        if cfg_scale == 1.0:
            # No CFG - single forward pass
            speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim).to(condition)
            for t in self.model.noise_scheduler.timesteps:
                eps = self.model.prediction_head(
                    speech, t.repeat(speech.shape[0]).to(speech), condition=condition
                )
                speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
            return speech

        # With CFG - batched forward pass
        combined_condition = torch.cat([condition, neg_condition], dim=0).to(
            self.model.prediction_head.device
        )
        speech = torch.randn(combined_condition.shape[0], self.config.acoustic_vae_dim).to(
            combined_condition
        )

        for t in self.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.model.prediction_head(
                combined, t.repeat(combined.shape[0]).to(combined), condition=combined_condition
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample

        return speech[: len(speech) // 2]

    @torch.no_grad()
    def generate(
        self,
        text_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        voice_cache: Optional[dict] = None,
        speech_input_mask: Optional[torch.Tensor] = None,
        speech_tensors: Optional[torch.Tensor] = None,
        speech_masks: Optional[torch.Tensor] = None,
        voice_prompt: Optional[torch.Tensor] = None,  # PATCH: legacy alias for speech_tensors
        cfg_scale: float = 3.0,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        temperature: float = 1.0,
        speech_end_penalty: float = 0.0,
        min_speech_diffusion_tokens: int = 0,
        show_progress: bool = True,
        **kwargs,
    ) -> KugelAudioGenerationOutput:
        """Generate speech from text.

        Args:
            text_ids: Tokenized text input (from processor)
            input_ids: Alternative name for text_ids
            voice_cache: Pre-encoded voice features dict (from .pt file or encode_voice_prompt())
            speech_input_mask: Boolean mask indicating where to insert voice embeddings
            # PATCH: speech_tensors and voice_prompt added — allow raw audio to be passed directly to generate() for on-the-fly voice cloning without a pre-encoded .pt.
            speech_tensors: Raw audio tensor [batch, channels, samples] — encoded at runtime
            speech_masks: Boolean mask for speech_tensors frames
            voice_prompt: Legacy alias for speech_tensors (same behaviour)
            cfg_scale: Classifier-free guidance scale (higher = more faithful to text)
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to sample or use greedy decoding
            temperature: Sampling temperature
            speech_end_penalty: Logit penalty for speech_end_id to reduce premature cutoff
            min_speech_diffusion_tokens: Minimum diffusion tokens before allowing speech_end_id
            show_progress: Whether to show progress bar

        Returns:
            KugelAudioGenerationOutput with sequences and speech_outputs
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Handle input_ids vs text_ids
        if text_ids is None and input_ids is not None:
            text_ids = input_ids
        if text_ids is None:
            raise ValueError("text_ids or input_ids is required")

        # Handle voice_prompt legacy alias
        if voice_prompt is not None and speech_tensors is None:
            speech_tensors = voice_prompt

        text_ids = text_ids.to(device)
        batch_size = text_ids.shape[0]

        # Get special token IDs
        speech_start_id = getattr(self.config, "speech_start_id", None) or 151652
        speech_end_id = getattr(self.config, "speech_end_id", None) or 151653
        speech_diffusion_id = getattr(self.config, "speech_diffusion_id", None) or 151654
        eos_token_id = getattr(self.config.decoder_config, "eos_token_id", None) or 151643

        # Initialize streaming cache for acoustic tokenizer only
        acoustic_cache_streaming = KugelAudioTokenizerStreamingCache()

        # Initialize sequences and attention masks
        current_ids = text_ids
        attention_mask = torch.ones_like(current_ids)

        # For CFG, create negative prompt (just speech_start token)
        negative_ids = torch.full((batch_size, 1), speech_start_id, dtype=torch.long, device=device)
        negative_attention_mask = torch.ones_like(negative_ids)

        # Storage for generated audio and tracking
        audio_chunks = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=device)
        diffusion_token_count = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Get initial embeddings
        inputs_embeds = self.model.get_input_embeddings()(current_ids)

        # Process voice inputs if provided
        if voice_cache is not None or speech_tensors is not None:
            _, speech_embeds = self._process_speech_inputs(
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                voice_cache=voice_cache,
            )
            if speech_input_mask is not None:
                speech_input_mask = speech_input_mask.to(device)
                inputs_embeds[speech_input_mask] = speech_embeds

        negative_inputs_embeds = self.model.get_input_embeddings()(negative_ids)

        # Setup logits processor to constrain to valid tokens
        valid_tokens = [speech_start_id, speech_end_id, speech_diffusion_id, eos_token_id]
        token_constraint = KugelAudioTokenConstraintProcessor(valid_tokens, device=device)

        # Initialize KV caches
        past_key_values = None
        negative_past_key_values = None

        # Progress bar
        progress_iter = (
            tqdm(range(max_new_tokens), desc="Generating", leave=False)
            if show_progress
            else range(max_new_tokens)
        )

        for step in progress_iter:
            if finished.all():
                break

            # Forward pass for positive (main) model
            if past_key_values is None:
                outputs = self(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                outputs = self(
                    inputs_embeds=inputs_embeds[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            # Apply token constraint
            logits = token_constraint(current_ids, logits)

            # Penalize early speech_end to reduce truncation
            if speech_end_penalty and speech_end_penalty > 0:
                logits[:, speech_end_id] -= speech_end_penalty

            # Hard block speech_end until enough diffusion tokens have been generated
            if min_speech_diffusion_tokens and min_speech_diffusion_tokens > 0:
                too_early = diffusion_token_count < min_speech_diffusion_tokens
                if too_early.any():
                    logits[too_early, speech_end_id] = float("-inf")

            # Sample or greedy decode
            if do_sample and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)

            # Force finished samples to output EOS
            next_tokens = torch.where(
                finished, torch.tensor(eos_token_id, device=device), next_tokens
            )

            # Update sequences
            current_ids = torch.cat([current_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype),
                ],
                dim=-1,
            )

            # Check for EOS tokens
            eos_mask = (next_tokens == eos_token_id) & ~finished
            if eos_mask.any():
                finished = finished | eos_mask

            # Check for speech_end tokens - mark as finished and clear caches
            speech_end_mask = (next_tokens == speech_end_id) & ~finished
            if speech_end_mask.any():
                finished = finished | speech_end_mask
                speech_end_indices = speech_end_mask.nonzero(as_tuple=False).squeeze(-1)
                acoustic_cache_streaming.set_to_zero(speech_end_indices)

            # Handle speech_start tokens - refresh negative model KV cache
            speech_start_mask = (next_tokens == speech_start_id) & ~finished
            if (
                speech_start_mask.any()
                and cfg_scale != 1.0
                and negative_past_key_values is not None
            ):
                speech_start_indices = speech_start_mask.nonzero(as_tuple=False).squeeze(-1)
                if speech_start_indices.dim() == 0:
                    speech_start_indices = speech_start_indices.unsqueeze(0)

                for sample_idx in speech_start_indices.tolist():
                    negative_attention_mask[sample_idx, :] = 0
                    negative_attention_mask[sample_idx, -1] = 1

                    key_caches, value_caches = _get_cache_tensors(negative_past_key_values)
                    for k_cache, v_cache in zip(key_caches, value_caches):
                        k_cache[sample_idx, :, -1, :] = k_cache[sample_idx, :, 0, :].clone()
                        v_cache[sample_idx, :, -1, :] = v_cache[sample_idx, :, 0, :].clone()

                    negative_ids[sample_idx, -1] = speech_start_id

            # Prepare next input embeddings
            next_inputs_embeds = self.model.get_input_embeddings()(next_tokens).unsqueeze(1)

            # Handle diffusion tokens - generate speech
            diffusion_mask = (next_tokens == speech_diffusion_id) & ~finished
            if diffusion_mask.any():
                diffusion_token_count = diffusion_token_count + diffusion_mask.long()
                diffusion_indices = diffusion_mask.nonzero(as_tuple=False).squeeze(-1)
                if diffusion_indices.dim() == 0:
                    diffusion_indices = diffusion_indices.unsqueeze(0)

                # Run negative forward pass for CFG
                if cfg_scale != 1.0:
                    if negative_past_key_values is None:
                        neg_outputs = self(
                            inputs_embeds=negative_inputs_embeds,
                            attention_mask=negative_attention_mask,
                            use_cache=True,
                            return_dict=True,
                        )
                    else:
                        neg_outputs = self(
                            inputs_embeds=negative_inputs_embeds[:, -1:],
                            attention_mask=negative_attention_mask,
                            past_key_values=negative_past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
                    negative_past_key_values = neg_outputs.past_key_values

                    # Handle non-diffusion samples KV cache correction
                    non_diffusion_mask = ~diffusion_mask & ~finished
                    if non_diffusion_mask.any():
                        non_diffusion_indices = non_diffusion_mask.nonzero(as_tuple=False).squeeze(
                            -1
                        )
                        if non_diffusion_indices.dim() == 0:
                            non_diffusion_indices = non_diffusion_indices.unsqueeze(0)

                        key_caches, value_caches = _get_cache_tensors(negative_past_key_values)
                        for sample_idx in non_diffusion_indices.tolist():
                            start_idx = correct_cnt[sample_idx].item()
                            seq_len = negative_attention_mask.shape[1]

                            if start_idx + 1 < seq_len - 1:
                                negative_attention_mask[sample_idx, start_idx + 1 :] = (
                                    negative_attention_mask[sample_idx, start_idx:-1].clone()
                                )
                            negative_attention_mask[sample_idx, start_idx] = 0

                            for k_cache, v_cache in zip(key_caches, value_caches):
                                if start_idx + 1 < k_cache.shape[2] - 1:
                                    k_cache[sample_idx, :, start_idx + 1 :, :] = k_cache[
                                        sample_idx, :, start_idx:-1, :
                                    ].clone()
                                    v_cache[sample_idx, :, start_idx + 1 :, :] = v_cache[
                                        sample_idx, :, start_idx:-1, :
                                    ].clone()

                            if start_idx + 1 < negative_ids.shape[1] - 1:
                                negative_ids[sample_idx, start_idx + 1 :] = negative_ids[
                                    sample_idx, start_idx:-1
                                ].clone()

                        correct_cnt[non_diffusion_indices] += 1

                    neg_condition = neg_outputs.last_hidden_state[diffusion_indices, -1, :]
                else:
                    neg_condition = torch.zeros(
                        diffusion_indices.shape[0],
                        self.config.decoder_config.hidden_size,
                        device=device,
                        dtype=dtype,
                    )

                # Get conditioning from last hidden state
                condition = outputs.last_hidden_state[diffusion_indices, -1, :]

                # Sample speech latents using diffusion
                speech_latents = self.sample_speech_tokens(condition, neg_condition, cfg_scale)

                # Unscale latents
                scaled_latent = (
                    speech_latents / self.speech_scaling_factor - self.speech_bias_factor
                )

                # Decode through acoustic tokenizer with streaming cache
                audio = self.acoustic_tokenizer.decode(
                    scaled_latent.unsqueeze(1).permute(0, 2, 1),
                    cache=acoustic_cache_streaming,
                    sample_indices=diffusion_indices,
                    use_cache=True,
                )

                # Store audio chunks
                for i, idx in enumerate(diffusion_indices.tolist()):
                    if not finished[idx]:
                        audio_chunks[idx].append(audio[i].cpu())

                # Compute embeddings for next step (acoustic only, no semantic re-encoding)
                acoustic_embed = self.acoustic_connector(speech_latents.unsqueeze(1))
                diffusion_embeds = acoustic_embed.squeeze(1)

                # Update embeddings for diffusion samples
                next_inputs_embeds[diffusion_indices] = diffusion_embeds.unsqueeze(1)

            # Update embeddings for next iteration
            inputs_embeds = torch.cat([inputs_embeds, next_inputs_embeds], dim=1)

            # Update negative model
            negative_inputs_embeds = torch.cat([negative_inputs_embeds, next_inputs_embeds], dim=1)
            negative_attention_mask = torch.cat(
                [
                    negative_attention_mask,
                    torch.ones((batch_size, 1), device=device, dtype=negative_attention_mask.dtype),
                ],
                dim=-1,
            )
            negative_ids = torch.cat([negative_ids, next_tokens.unsqueeze(-1)], dim=-1)

        # Concatenate audio chunks with normalization
        speech_outputs = []
        for chunks in audio_chunks:
            if chunks:
                concatenated = torch.cat(chunks, dim=-1).squeeze()
                # Normalize audio to prevent clipping
                max_val = concatenated.abs().max()
                if max_val > 1.0:
                    concatenated = concatenated * (0.95 / max_val)
                # Apply watermark to all generated audio
                concatenated = self._apply_watermark(concatenated, sample_rate=24000)
                speech_outputs.append(concatenated)
            else:
                speech_outputs.append(None)

        return KugelAudioGenerationOutput(
            sequences=current_ids,
            speech_outputs=speech_outputs,
        )

    @staticmethod
    def load_voice(voice_path: str) -> dict:
        """Load a pre-encoded voice from a .pt file.

        Args:
            voice_path: Path to the .pt file containing pre-encoded voice features.
                The file should contain a dict with at least "acoustic_mean" tensor.

        Returns:
            Dict suitable for passing as voice_cache to generate().
        """
        voice_cache = torch.load(voice_path, map_location="cpu", weights_only=True)
        if "acoustic_mean" not in voice_cache:
            raise ValueError(
                f"Voice file {voice_path} does not contain 'acoustic_mean'. "
                f"Available keys: {list(voice_cache.keys())}"
            )
        return voice_cache

    def _apply_watermark(self, audio: torch.Tensor, sample_rate: int = 24000) -> torch.Tensor:
        """Apply imperceptible watermark to generated audio.

        PATCH: watermark disabled — audioseal uses torch.compile which causes
        audible artifacts on PyTorch 2.9 + CUDA 12.6 (Kaggle environment) due
        to recompilation shape mismatches. Re-enable in production once
        environment is stable.
        """
        return audio  # PATCH: disabled for Kaggle compatibility
        try:
            import torchaudio.functional as F
            from audioseal import AudioSeal
        except ImportError:
            return audio  # Graceful fallback if audioseal not available

        device = audio.device
        dtype = audio.dtype
        original_shape = audio.shape

        # Prepare audio for watermarking (AudioSeal expects [batch, channels, samples] at 16kHz)
        if audio.dim() == 1:
            audio_for_wm = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio_for_wm = audio.unsqueeze(0)
        else:
            audio_for_wm = audio

        audio_for_wm = audio_for_wm.float()

        # Resample to 16kHz for AudioSeal
        if sample_rate != 16000:
            audio_16k = F.resample(audio_for_wm, sample_rate, 16000)
        else:
            audio_16k = audio_for_wm

        # Load watermark generator (cached after first use)
        if not hasattr(self, "_wm_generator"):
            self._wm_generator = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
            self._wm_generator.eval()

        # Generate and apply watermark
        with torch.no_grad():
            watermark_16k = self._wm_generator.get_watermark(audio_16k.to(device), 16000)

        # Resample watermark back to original sample rate
        if sample_rate != 16000:
            watermark = F.resample(watermark_16k, 16000, sample_rate)
            # Ensure same length
            if watermark.shape[-1] != audio_for_wm.shape[-1]:
                if watermark.shape[-1] > audio_for_wm.shape[-1]:
                    watermark = watermark[..., : audio_for_wm.shape[-1]]
                else:
                    watermark = torch.nn.functional.pad(
                        watermark, (0, audio_for_wm.shape[-1] - watermark.shape[-1])
                    )
        else:
            watermark = watermark_16k

        # Add watermark to audio
        watermarked = audio_for_wm + watermark.to(audio_for_wm.device)

        # Normalize to prevent clipping
        max_val = watermarked.abs().max()
        if max_val > 1.0:
            watermarked = watermarked * (0.95 / max_val)

        # Restore original shape
        if len(original_shape) == 1:
            watermarked = watermarked.squeeze(0).squeeze(0)
        elif len(original_shape) == 2:
            watermarked = watermarked.squeeze(0)

        return watermarked.to(dtype=dtype)


# Register with AutoModel
AutoModel.register(KugelAudioConfig, KugelAudioModel)
AutoModelForCausalLM.register(KugelAudioConfig, KugelAudioForConditionalGenerationInference)


__all__ = [
    "KugelAudioForConditionalGenerationInference",
    "KugelAudioCausalLMOutputWithPast",
    "KugelAudioGenerationOutput",
]