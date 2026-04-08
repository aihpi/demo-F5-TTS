"""KugelAudio - Open Source Text-to-Speech Model with Voice Cloning

Example (INT4, ~8GB VRAM):
    >>> from kugelaudio_open.utils import load_model_int4, chunk_and_generate
    >>> model, processor = load_model_int4("kugelaudio/kugelaudio-0-open")
    >>> audio = chunk_and_generate(model, processor, "Hello world!")
    >>> processor.save_audio(audio, "output.wav")
"""

__version__ = "0.1.0"

from .configs import (
    KugelAudioAcousticTokenizerConfig,
    KugelAudioConfig,
    KugelAudioDiffusionHeadConfig,
    KugelAudioSemanticTokenizerConfig,
)
from .models import (
    KugelAudioAcousticTokenizerModel,
    KugelAudioDiffusionHead,
    KugelAudioForConditionalGeneration,
    KugelAudioForConditionalGenerationInference,
    KugelAudioModel,
    KugelAudioPreTrainedModel,
    KugelAudioSemanticTokenizerModel,
)
from .processors import KugelAudioProcessor
from .schedule import DPMSolverMultistepScheduler

__all__ = [
    "__version__",
    "KugelAudioConfig",
    "KugelAudioAcousticTokenizerConfig",
    "KugelAudioSemanticTokenizerConfig",
    "KugelAudioDiffusionHeadConfig",
    "KugelAudioModel",
    "KugelAudioPreTrainedModel",
    "KugelAudioForConditionalGeneration",
    "KugelAudioForConditionalGenerationInference",
    "KugelAudioAcousticTokenizerModel",
    "KugelAudioSemanticTokenizerModel",
    "KugelAudioDiffusionHead",
    "DPMSolverMultistepScheduler",
    "KugelAudioProcessor",
]
