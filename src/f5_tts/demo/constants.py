import os


# Define model constants
KUGEL_AUDIO_ENABLED = os.getenv("KUGEL_AUDIO_ENABLED", "false").strip().lower() == "true"
KUGEL_AUDIO_MODEL_ID_DE = os.getenv("KUGEL_AUDIO_MODEL_ID_DE", os.getenv("KUGEL_AUDIO_MODEL_ID", "kugel-audio-german-local"))
KUGEL_AUDIO_MODEL_ID_EN = os.getenv("KUGEL_AUDIO_MODEL_ID_EN", "kugel-audio-english-local")
KUGEL_AUDIO_HF_REPO = os.getenv("KUGEL_AUDIO_HF_REPO", "kugelaudio/kugelaudio-0-open")
KUGEL_AUDIO_HF_REPO_EN = os.getenv("KUGEL_AUDIO_HF_REPO_EN", KUGEL_AUDIO_HF_REPO)
KUGEL_AUDIO_PROCESSOR_PATH = os.getenv("KUGEL_AUDIO_PROCESSOR_PATH", "").strip() or None

MODEL_CHECKPOINTS = {
    # German vocos checkpoints
    "kisz-german-vocos": "hf://aihpi/F5-TTS-German/F5TTS_Base/model_365000.safetensors",
    # German bigvgan checkpoints
    "kisz-german-bigvgan": "hf://aihpi/F5-TTS-German/F5TTS_Base_bigvgan/model_430000.safetensors",
    # English checkpoints
    "original-english-vocos": "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt",
    "original-english-bigvgan": "hf://SWivid/F5-TTS/F5TTS_Base_bigvgan/model_1250000.pt"
}

if KUGEL_AUDIO_ENABLED:
    MODEL_CHECKPOINTS[KUGEL_AUDIO_MODEL_ID_DE] = KUGEL_AUDIO_HF_REPO
    MODEL_CHECKPOINTS[KUGEL_AUDIO_MODEL_ID_EN] = KUGEL_AUDIO_HF_REPO_EN

MODEL_METADATA = {
    # German vocos checkpoints
    "kisz-german-vocos": {
        "vocoder_name": "vocos",
        "language": "de"
    },
    # German bigvgan checkpoints
    "kisz-german-bigvgan": {
        "vocoder_name": "bigvgan",
        "language": "de"
    },
    # English checkpoints
    "original-english-vocos": {
        "vocoder_name": "vocos",
        "language": "en"
    },
    "original-english-bigvgan": {
        "vocoder_name": "bigvgan",
        "language": "en"
    }
}

if KUGEL_AUDIO_ENABLED:
    MODEL_METADATA[KUGEL_AUDIO_MODEL_ID_DE] = {
        "vocoder_name": None,
        "language": "de",
        "display_name": "kisz-german-kugelaudio",
        "model_type": "kugelaudio",
    }
    MODEL_METADATA[KUGEL_AUDIO_MODEL_ID_EN] = {
        "vocoder_name": None,
        "language": "en",
        "display_name": "kisz-english-kugelaudio",
        "model_type": "kugelaudio",
    }
