from importlib.resources import files
from pathlib import Path

import torch
from huggingface_hub import HfApi, HfFolder
from safetensors.torch import save_file


def convert_checkpoints(directory):
    """Convert all .pt files in directory to .safetensors format."""
    directory = Path(directory)
    safetensor_files = []

    # Iterate through all .pt files in directory
    for pt_file in directory.glob("*.pt"):
        safetensor_path = pt_file.with_suffix(".safetensors")

        # Skip if safetensor file already exists
        if safetensor_path.exists():
            safetensor_files.append(safetensor_path)
            continue

        try:
            # Load the .pt file
            checkpoint = torch.load(pt_file, map_location="cpu")
            model_state_dict = checkpoint["model_state_dict"]

            # Save to .safetensors
            save_file(model_state_dict, safetensor_path)
            safetensor_files.append(safetensor_path)
            print(f"Converted {pt_file.name} to {safetensor_path.name}")
        except Exception as e:
            print(f"Error converting {pt_file.name}: {e}")

    return safetensor_files


def upload_model_to_hf(checkpoint_directory, remote_directory_name, vocab_txt_path, repo_name, token):
    """Upload model checkpoints and vocab file to HuggingFace."""
    HfFolder.save_token(token)

    # Create the repository (if it doesn't exist)
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)
        print(f"Repository {repo_name} created successfully.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Convert checkpoints to safetensors format
    safetensor_files = convert_checkpoints(checkpoint_directory)

    # Upload each safetensor file
    for model_path in safetensor_files:
        try:
            remote_path = f"{remote_directory_name}/{model_path.name}"
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=remote_path,
                repo_id=repo_name,
                repo_type="model",
                token=token,
            )
            print(f"Model file {model_path.name} uploaded successfully to {remote_path}")
        except Exception as e:
            print(f"Error uploading model file {model_path.name}: {e}")

    # Upload the vocab.txt file
    try:
        vocab_path = Path(vocab_txt_path)
        api.upload_file(
            path_or_fileobj=str(vocab_path),
            path_in_repo=vocab_path.name,
            repo_id=repo_name,
            repo_type="model",
            token=token,
        )
        print(f"Vocab file {vocab_path.name} uploaded successfully.")
    except Exception as e:
        print(f"Error uploading vocab file: {e}")


if __name__ == "__main__":
    # Paths to files
    vocab_txt_path = files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt")
    # Name of the model repository
    repo_name = "aihpi/F5-TTS-German"
    # The Hugging Face token
    token = "..."

    # Upload all safetensor checkpoints from directory to F5TTS_Base/*.safetensors and F5TTS_Base_bigvgan/*.safetensors
    upload_model_to_hf("/raid/shared/models/tts-demo/f5-tts/german/vocos/", "F5TTS_Base", vocab_txt_path, repo_name,
                       token)
    upload_model_to_hf("/raid/shared/models/tts-demo/f5-tts/german/bigvgan/", "F5TTS_Base_bigvgan", vocab_txt_path,
                       repo_name, token)
