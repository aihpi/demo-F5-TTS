import os
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter
import concurrent.futures


def apply_phone_effect(input_path, output_path):
    """
    Apply telephone-like audio effects to make the audio sound like it's coming through a phone
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(input_path)

        # Apply telephone-like effects:

        # 1. Reduce quality to 8kHz (typical phone sample rate)
        audio = audio.set_frame_rate(8000)

        # 2. Convert to mono
        audio = audio.set_channels(1)

        # 3. Apply band-pass filter (300-3400 Hz is typical telephone bandwidth)
        audio = high_pass_filter(audio, 300)
        audio = low_pass_filter(audio, 3400)

        # 4. Reduce bit depth
        audio = audio.set_sample_width(2)

        # 5. Boost volume slightly and add distortion
        audio = audio + 3  # Increase volume by 3dB

        # Normalize audio to prevent clipping
        normalized_audio = audio.normalize()

        # Export with low quality settings for phone effect
        normalized_audio.export(
            output_path,
            format="mp3",
            bitrate="16k",
            parameters=[
                "-codec:a", "libmp3lame",
                "-q:a", "5",
                "-ac", "1"  # Force mono output
            ]
        )

        print(f'Processed: {os.path.basename(input_path)} -> {os.path.basename(output_path)}')
        return True

    except Exception as e:
        print(f'Error processing {input_path}: {str(e)}')
        return False


def process_directory(input_dir):
    input_dir = Path(input_dir)
    output_dir = input_dir / 'phone_effect'
    output_dir.mkdir(exist_ok=True)

    # Get all audio files
    audio_files = list(input_dir.glob('*.wav')) + list(input_dir.glob('*.mp3'))

    # Create output paths
    output_paths = [output_dir / f"{input_file.stem}_phone.mp3" for input_file in audio_files]

    print(f"Found {len(audio_files)} audio files to process")

    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(apply_phone_effect, str(input_file), str(output_file))
            for input_file, output_file in zip(audio_files, output_paths)
        ]

        # Wait for all files to be processed
        concurrent.futures.wait(futures)

    # Count successful conversions
    successful = sum(1 for future in futures if future.result())
    print("\nProcessing complete!")
    print(f"Successfully processed: {successful}/{len(audio_files)} files")
    print(f"Output directory: {output_dir}")


def main():
    input_dir = "out/315000"

    # Check if directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Directory not found: {input_dir}")
        return

    # Check for required libraries
    try:
        import pydub
    except ImportError:
        print("Error: pydub library not found. Please install it using:")
        print("pip install pydub")
        return

    print(f"Starting audio processing in: {input_dir}")
    process_directory(input_dir)


if __name__ == "__main__":
    main()