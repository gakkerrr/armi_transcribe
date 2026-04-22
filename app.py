import subprocess
import sys
from pathlib import Path


def ensure_python_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing missing package: {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


def select_model_name() -> str:
    # VRAM-based model selection for CUDA GPUs.
    # Thresholds are conservative to reduce out-of-memory risks.
    try:
        import torch  # Imported lazily to avoid mandatory dependency before install.
    except ImportError:
        return "small"

    if torch.cuda.is_available():
        total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
        total_vram_gb = total_vram_bytes / (1024**3)

        if total_vram_gb >= 15:
            return "large"
        if total_vram_gb >= 6:
            return "medium"
        if total_vram_gb >= 4:
            return "small"
        return "base"

    return "small"


def load_audio_with_imageio_ffmpeg(input_path: str, sample_rate: int = 16000):
    import imageio_ffmpeg
    import numpy as np

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        input_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    out = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def maybe_convert_video_to_mp3(input_path: str) -> tuple[str, bool]:
    import imageio_ffmpeg

    video_extensions = {
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".webm",
        ".flv",
        ".wmv",
        ".mpeg",
        ".mpg",
        ".m4v",
    }
    source = Path(input_path)
    if source.suffix.lower() not in video_extensions:
        return input_path, False

    cache_dir = Path.cwd() / ".audio_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_mp3_path = cache_dir / f"{source.stem}.mp3"
    if cached_mp3_path.exists():
        print(f"Found cached mp3, using: {cached_mp3_path}")
        return str(cached_mp3_path), True

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"Converting video to mp3: {cached_mp3_path}")
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "2",
        str(cached_mp3_path),
    ]
    subprocess.run(command, capture_output=True, check=True)
    return str(cached_mp3_path), True


def parse_cli_args() -> tuple[str, str]:
    if len(sys.argv) != 3:
        script_name = Path(sys.argv[0]).name
        raise SystemExit(f"Usage: python {script_name} <input_file> <output_file>")
    return sys.argv[1], sys.argv[2]


def main() -> None:
    ensure_python_package("numpy", "numpy")
    ensure_python_package("whisper", "openai-whisper")
    ensure_python_package("imageio_ffmpeg", "imageio-ffmpeg")

    import whisper

    model_name = select_model_name()
    print(f"Selected model: {model_name}")
    model = whisper.load_model(model_name)

    input_file, output_file = parse_cli_args()
    prepared_audio_path, input_was_video = maybe_convert_video_to_mp3(input_file)
    if input_was_video:
        print(f"Audio source for transcription: {prepared_audio_path}")

    audio = load_audio_with_imageio_ffmpeg(prepared_audio_path)
    result = model.transcribe(audio, fp16=False)
    print(f"Detected language: {result.get('language', 'unknown')}")
    transcription_text = result["text"].strip()
    print(transcription_text)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(transcription_text, encoding="utf-8")
    print(f"Saved transcription to: {output_path}")


if __name__ == "__main__":
    main()