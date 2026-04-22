import subprocess
import sys
from pathlib import Path
import spacy

nlp = spacy.load("ru_core_news_sm")

def smart_paragraphs(text: str, max_sents_per_para: int = 4) -> str:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    paragraphs = []
    current_para = []

    for sent in sentences:
        # Если параграф уже достаточно большой, сохраняем и начинаем новый
        if len(current_para) >= max_sents_per_para:
            paragraphs.append(" ".join(current_para))
            current_para = []

        current_para.append(sent)

    if current_para:
        paragraphs.append(" ".join(current_para))

    return "\n\n".join(paragraphs)


def ensure_python_package(import_name: str, pip_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing missing package: {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


def select_model_name(memory_gb: float) -> str:
    # User-provided memory in GB controls model choice.
    if memory_gb >= 15:
        return "large"
    if memory_gb >= 6:
        return "medium"
    if memory_gb >= 4:
        return "small"
    return "base"


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


def parse_cli_args() -> tuple[str, str, float]:
    if len(sys.argv) != 4:
        script_name = Path(sys.argv[0]).name
        raise SystemExit(
            f"Usage: python {script_name} <input_file> <output_file> <memory_gb>"
        )
    try:
        memory_gb = float(sys.argv[3])
    except ValueError as exc:
        raise SystemExit("memory_gb must be a number, e.g. 16") from exc
    return sys.argv[1], sys.argv[2], memory_gb


def main() -> None:
    ensure_python_package("numpy", "numpy")
    ensure_python_package("spacy", "spacy")
    ensure_python_package("whisper", "openai-whisper")
    ensure_python_package("imageio_ffmpeg", "imageio-ffmpeg")

    import whisper

    input_file, output_file, memory_gb = parse_cli_args()
    model_name = select_model_name(memory_gb)
    print(f"Selected model: {model_name}")
    model = whisper.load_model(model_name)

    prepared_audio_path, input_was_video = maybe_convert_video_to_mp3(input_file)
    if input_was_video:
        print(f"Audio source for transcription: {prepared_audio_path}")

    audio = load_audio_with_imageio_ffmpeg(prepared_audio_path)
    result = model.transcribe(audio, fp16=False)
    print(f"Detected language: {result.get('language', 'unknown')}")
    transcription_text = result["text"].strip()
    
    paragraphed_text = smart_paragraphs(transcription_text)
    print(paragraphed_text)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(paragraphed_text, encoding="utf-8")
    print(f"Saved transcription to: {output_path}")


if __name__ == "__main__":
    main()
