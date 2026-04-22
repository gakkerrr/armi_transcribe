# armi_transcribe

Минимальный скрипт для транскрибации аудио/видео через Whisper.

## Быстрый результат

После `git clone` запуск одной командой:

`python transcribe_media.py <input_file> <output_file> <memory_gb>`

Пример:

`python transcribe_media.py "input.mp4" "result.txt" 16`

## Полная инструкция с нуля (Windows)

### 1) Установить Python

1. Скачайте Python 3.9+ с [python.org](https://www.python.org/downloads/windows/).
2. При установке включите галочку `Add Python to PATH`.
3. Проверьте в PowerShell:

`python --version`

Если команда не найдена, перезапустите терминал и проверьте снова.

### 2) Клонировать проект

В сmd:

`git clone https://github.com/gakkerrr/armi_transcribe`

`cd armi_transcribe`

### 3) Запустить скрипт

Запуск:

`python transcribe_media.py "путь_к_входному_файлу" "путь_к_выходному_файлу.txt" 16`

Примеры:

`python transcribe_media.py "testmp3.mp3" "result.txt" 16`

`python transcribe_media.py "video.mp4" "out/transcript.txt" 16`

## Что делает скрипт автоматически

- Устанавливает недостающие зависимости: `numpy`, `openai-whisper`, `imageio-ffmpeg`.
- Использует встроенный `ffmpeg` из `imageio-ffmpeg` (не требуется отдельная ручная установка `ffmpeg`).
- Если входной файл видео, конвертирует его в mp3.
- Сохраняет mp3-кэш в `.audio_cache/` и при следующем запуске использует готовый mp3, не конвертируя заново.
- Выбирает модель Whisper по аргументу `memory_gb`, который вы передаете при запуске.

## Важные замечания

- При первом запуске нужен интернет (чтобы скачать Python-пакеты и модель Whisper).
- Если на машине нет CUDA/GPU, скрипт использует CPU.
- Первый запуск может занять больше времени, последующие обычно быстрее.