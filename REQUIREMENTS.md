# Требования и установка (Windows)

Необходимое ПО
- Python 3.12+
- FFmpeg (консольные утилиты ffmpeg/ffprobe в PATH)

Способы установки
1) Python
- Скачать с python.org или через winget:
  winget install Python.Python.3.12
- Создать окружение и поставить зависимости:
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt

2) FFmpeg
- Через winget (рекомендуется):
  winget install Gyan.FFmpeg
- Альтернатива: загрузить сборку с https://www.gyan.dev/ffmpeg/builds/ и добавить путь bin в переменную PATH.

Примечания
- Приложение использует системный ffmpeg/ffprobe через subprocess (без всплывающих окон).
- Для сборки exe используйте скрипт scripts/build.ps1.
