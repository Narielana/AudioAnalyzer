"""
Runtime hook: делает встроенные ffmpeg/ffprobe видимыми для pydub при запуске PyInstaller (onefile).

Назначение:
- Поиск FFmpeg/FFprobe в _MEIPASS (временная директория PyInstaller)
- Настройка переменных окружения FFMPEG_BINARY и FFPROBE_BINARY
- Конфигурация pydub для использования найденных бинарников

Диагностика:
- Все действия логируются в stdout/stderr
- При проблемах проверьте лог на сообщения [runtime_ffmpeg]
"""
import os
import sys
import logging

# Настройка логирования для runtime hook
# Важно: в runtime hook логирование может быть ещё не настроено,
# поэтому создаём свой handler с выводом в stderr
_logger = logging.getLogger("runtime_ffmpeg")

if not _logger.handlers:
    try:
        _handler = logging.StreamHandler(sys.stderr)
        _handler.setLevel(logging.DEBUG)
        _handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
        _logger.addHandler(_handler)
        _logger.setLevel(logging.DEBUG)
    except Exception as e:
        # Крайний случай: выводим напрямую в stderr
        try:
            sys.stderr.write(f"[runtime_ffmpeg] Failed to setup logging: {e}\n")
        except Exception:
            pass


def _find_in_meipass(names):
    """Поиск файла в _MEIPASS (директория распаковки PyInstaller).

    Параметры:
    - names: список имён файлов для поиска (например, ['ffmpeg.exe', 'ffmpeg'])

    Возвращает: полный путь к найденному файлу или None.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        _logger.debug("_MEIPASS not found - not running in PyInstaller bundle")
        return None

    _logger.debug("Searching in _MEIPASS: %s", meipass)

    # Ищем в корне и в подпапке bin/
    for rel in ("{0}", os.path.join("bin", "{0}")):
        for name in names:
            p = os.path.join(meipass, rel.format(name))
            if os.path.exists(p):
                _logger.info("Found %s at %s", name, p)
                return p
            else:
                _logger.debug("Not found: %s", p)

    return None


def _find_in_system_path(names):
    """Поиск файла в системном PATH.

    Параметры:
    - names: список имён файлов для поиска

    Возвращает: полный путь к найденному файлу или None.
    """
    for name in names:
        for path_dir in os.environ.get("PATH", "").split(os.pathsep):
            if not path_dir:
                continue
            cand = os.path.join(path_dir, name)
            if os.path.exists(cand):
                _logger.debug("Found %s in PATH: %s", name, cand)
                return cand
    return None


def _set_ff_tools():
    """Настроить пути к FFmpeg и FFprobe.

    Приоритет поиска:
    1. _MEIPASS (встроенные в exe бинарники)
    2. Переменные окружения FFMPEG_BINARY / FFPROBE_BINARY
    3. Системный PATH
    """
    _logger.info("Configuring FFmpeg/FFprobe paths...")

    # Ищем FFmpeg
    ff = _find_in_meipass(["ffmpeg.exe", "ffmpeg"])
    if not ff:
        ff = os.environ.get("FFMPEG_BINARY")
        if ff:
            _logger.debug("Using FFMPEG_BINARY from env: %s", ff)

    if not ff:
        ff = _find_in_system_path(["ffmpeg.exe", "ffmpeg"])

    # Ищем FFprobe
    fp = _find_in_meipass(["ffprobe.exe", "ffprobe"])
    if not fp:
        fp = os.environ.get("FFPROBE_BINARY")
        if fp:
            _logger.debug("Using FFPROBE_BINARY from env: %s", fp)

    if not fp:
        # Пробуем найти рядом с ffmpeg
        if ff and ff.lower().endswith("ffmpeg.exe"):
            sibling = ff[:-10] + "ffprobe.exe"
            if os.path.exists(sibling):
                fp = sibling
                _logger.debug("Found ffprobe sibling to ffmpeg: %s", fp)

    if not fp:
        fp = _find_in_system_path(["ffprobe.exe", "ffprobe"])

    # Логируем результаты
    _logger.info("FFmpeg path: %s", ff or "NOT FOUND")
    _logger.info("FFprobe path: %s", fp or "NOT FOUND")

    # Устанавливаем переменные окружения
    if ff:
        os.environ["FFMPEG_BINARY"] = ff
        _logger.debug("Set FFMPEG_BINARY=%s", ff)
    else:
        _logger.warning("FFmpeg not found! Audio processing will not work.")

    if fp:
        os.environ["FFPROBE_BINARY"] = fp
        _logger.debug("Set FFPROBE_BINARY=%s", fp)
    else:
        _logger.warning("FFprobe not found! Metadata reading may fail.")

    # Настраиваем pydub, если доступен
    try:
        from pydub import AudioSegment  # type: ignore
        if ff:
            AudioSegment.converter = ff
            _logger.debug("Set AudioSegment.converter=%s", ff)
        if fp:
            AudioSegment.ffprobe = fp
            _logger.debug("Set AudioSegment.ffprobe=%s", fp)
        _logger.info("pydub configured successfully")
    except ImportError:
        _logger.debug("pydub not available - skipping configuration")
    except Exception as e:
        _logger.warning("Failed to configure pydub: %s", e)

    # Проверяем, что всё настроено
    if ff and fp:
        _logger.info("FFmpeg/FFprobe configuration complete")
    else:
        _logger.error("FFmpeg/FFprobe configuration incomplete - some features may not work")


# Выполняем настройку при импорте модуля
_set_ff_tools()
