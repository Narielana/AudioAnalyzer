"""
Кодеки и I/O для аудио.

Назначение:
- Единая точка для всех операций кодирования/декодирования и пробинга метаданных.
- Поддержка FFmpeg/ffprobe (через pydub и прямые вызовы), soundfile для WAV.

Внешние библиотеки:
- pydub/FFmpeg: универсальное декодирование/кодирование.
- soundfile: точная работа с WAV и определением битовой глубины.
- numpy: преобразование форматов PCM и буферов stdin/stdout для ffmpeg.
- subprocess/os/shutil: вызовы внешних утилит и настройка путей.
"""
from __future__ import annotations

import os
import shutil
import logging
from typing import Dict, Tuple

import numpy as np
import soundfile as sf
from pydub import AudioSegment
import subprocess
import sys

logger = logging.getLogger("audio.codecs")

_FFMPEG_CONFIGURED = False


def configure_ffmpeg_search() -> None:
    """Автонастройка путей ffmpeg/ffprobe для pydub.

    Логика:
    - Сначала берём FFMPEG_BINARY из ENV или which('ffmpeg')
    - На Windows дополнительно ищем WinGet-путь (gyan.ffmpeg)
    - Для ffprobe: рядом с ffmpeg, затем FFPROBE_BINARY, затем which('ffprobe')
    - Проставляем пути в pydub и ENV для стабильной работы внутри exe
    """
    global _FFMPEG_CONFIGURED
    if _FFMPEG_CONFIGURED:
        return
    # Ищем ffmpeg по приоритетам
    path = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not path:
        try:
            base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
            if os.path.isdir(base):
                for name in os.listdir(base):
                    if name.lower().startswith("gyan.ffmpeg"):
                        cand = os.path.join(base, name, "ffmpeg-8.0-full_build", "bin", "ffmpeg.exe")
                        if os.path.exists(cand):
                            path = cand
                            break
        except Exception as e:
            logger.debug("WinGet ffmpeg search failed: %s", e)
    if path and os.path.exists(path):
        os.environ["FFMPEG_BINARY"] = path
        AudioSegment.converter = path  # уведомляем pydub о местоположении ffmpeg
    # Ищем ffprobe по соседству с ffmpeg либо по ENV/PATH
    probe = None
    try:
        if path and path.endswith("ffmpeg.exe"):
            cand = path.replace("ffmpeg.exe", "ffprobe.exe")
            if os.path.exists(cand):
                probe = cand
    except Exception as e:
        logger.debug("ffprobe path lookup failed: %s", e)
    if not probe:
        p_env = os.environ.get("FFPROBE_BINARY")
        if p_env and os.path.exists(p_env):
            probe = p_env
    if not probe:
        w = shutil.which("ffprobe")
        if w and os.path.exists(w):
            probe = w
    if probe:
        AudioSegment.ffprobe = probe
        os.environ["FFPROBE_BINARY"] = probe
    _FFMPEG_CONFIGURED = True
    logger.debug("FFmpeg configured: %s; ffprobe: %s", os.environ.get("FFMPEG_BINARY"), os.environ.get("FFPROBE_BINARY"))


def ensure_ffmpeg_available() -> None:
    """Убедиться, что ffmpeg/ffprobe доступны (сконфигурированы в pydub/ENV).

    Бросает RuntimeError с подсказкой, если обнаружить бинарники не удалось.
    """
    configure_ffmpeg_search()
    try:
        AudioSegment.converter
    except Exception as e:
        raise RuntimeError("FFmpeg недоступен. Установите FFmpeg и добавьте его в PATH.") from e


def load_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    """Загрузить WAV и привести к моно float32.

    Возвращает: (x, sample_rate). Для стерео берётся среднее по каналам.
    """
    data, sr = sf.read(path, always_2d=False)
    if getattr(data, 'ndim', 1) == 2:
        data = data.mean(axis=1)
    x = data.astype(np.float32)
    logger.debug("Loaded WAV mono: %s sr=%d len=%d", path, sr, len(x))
    return x, sr


def _ffmpeg_creationflags() -> int:
    """Флаг для скрытого запуска ffmpeg/ffprobe в Windows (и 0 для других ОС)."""
    if sys.platform.startswith('win'):
        return 0x08000000  # CREATE_NO_WINDOW
    return 0


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    """Запустить ffmpeg/ffprobe с подавлением окна и собрать stdout/stderr."""
    configure_ffmpeg_search()
    logger.debug("Run ffmpeg: %s", " ".join(args))
    return subprocess.run(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=_ffmpeg_creationflags(),
        check=True,
    )


def _probe_sample_rate(path: str) -> int:
    """Определить частоту дискретизации файла через ffprobe; фоллбек 44100."""
    try:
        cp = _run_ffmpeg([
            AudioSegment.ffprobe if hasattr(AudioSegment, 'ffprobe') else 'ffprobe',
            '-v','error','-select_streams','a:0','-show_entries','stream=sample_rate','-of','default=nk=1:nw=1',
            path
        ])
        sr = int(cp.stdout.strip().decode('utf-8', errors='ignore') or '0')
        return sr if sr > 0 else 44100
    except Exception as e:
        logger.debug("_probe_sample_rate failed for %s: %s", path, e)
        return 44100


def decode_audio_to_mono(path: str) -> Tuple[np.ndarray, int]:
    """Декодировать аудиофайл через ffmpeg в моно PCM float32 без временных файлов.

    - Частота выбирается через ffprobe; поток читается из stdout (-f s16le).
    - Возвращает (x, sample_rate).
    """
    sr = _probe_sample_rate(path)
    # ffmpeg пишет raw PCM (s16le) в stdout; читаем из CompletedProcess.stdout
    cp = _run_ffmpeg([
        os.environ.get('FFMPEG_BINARY','ffmpeg'),
        '-v','error','-nostdin','-y','-i', path,
        '-ac','1','-ar',str(sr),
        '-f','s16le','-acodec','pcm_s16le','-'
    ])
    data = np.frombuffer(cp.stdout, dtype=np.int16)
    # Нормируем в диапазон [-1,1]
    x = (data.astype(np.float32) / 32768.0)
    logger.debug("Decoded audio: %s sr=%d len=%d", path, sr, len(x))
    return x, sr


def standard_convert_to_mp3(wav_path: str, out_dir: str, bitrate: str = "192k") -> Tuple[str, float]:
    """WAV → MP3 через ffmpeg CLI. Возвращает (путь, время сек)."""
    import time

    ensure_ffmpeg_available()
    base = os.path.splitext(os.path.basename(wav_path))[0]
    out_mp3 = os.path.join(out_dir, f"{base}_standard.mp3")
    t0 = time.perf_counter()
    _run_ffmpeg([
        os.environ.get('FFMPEG_BINARY','ffmpeg'),
        '-v','error','-nostdin','-y','-i', wav_path,
        '-vn','-b:a', bitrate,
        out_mp3,
    ])
    dt = time.perf_counter() - t0
    logger.info("standard_convert_to_mp3 done out=%s dt=%.3f", out_mp3, dt)
    return out_mp3, dt


def _bitrate_to_qscale(bitrate: str) -> int:
    """Грубое сопоставление битрейта к шкале качества LAME (0..9), ниже — лучше.

    Используется для профиля VBR (-q:a). Возвращает 0..9.
    """
    try:
        s = str(bitrate).strip().lower()
        if s.endswith('k'):
            kb = int(float(s[:-1]))
        else:
            kb = int(float(s))
    except Exception:
        kb = 192
    # Маппинг по типовым значениям
    if kb >= 320:
        return 0
    if kb >= 256:
        return 1
    if kb >= 224:
        return 2
    if kb >= 192:
        return 2
    if kb >= 160:
        return 4
    if kb >= 128:
        return 5
    if kb >= 96:
        return 6
    return 7


def encode_wav_to_mp3(in_wav: str, out_mp3: str, bitrate: str = "192k", *, profile: str = 'cbr', vbr_quality: int | None = None) -> float:
    """Кодирует WAV → MP3; возвращает время (сек).
    profile: 'cbr' (по умолчанию) или 'vbr' — для VBR используем -q:a (0..9; 0 — лучше всего).
    """
    import time

    ensure_ffmpeg_available()
    t0 = time.perf_counter()
    args = [
        os.environ.get('FFMPEG_BINARY','ffmpeg'),
        '-v','error','-nostdin','-y','-i', in_wav,
        '-vn',
    ]
    prof = (profile or 'cbr').lower()
    if prof == 'vbr':
        q = _bitrate_to_qscale(bitrate) if vbr_quality is None else int(vbr_quality)
        args += ['-q:a', str(max(0, min(9, q)))]
    else:
        args += ['-b:a', bitrate]
    args.append(out_mp3)
    _run_ffmpeg(args)
    dt = time.perf_counter() - t0
    logger.debug("encode_wav_to_mp3 done out=%s dt=%.3f profile=%s", out_mp3, dt, prof)
    return dt


def encode_pcm_to_mp3(pcm: np.ndarray, sample_rate: int, out_mp3: str, bitrate: str = "192k", *, profile: str = 'vbr', vbr_quality: int | None = None) -> float:
    """Кодирует моно PCM (float32 в диапазоне [-1,1]) напрямую в MP3 через stdin ffmpeg, без временных WAV.
    Возвращает время кодирования (сек).
    """
    import time
    import subprocess as _sp

    ensure_ffmpeg_available()
    # Подготовка s16le буфера
    x = np.asarray(pcm, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    buf = (np.round(x * 32767.0).astype(np.int16)).tobytes()

    t0 = time.perf_counter()
    args = [
        os.environ.get('FFMPEG_BINARY','ffmpeg'),
        '-v','error','-y',
        '-f','s16le','-ac','1','-ar', str(int(sample_rate)), '-i','pipe:0',
        '-vn',
    ]
    prof = (profile or 'vbr').lower()
    if prof == 'vbr':
        q = _bitrate_to_qscale(bitrate) if vbr_quality is None else int(vbr_quality)
        args += ['-q:a', str(max(0, min(9, q)))]
    else:
        args += ['-b:a', bitrate]
    args.append(out_mp3)

    # ВАЖНО: без -nostdin, т.к. читаем из pipe:0
    cp = _sp.run(
        args,
        input=buf,
        stdout=_sp.PIPE,
        stderr=_sp.PIPE,
        creationflags=_ffmpeg_creationflags(),
        check=True,
    )
    dt = time.perf_counter() - t0
    logger.debug("encode_pcm_to_mp3 done out=%s dt=%.3f profile=%s size_in=%d", out_mp3, dt, prof, len(buf))
    return dt


def _sf_bit_depth_from_subtype(subtype: str) -> int:
    """Вернуть битовую глубину для подтипа soundfile; по умолчанию 16 бит."""
    m = {"PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64}
    return m.get(subtype, 16)


def _probe_via_ffmpeg_i(path: str) -> tuple[float,int,int]:
    """Пробинг метаданных через ffprobe: (duration_sec, sample_rate_hz, channels).
    Используем ffprobe c JSON-выводом — устойчиво внутри собранного exe.
    """
    import subprocess, json
    configure_ffmpeg_search()
    # Ищем ffprobe рядом с ffmpeg или в PATH/ENV, читаем JSON и извлекаем поля
    ffprobe_exe = getattr(AudioSegment, 'ffprobe', None) or 'ffprobe'
    try:
        cp = subprocess.run(
            [ffprobe_exe, '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_ffmpeg_creationflags(),
            check=True,
        )
        info = json.loads(cp.stdout.decode('utf-8', errors='ignore') or '{}')
    except Exception as e:
        logger.debug("ffprobe failed: %s", e)
        return 0.0, 44100, 2
    # Длительность
    dur_sec = 0.0
    try:
        dur_str = (info.get('format') or {}).get('duration')
        if dur_str is not None:
            dur_sec = float(dur_str)
    except Exception:
        dur_sec = 0.0
    # Потоки аудио: берём первый аудио-стрим
    sr = 44100
    ch = 2
    try:
        for st in info.get('streams') or []:
            if (st.get('codec_type') or '').lower() == 'audio':
                try:
                    sr = int(st.get('sample_rate') or sr)
                except Exception as e:
                    logger.debug("sample_rate parse error: %s", e)
                try:
                    ch = int(st.get('channels') or ch)
                except Exception as e:
                    logger.debug("channels parse error: %s", e)
                break
    except Exception as e:
        logger.debug("audio stream parsing failed: %s", e)
    return float(dur_sec or 0.0), int(sr or 44100), int(ch or 2)


def get_audio_meta(path: str) -> Dict[str, int]:
    """Метаданные: sample_rate_hz, bit_depth_bits, channels, bitrate_bps.
    Для несжатых WAV используем soundfile; для прочих — ffprobe JSON.
    """
    # WAV читаем напрямую, остальное — оцениваем через длительность и размер
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".wav":
            with sf.SoundFile(path) as f:
                sr = int(f.samplerate)
                ch = int(f.channels)
                try:
                    bd = _sf_bit_depth_from_subtype(f.subtype)
                except Exception as e:
                    logger.debug("bit_depth subtype parse error: %s", e)
                    bd = 16
                meta = {"sample_rate_hz": sr, "bit_depth_bits": bd, "channels": ch, "bitrate_bps": sr * bd * ch}
                logger.debug("WAV meta: %s %s", path, meta)
                return meta
    except Exception as e:
        logger.debug("WAV meta read failed for %s: %s", path, e)
    # Use ffmpeg -i probing for non-wav (works in frozen exe if ffmpeg is configured)
    dur, sr, ch = _probe_via_ffmpeg_i(path)
    size_b = os.path.getsize(path)
    duration_sec = max(1e-6, dur)
    # битовая глубина неизвестна для сжатых форматов — репортим 16 по умолчанию
    bd = 16
    br_bps = int(8.0 * size_b / duration_sec)
    meta = {"sample_rate_hz": int(sr), "bit_depth_bits": int(bd), "channels": int(ch), "bitrate_bps": int(br_bps)}
    logger.debug("File meta (ffmpeg -i): %s %s", path, meta)
    return meta
