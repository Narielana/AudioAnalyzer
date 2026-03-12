"""
Аудио-пайплайны обработки.

Назначение:
- Реализации преобразований: FWHT, FFT/rFFT, DCT-II/IDCT-III, DWT (Хаар).
- Вспомогательные варианты: Хаффман-подобный (μ-law), Розенброк-подобный.
- Кодирование в MP3 через ffmpeg (см. processing.codecs).
- Метрики качества вынесены в processing.metrics.

Обзор методов обработки:
========================

1. Стандартный MP3 - прямое кодирование WAV в MP3 через ffmpeg.
   Это базовый метод для сравнения.

2. FFT (Быстрое преобразование Фурье) - классический метод частотного анализа.
   Разлагает сигнал на синусоидальные компоненты.
   - Преимущества: хорошая частотная локализация, широкий спектр применения
   - Недостатки: требует комплексных вычислений, не оптимален для некоторых типов сигналов

3. FWHT (Быстрое преобразование Уолша-Адамара) - использует только сложения/вычитания.
   Разлагает сигнал на функции Уолша (прямоугольные волны).
   - Преимущества: быстрые вычисления, хорош для бинарных/ступенчатых сигналов
   - Недостатки: частотная интерпретация менее интуитивна чем FFT

4. DCT (Дискретное косинусное преобразование) - используется в JPEG, MP3.
   Разлагает сигнал на косинусоиды.
   - Преимущества: хорошее энергетическое сжатие, используется в MP3

5. DWT (Дискретное вейвлет-преобразование, Хаар) - многоуровневое разложение.
   Разделяет сигнал на аппроксимацию (низкие частоты) и детали (высокие частоты).
   - Преимущества: хорошее временное и частотное разрешение

6. Huffman-like (μ-law компандирование) - нелинейное сжатие динамического диапазона.
   Сжимает громкие звуки меньше чем тихие (логарифмическая шкала).

7. Rosenbrock-like - эвристическое нелинейное преобразование для сглаживания.

Внешние библиотеки:
- numpy: численные операции, FFT.
- logging: диагностические сообщения пайплайнов.
"""
from __future__ import annotations

import os
import time
import math
import logging
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np

# Используем специализированные модули
from .codecs import (
    ensure_ffmpeg_available,
    load_wav_mono,
    decode_audio_to_mono,
    standard_convert_to_mp3 as _standard_convert_to_mp3,
    encode_pcm_to_mp3,
    get_audio_meta,
)
from .fwht import fwht_ortho, ifwht_ortho
from .metrics import compute_metrics_batch as _compute_metrics_batch_internal
from .utils import is_power_of_two, normalize_ratio

logger = logging.getLogger("audio.processing")


# =============================================================================
# УНИВЕРСАЛЬНЫЕ ФУНКЦИИ-УТИЛИТЫ
# =============================================================================

def _load_audio_safe(wav_path: str) -> Tuple[np.ndarray, int]:
    """Безопасная загрузка аудио с fallback на soundfile.

    Пытается декодировать через ffmpeg (предпочтительно для exe),
    при ошибке падает на soundfile для локального окружения.

    Параметры:
    - wav_path: путь к аудиофайлу

    Возвращает: (pcm_data, sample_rate)
    """
    try:
        return decode_audio_to_mono(wav_path)
    except Exception as e:
        logger.debug("decode_audio_to_mono failed (%s), falling back to load_wav_mono", e)
        return load_wav_mono(wav_path)


def _create_ola_window(block_size: int) -> np.ndarray:
    """Создать sqrt-Hann окно для OLA с 50% перекрытием."""
    return np.sqrt(np.hanning(block_size) + 1e-12).astype(np.float32)


def _finalize_ola(y_accum: np.ndarray, w_accum: np.ndarray, original_len: int) -> np.ndarray:
    """Завершить OLA: нормировка окна и обрезка до исходной длины.

    Параметры:
    - y_accum: накопленный сигнал
    - w_accum: накопленные веса окна
    - original_len: исходная длина сигнала

    Возвращает: восстановленный сигнал
    """
    y = np.divide(y_accum, np.maximum(w_accum, 1e-8))[:original_len]
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / peak
    return y


def _get_output_path(wav_path: str, out_dir: str, suffix: str) -> str:
    """Сформировать путь к выходному MP3 файлу."""
    base = os.path.splitext(os.path.basename(wav_path))[0]
    return os.path.join(out_dir, f"{base}_{suffix}.mp3")


# =============================================================================
# ЭКСПОРТ МЕТРИК (обратная совместимость)
# =============================================================================

# Переэкспортируем метрики для обратной совместимости
from .metrics import (
    compute_snr_db,
    compute_rmse,
    compute_si_sdr_db,
    compute_lsd_db,
    compute_spectral_convergence,
    compute_spectral_centroid_diff_hz,
    compute_spectral_cosine_similarity,
)


# =============================================================================
# СТАНДАРТНОЕ ПРЕОБРАЗОВАНИЕ
# =============================================================================

def standard_convert_to_mp3(wav_path: str, out_dir: str, bitrate: str = "192k") -> Tuple[str, float]:
    """Обёртка над codecs.standard_convert_to_mp3 с логированием.

    Параметры:
    - wav_path: путь к исходному WAV файлу
    - out_dir: директория для сохранения MP3
    - bitrate: битрейт MP3 (например, '192k')

    Возвращает: (путь к MP3, время обработки в секундах)
    """
    logger.info("standard_convert_to_mp3 start path=%s bitrate=%s", wav_path, bitrate)
    out, dt = _standard_convert_to_mp3(wav_path, out_dir, bitrate)
    logger.info("standard_convert_to_mp3 done path=%s out=%s dt=%.3f", wav_path, out, dt)
    return out, dt


# =============================================================================
# FFT ПРЕОБРАЗОВАНИЕ
# =============================================================================

def fft_transform_and_mp3(
    wav_path: str,
    out_dir: str,
    *,
    block_size: int = 2048,
    bitrate: str = "192k",
    select_mode: str = "none",
    keep_energy_ratio: float = 1.0,
    sequency_keep_ratio: float = 1.0,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[str, float]:
    """Блочная rFFT/iFFT с OLA (Overlap-Add), опциональным отбором коэффициентов → MP3.
    
    Метод выполняет частотное преобразование Фурье с последующим кодированием в MP3.
    
    Алгоритм обработки:
    1. Загрузка и декодирование аудио в моно PCM
    2. Разбиение на перекрывающиеся блоки (50% перекрытие)
    3. Применение окна sqrt-Hann к каждому блоку
    4. Прямое rFFT (real FFT) - только для действительных сигналов
    5. Опциональный отбор коэффициентов (по энергии или lowpass)
    6. Обратное irFFT для восстановления сигнала
    7. OLA сборка и кодирование в MP3
    
    Отбор коэффициентов:
    - 'none': без отбора, полная реконструкция
    - 'energy': сохранение k% энергии (сжатие)
    - 'lowpass': сохранение низких частот (фильтрация)

    Параметры:
    - wav_path: входной WAV/аудиофайл
    - out_dir: каталог для сохранения MP3
    - block_size: длина блока (2^n), типичные значения 1024-4096
    - bitrate: битрейт MP3 (например, '192k')
    - select_mode: 'none' | 'energy' | 'lowpass'
    - keep_energy_ratio: доля энергии для сохранения (mode='energy'), 0..1
    - sequency_keep_ratio: доля низких частот (mode='lowpass'), 0..1
    - progress_cb: колбэк прогресса progress_cb(frac, msg)

    Возвращает: (путь к MP3, общее время сек).
    """
    # Нормализация параметров
    keep_energy_ratio = normalize_ratio(keep_energy_ratio)
    sequency_keep_ratio = normalize_ratio(sequency_keep_ratio)
    select_mode = (select_mode or "none").lower()

    t0_total = time.perf_counter()
    logger.info(
        "fft_transform_and_mp3 start path=%s block_size=%d mode=%s keep_energy=%.3f seq_keep=%.3f",
        wav_path, int(block_size), select_mode, keep_energy_ratio, sequency_keep_ratio,
    )
    ensure_ffmpeg_available()

    if progress_cb:
        progress_cb(0.0, "FFT: декодирование входа")

    # Безопасное декодирование
    x, sr = _load_audio_safe(wav_path)
    n = len(x)
    N = int(block_size)
    if not is_power_of_two(N):
        raise ValueError("block_size должен быть степенью двойки")

    H = max(1, N // 2)
    win = _create_ola_window(N)

    # OLA подготовка
    frames = max(1, int(np.ceil(max(0, n - N) / H)) + 1)
    total_len = (frames - 1) * H + N
    pad = total_len - n
    x_padded = np.pad(x, (0, pad), mode="constant")
    y_accum = np.zeros_like(x_padded)
    w_accum = np.zeros_like(x_padded)

    for fi in range(frames):
        i0 = fi * H
        blk = x_padded[i0 : i0 + N]
        xb = blk * win
        X = np.fft.rfft(xb)

        if select_mode == "energy" and keep_energy_ratio < 1.0:
            magsq = X.real * X.real + X.imag * X.imag
            order = np.argsort(magsq)[::-1]
            cumsum = np.cumsum(magsq[order])
            total_e = cumsum[-1] + 1e-12
            need = keep_energy_ratio * total_e
            keep_n = int(np.searchsorted(cumsum, need, side="left")) + 1
            keep_idx = order[:keep_n]
            mask = np.zeros_like(X, dtype=bool)
            mask[keep_idx] = True
            mask[0] = True
            X = np.where(mask, X, 0.0)
        elif select_mode == "lowpass" and sequency_keep_ratio < 1.0:
            k_lp = max(1, int(sequency_keep_ratio * X.shape[0]))
            mask = np.zeros_like(X, dtype=bool)
            mask[:k_lp] = True
            X = np.where(mask, X, 0.0)

        # iFFT + OLA
        rec = np.fft.irfft(X, n=N).astype(np.float32) * win
        y_accum[i0 : i0 + N] += rec
        w_accum[i0 : i0 + N] += win * win

        if progress_cb:
            progress_cb(min(0.95, 0.1 + 0.8 * (fi + 1) / frames), f"FFT: блок {fi+1}/{frames}")

    y = _finalize_ola(y_accum, w_accum, n)
    out_mp3 = _get_output_path(wav_path, out_dir, "fft")

    if progress_cb:
        progress_cb(0.97, "FFT: кодирование MP3")
    encode_pcm_to_mp3(y, sr, out_mp3, bitrate, profile="vbr")

    total_dt = time.perf_counter() - t0_total
    if progress_cb:
        progress_cb(1.0, "FFT: готово")
    logger.info("fft_transform_and_mp3 done out=%s dt=%.3f", out_mp3, total_dt)
    return out_mp3, total_dt


# =============================================================================
# DCT ПРЕОБРАЗОВАНИЕ
# =============================================================================

def _dct2(x: np.ndarray) -> np.ndarray:
    """DCT-II (ортонормированная) без SciPy через rFFT чётного отражения.

    Совместима по масштабу с _idct3.
    """
    N = int(x.shape[0])
    xr = x.astype(np.float64, copy=False)
    y = np.empty(2 * N, dtype=np.float64)
    y[:N] = xr
    y[N:] = xr[::-1]
    Y = np.fft.rfft(y)
    k = np.arange(N, dtype=np.float64)
    W = np.exp(-1j * np.pi * k / (2.0 * N))
    C = (Y[:N] * W).real
    C *= np.sqrt(2.0 / N)
    C[0] /= np.sqrt(2.0)
    return C.astype(np.float32)


def _idct3(X: np.ndarray) -> np.ndarray:
    """IDCT-III (ортонормированная), парная к _dct2."""
    N = int(X.shape[0])
    Xd = X.astype(np.float64, copy=False).copy()
    Xd[0] /= np.sqrt(2.0)
    k = np.arange(N, dtype=np.float64)
    W = np.exp(1j * np.pi * k / (2.0 * N))
    Z = Xd * W
    H = np.zeros(N + 1, dtype=np.complex128)
    H[:N] = Z
    H[N] = 0.0
    y = np.fft.irfft(H, n=2 * N)
    x = y[:N] * np.sqrt(2.0 / N)
    return x.astype(np.float32)


def dct_transform_and_mp3(
    wav_path: str,
    out_dir: str,
    *,
    block_size: int = 2048,
    bitrate: str = "192k",
    select_mode: str = "none",
    keep_energy_ratio: float = 1.0,
    sequency_keep_ratio: float = 1.0,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[str, float]:
    """Блочная DCT-II/IDCT-III c OLA и отбором коэффициентов → MP3.

    Параметры аналогичны FFT-пайплайну.
    Возвращает: (путь, время сек).
    """
    # Нормализация параметров
    keep_energy_ratio = normalize_ratio(keep_energy_ratio)
    sequency_keep_ratio = normalize_ratio(sequency_keep_ratio)
    select_mode = (select_mode or "none").lower()

    t0 = time.perf_counter()
    logger.info(
        "dct_transform_and_mp3 start path=%s block_size=%d mode=%s keep_energy=%.3f seq_keep=%.3f",
        wav_path, int(block_size), select_mode, keep_energy_ratio, sequency_keep_ratio,
    )
    ensure_ffmpeg_available()

    if progress_cb:
        progress_cb(0.0, "DCT: декодирование входа")

    x, sr = _load_audio_safe(wav_path)
    n = len(x)
    N = int(block_size)
    if not is_power_of_two(N):
        raise ValueError("block_size должен быть степенью двойки")

    H = max(1, N // 2)
    win = _create_ola_window(N)
    frames = max(1, int(np.ceil(max(0, n - N) / H)) + 1)
    total_len = (frames - 1) * H + N
    pad = total_len - n
    x_padded = np.pad(x, (0, pad), mode="constant")
    y_accum = np.zeros_like(x_padded)
    w_accum = np.zeros_like(x_padded)

    for fi in range(frames):
        i0 = fi * H
        blk = x_padded[i0 : i0 + N]
        xb = blk * win

        identity = (select_mode == "none") and (keep_energy_ratio >= 1.0) and (sequency_keep_ratio >= 1.0)
        if identity:
            rec = (xb * win).astype(np.float32)
        else:
            C = _dct2(xb)
            if select_mode == "energy" and keep_energy_ratio < 1.0:
                magsq = C * C
                order = np.argsort(magsq)[::-1]
                cumsum = np.cumsum(magsq[order])
                total_e = cumsum[-1] + 1e-12
                need = keep_energy_ratio * total_e
                keep_n = int(np.searchsorted(cumsum, need, side="left")) + 1
                keep_idx = order[:keep_n]
                mask = np.zeros_like(C, dtype=bool)
                mask[keep_idx] = True
                mask[0] = True
                C = np.where(mask, C, 0.0)
            elif select_mode == "lowpass" and sequency_keep_ratio < 1.0:
                k_lp = max(1, int(sequency_keep_ratio * C.shape[0]))
                mask = np.zeros_like(C, dtype=bool)
                mask[:k_lp] = True
                C = np.where(mask, C, 0.0)
            rec = _idct3(C) * win

        y_accum[i0 : i0 + N] += rec
        w_accum[i0 : i0 + N] += win * win

        if progress_cb:
            progress_cb(min(0.95, 0.1 + 0.8 * (fi + 1) / frames), f"DCT: блок {fi+1}/{frames}")

    y = _finalize_ola(y_accum, w_accum, n)
    out_mp3 = _get_output_path(wav_path, out_dir, "dct")

    if progress_cb:
        progress_cb(0.97, "DCT: кодирование MP3")
    encode_pcm_to_mp3(y, sr, out_mp3, bitrate, profile="vbr")

    dt = time.perf_counter() - t0
    if progress_cb:
        progress_cb(1.0, "DCT: готово")
    logger.info("dct_transform_and_mp3 done out=%s dt=%.3f", out_mp3, dt)
    return out_mp3, dt


# =============================================================================
# DWT (HAAR) ПРЕОБРАЗОВАНИЕ
# =============================================================================

def _haar_dwt_1level(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Одноуровневое вейвлет‑преобразование Хаара.

    Возвращает пару (A, D): аппроксимации и детали.
    """
    n = len(x)
    if n % 2 == 1:
        x = np.pad(x, (0, 1))
        n += 1
    a = (x[0:n:2] + x[1:n:2]) / np.sqrt(2.0)
    d = (x[0:n:2] - x[1:n:2]) / np.sqrt(2.0)
    return a.astype(np.float32), d.astype(np.float32)


def _haar_idwt_1level(a: np.ndarray, d: np.ndarray, orig_len: int) -> np.ndarray:
    """Обратное одноуровневое преобразование Хаара.

    Собирает сигнал из (A,D) и обрезает до orig_len.
    """
    n = a.shape[0] + d.shape[0]
    x = np.empty(n, dtype=np.float32)
    x[0:n:2] = (a + d) / np.sqrt(2.0)
    x[1:n:2] = (a - d) / np.sqrt(2.0)
    return x[:orig_len]


def wavelet_transform_and_mp3(
    wav_path: str,
    out_dir: str,
    *,
    block_size: int = 2048,
    bitrate: str = "192k",
    select_mode: str = "none",
    keep_energy_ratio: float = 1.0,
    sequency_keep_ratio: float = 1.0,
    levels: int = 4,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[str, float]:
    """Многоуровневое DWT (Хаар) с OLA → MP3.

    Параметры:
    - select_mode: 'none' | 'energy' | 'lowpass' применяется к вейвлет‑коэфф.
    - levels: число уровней декомпозиции.

    Возвращает: (путь к MP3, время сек).
    """
    # Нормализация параметров
    keep_energy_ratio = normalize_ratio(keep_energy_ratio)
    sequency_keep_ratio = normalize_ratio(sequency_keep_ratio)
    select_mode = (select_mode or "none").lower()
    try:
        levels = int(max(1, int(levels)))
    except (TypeError, ValueError) as e:
        logger.debug("levels_parse_error: %s, using default 4", e)
        levels = 4

    t0 = time.perf_counter()
    logger.info(
        "wavelet_transform_and_mp3 start path=%s block_size=%d mode=%s keep_energy=%.3f seq_keep=%.3f levels=%d",
        wav_path, int(block_size), select_mode, keep_energy_ratio, sequency_keep_ratio, levels,
    )
    ensure_ffmpeg_available()

    if progress_cb:
        progress_cb(0.0, "DWT: декодирование входа")

    x, sr = _load_audio_safe(wav_path)
    n = len(x)
    N = int(block_size)
    if not is_power_of_two(N):
        raise ValueError("block_size должен быть степенью двойки")

    H = max(1, N // 2)
    win = _create_ola_window(N)
    frames = max(1, int(np.ceil(max(0, n - N) / H)) + 1)
    total_len = (frames - 1) * H + N
    pad = total_len - n
    x_padded = np.pad(x, (0, pad), mode="constant")
    y_accum = np.zeros_like(x_padded)
    w_accum = np.zeros_like(x_padded)

    for fi in range(frames):
        i0 = fi * H
        blk = (x_padded[i0 : i0 + N] * win).astype(np.float32)

        # Разложение
        coeffs = []
        a = blk
        for _ in range(levels):
            a, d = _haar_dwt_1level(a)
            coeffs.append(d)
        coeffs.append(a)

        # Собираем единый вектор [A | D_L-1 | ... | D_0]
        flat = np.concatenate(coeffs[::-1])

        if select_mode == "energy" and keep_energy_ratio < 1.0:
            magsq = flat * flat
            order = np.argsort(magsq)[::-1]
            cumsum = np.cumsum(magsq[order])
            total_e = cumsum[-1] + 1e-12
            need = keep_energy_ratio * total_e
            keep_n = int(np.searchsorted(cumsum, need, side="left")) + 1
            mask = np.zeros_like(flat, dtype=bool)
            mask[order[:keep_n]] = True
            mask[0] = True
            flat = np.where(mask, flat, 0.0)
        elif select_mode == "lowpass" and sequency_keep_ratio < 1.0:
            k_lp = max(1, int(sequency_keep_ratio * flat.shape[0]))
            mask = np.zeros_like(flat, dtype=bool)
            mask[:k_lp] = True
            flat = np.where(mask, flat, 0.0)

        # Восстановление
        a_len = int(np.ceil(N / (2 ** levels)))
        a = flat[:a_len]
        ptr = a_len
        for _ in range(levels):
            d = flat[ptr : ptr + a.shape[0]] if ptr + a.shape[0] <= flat.shape[0] else np.zeros_like(a)
            ptr += a.shape[0]
            a = _haar_idwt_1level(a, d, a.shape[0] * 2)

        rec = (a[:N] * win).astype(np.float32)
        y_accum[i0 : i0 + N] += rec
        w_accum[i0 : i0 + N] += win * win

        if progress_cb:
            progress_cb(min(0.95, 0.1 + 0.8 * (fi + 1) / frames), f"DWT: блок {fi+1}/{frames}")

    y = _finalize_ola(y_accum, w_accum, n)
    out_mp3 = _get_output_path(wav_path, out_dir, "dwt")

    if progress_cb:
        progress_cb(0.97, "DWT: кодирование MP3")
    encode_pcm_to_mp3(y, sr, out_mp3, bitrate, profile="vbr")

    dt = time.perf_counter() - t0
    if progress_cb:
        progress_cb(1.0, "DWT: готово")
    logger.info("wavelet_transform_and_mp3 done out=%s dt=%.3f", out_mp3, dt)
    return out_mp3, dt


# =============================================================================
# HUFFMAN-LIKE ПРЕОБРАЗОВАНИЕ
# =============================================================================

def huffman_like_transform_and_mp3(
    wav_path: str,
    out_dir: str,
    *,
    block_size: int = 2048,
    bitrate: str = "192k",
    mu: float = 255.0,
    bits: int = 8,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[str, float]:
    """Условное "преобразование Хаффмана": μ-law компандирование + квантизация → обратное.

    Служит как имитация предобработки под энтропийное кодирование, затем MP3.

    Параметры:
    - mu: параметр μ-law компандирования
    - bits: число бит квантования

    Возвращает: (путь к MP3, время сек).
    """
    # Нормализация параметров
    try:
        mu = float(max(1.0, mu))
    except (TypeError, ValueError) as e:
        logger.debug("mu_parse_error: %s, using default 255.0", e)
        mu = 255.0
    try:
        bits = int(max(1, min(16, int(bits))))
    except (TypeError, ValueError) as e:
        logger.debug("bits_parse_error: %s, using default 8", e)
        bits = 8

    t0 = time.perf_counter()
    logger.info("huffman_like_transform_and_mp3 start path=%s mu=%.1f bits=%d", wav_path, mu, bits)
    ensure_ffmpeg_available()

    if progress_cb:
        progress_cb(0.0, "Huffman: декодирование входа")

    x, sr = _load_audio_safe(wav_path)

    # μ-law компандирование
    x_mu = np.sign(x) * (np.log1p(mu * np.abs(x)) / np.log1p(mu))

    # Равномерное квантование
    Q = max(2, int(2 ** bits))
    xi = np.clip((x_mu + 1.0) * 0.5 * (Q - 1), 0, Q - 1).astype(np.int32)
    x_rec = (xi.astype(np.float32) / (Q - 1)) * 2.0 - 1.0

    # Обратное μ-law
    y = np.sign(x_rec) * ((1.0 + mu) ** np.abs(x_rec) - 1.0) / mu

    out_mp3 = _get_output_path(wav_path, out_dir, "huffman")

    if progress_cb:
        progress_cb(0.97, "Huffman: кодирование MP3")
    encode_pcm_to_mp3(y.astype(np.float32), sr, out_mp3, bitrate, profile="vbr")

    dt = time.perf_counter() - t0
    if progress_cb:
        progress_cb(1.0, "Huffman: готово")
    logger.info("huffman_like_transform_and_mp3 done out=%s dt=%.3f", out_mp3, dt)
    return out_mp3, dt


# =============================================================================
# ROSENBROCK-LIKE ПРЕОБРАЗОВАНИЕ
# =============================================================================

def rosenbrock_like_transform_and_mp3(
    wav_path: str,
    out_dir: str,
    *,
    alpha: float = 0.2,
    beta: float = 1.0,
    bitrate: str = "192k",
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[str, float]:
    """Эвристическое "преобразование Розенброка": сглаживающая нелинейность.

    По сути — мягкое сжатие динамики: y = x / (1 + α*(x-β)^2).

    Параметры:
    - alpha: параметр сглаживания (0..10)
    - beta: параметр сдвига (-5..5)

    Возвращает: (путь к MP3, время сек).
    """
    # Нормализация параметров
    try:
        alpha = float(max(0.0, min(10.0, alpha)))
    except (TypeError, ValueError) as e:
        logger.debug("alpha_parse_error: %s, using default 0.2", e)
        alpha = 0.2
    try:
        beta = float(max(-5.0, min(5.0, beta)))
    except (TypeError, ValueError) as e:
        logger.debug("beta_parse_error: %s, using default 1.0", e)
        beta = 1.0

    t0 = time.perf_counter()
    logger.info("rosenbrock_like_transform_and_mp3 start path=%s alpha=%.3f beta=%.3f", wav_path, alpha, beta)
    ensure_ffmpeg_available()

    if progress_cb:
        progress_cb(0.0, "Rosenbrock: декодирование входа")

    x, sr = _load_audio_safe(wav_path)

    # Нелинейное преобразование
    y = x.astype(np.float32) / (1.0 + alpha * (x.astype(np.float32) - beta) ** 2)

    # Нормировка
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / peak

    out_mp3 = _get_output_path(wav_path, out_dir, "rosenbrock")

    if progress_cb:
        progress_cb(0.97, "Rosenbrock: кодирование MP3")
    encode_pcm_to_mp3(y.astype(np.float32), sr, out_mp3, bitrate, profile="vbr")

    dt = time.perf_counter() - t0
    if progress_cb:
        progress_cb(1.0, "Rosenbrock: готово")
    logger.info("rosenbrock_like_transform_and_mp3 done out=%s dt=%.3f", out_mp3, dt)
    return out_mp3, dt


# =============================================================================
# FWHT ПРЕОБРАЗОВАНИЕ
# =============================================================================

def fwht_transform_and_mp3(
    wav_path: str,
    out_dir: str,
    block_size: int = 2048,
    keep_ratio: float = 0.0,
    bitrate: str = "192k",
    progress_cb: Optional[Callable[[float, str], None]] = None,
    keep_energy_ratio: float = 1.0,
    select_mode: str = "none",
    sequency_keep_ratio: float = 1.0,
) -> Tuple[str, float]:
    """FWHT → MP3 с блочной OLA.

    Параметры отбора: top-k по коэффициентам, по энергии, либо lowpass по последовательности.

    Параметры:
    - wav_path: путь к исходному файлу
    - out_dir: директория для вывода
    - block_size: размер блока (степень двойки)
    - keep_ratio: доля коэффициентов top-k (устаревший параметр)
    - bitrate: битрейт MP3
    - progress_cb: колбэк прогресса
    - keep_energy_ratio: доля энергии для сохранения (mode='energy')
    - select_mode: 'none' | 'energy' | 'lowpass'
    - sequency_keep_ratio: доля низких частот (mode='lowpass')

    Возвращает: (путь к MP3, время сек).
    """
    # Нормализация параметров
    keep_energy_ratio = normalize_ratio(keep_energy_ratio)
    sequency_keep_ratio = normalize_ratio(sequency_keep_ratio)
    try:
        keep_ratio = float(max(0.0, min(1.0, keep_ratio or 0.0)))
    except (TypeError, ValueError) as e:
        logger.debug("keep_ratio_parse_error: %s, using default 0.0", e)
        keep_ratio = 0.0
    select_mode = (select_mode or "none").lower()

    logger.info(
        "fwht_transform_and_mp3 start path=%s block_size=%d mode=%s keep_ratio=%.3f keep_energy=%.3f seq_keep=%.3f",
        wav_path, block_size, select_mode, keep_ratio, keep_energy_ratio, sequency_keep_ratio,
    )

    t_total0 = time.perf_counter()
    ensure_ffmpeg_available()

    if progress_cb:
        progress_cb(0.0, "FWHT: загрузка WAV")
        progress_cb(0.02, "FWHT: декодирование входа")

    # Декодирование
    t_dec0 = time.perf_counter()
    x, sr = _load_audio_safe(wav_path)
    t_dec = time.perf_counter() - t_dec0

    n = len(x)
    N = int(block_size)
    if not is_power_of_two(N):
        raise ValueError("block_size должен быть степенью двойки")

    hop = N // 2
    if hop <= 0:
        hop = N

    win = _create_ola_window(N)

    frames = max(1, int(np.ceil(max(0, n - N) / hop)) + 1)
    total_len = (frames - 1) * hop + N
    pad = total_len - n
    x_padded = np.pad(x, (0, pad), mode="constant")

    y_accum = np.zeros_like(x_padded)
    w_accum = np.zeros_like(x_padded)

    use_topk = keep_ratio is not None and 0.0 < keep_ratio < 1.0
    k = max(1, int(keep_ratio * N)) if use_topk else None
    use_energy = (select_mode == "energy") and (keep_energy_ratio is not None and keep_energy_ratio < 1.0)
    use_lowpass = (select_mode == "lowpass") and (sequency_keep_ratio is not None and sequency_keep_ratio < 1.0)

    t_proc0 = time.perf_counter()
    for fi in range(frames):
        i = fi * hop
        blk = x_padded[i : i + N]
        blk_w = blk * win
        coeffs = fwht_ortho(blk_w)

        if select_mode == "none" and not use_topk:
            pass
        elif use_topk and k < N:
            thresh = np.partition(np.abs(coeffs), -k)[-k]
            coeffs = coeffs * (np.abs(coeffs) >= thresh)
        elif use_energy:
            magsq = coeffs * coeffs
            order = np.argsort(magsq)[::-1]
            cumsum = np.cumsum(magsq[order])
            total_e = cumsum[-1] + 1e-12
            need = keep_energy_ratio * total_e
            keep_n = int(np.searchsorted(cumsum, need, side="left")) + 1
            keep_idx = order[:keep_n]
            mask = np.zeros_like(coeffs, dtype=bool)
            mask[keep_idx] = True
            mask[0] = True
            coeffs = np.where(mask, coeffs, 0.0)
        elif use_lowpass:
            k_lp = max(1, int(sequency_keep_ratio * N))
            mask = np.zeros_like(coeffs, dtype=bool)
            mask[:k_lp] = True
            coeffs = np.where(mask, coeffs, 0.0)

        rec = ifwht_ortho(coeffs) * win
        y_accum[i : i + N] += rec
        w_accum[i : i + N] += win * win

        if progress_cb:
            progress_cb(min(0.95, 0.1 + 0.8 * (fi + 1) / frames), f"FWHT: блок {fi+1}/{frames}")

    y = np.divide(y_accum, np.maximum(w_accum, 1e-8))
    y = y[:n]
    t_proc = time.perf_counter() - t_proc0

    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / peak

    out_mp3 = _get_output_path(wav_path, out_dir, "fwht")

    if progress_cb:
        progress_cb(0.97, "FWHT: кодирование MP3")

    t_enc0 = time.perf_counter()
    encode_pcm_to_mp3(y, sr, out_mp3, bitrate, profile="vbr")
    t_enc = time.perf_counter() - t_enc0

    if progress_cb:
        progress_cb(1.0, "FWHT: готово")

    total_dt = time.perf_counter() - t_total0
    logger.info(
        "fwht_transform_and_mp3 done out=%s dt=%.3f t_decode=%.3f t_process=%.3f t_encode=%.3f",
        out_mp3, total_dt, t_dec, t_proc, t_enc
    )
    return out_mp3, total_dt


# =============================================================================
# ПАКЕТНЫЙ РАСЧЁТ МЕТРИК (обратная совместимость)
# =============================================================================

def _compute_metrics_batch(
    original_wav: str,
    items: List[Tuple[str, str, float]],
) -> List[Dict]:
    """Посчитать метрики качества для набора результатов.

    Параметры:
    - original_wav: путь к исходному WAV (референс)
    - items: список кортежей (variant, path_to_mp3, time_sec)

    Возвращает: список словарей с полями размера, метрик, времени и score.
    """
    return _compute_metrics_batch_internal(
        original_wav,
        items,
        load_wav_func=load_wav_mono,
        decode_audio_func=decode_audio_to_mono,
        get_meta_func=get_audio_meta,
    )


def compare_results(
    original_wav: str,
    std_mp3: str,
    fwht_mp3: str,
    t_std: float,
    t_fwht: float,
    fft_mp3: Optional[str] = None,
    t_fft: Optional[float] = None,
) -> List[Dict]:
    """СОВМЕСТИМОСТЬ: старая сигнатура. Теперь поддержка произвольного числа вариантов.

    Параметры:
    - original_wav: путь к исходному WAV
    - std_mp3: путь к стандартному MP3
    - fwht_mp3: путь к FWHT MP3
    - t_std: время обработки стандартного MP3
    - t_fwht: время обработки FWHT MP3
    - fft_mp3: опционально путь к FFT MP3
    - t_fft: опционально время обработки FFT MP3

    Возвращает: список результатов.
    """
    items: List[Tuple[str, str, float]] = [
        ("Стандартный MP3", std_mp3, float(t_std)),
        ("FWHT MP3", fwht_mp3, float(t_fwht)),
    ]
    if fft_mp3 is not None:
        items.append(("FFT MP3", fft_mp3, float(t_fft or float("nan"))))
    return _compute_metrics_batch(original_wav, items)


# =============================================================================
# ЭКСПОРТ ИМЁН
# =============================================================================

__all__ = [
    # Пайплайны
    "standard_convert_to_mp3",
    "fft_transform_and_mp3",
    "dct_transform_and_mp3",
    "wavelet_transform_and_mp3",
    "huffman_like_transform_and_mp3",
    "rosenbrock_like_transform_and_mp3",
    "fwht_transform_and_mp3",
    # Метрики (переэкспорт)
    "compute_snr_db",
    "compute_rmse",
    "compute_si_sdr_db",
    "compute_lsd_db",
    "compute_spectral_convergence",
    "compute_spectral_centroid_diff_hz",
    "compute_spectral_cosine_similarity",
    # Пакетный расчёт
    "_compute_metrics_batch",
    "compare_results",
]
