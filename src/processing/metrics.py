"""
Метрики качества аудио.

Назначение:
- Все функции вычисления метрик качества сгруппированы здесь.
- Сравнение оригинального и обработанного сигналов.
- Расчёт агрегированного балла.

Метрики:
- SNR (дБ): Signal-to-Noise Ratio — выше лучше
- RMSE: Root Mean Square Error — ниже лучше
- SI-SDR (дБ): Scale-Invariant Signal-to-Distortion Ratio — выше лучше
- LSD (дБ): Log-Spectral Distance — ниже лучше
- Spectral Convergence: ошибка амплитуд спектра — ниже лучше
- Spectral Centroid Δ (Гц): разница центров спектра — ниже лучше
- Cosine Similarity: схожесть спектров (0..1) — выше лучше

Внешние библиотеки: numpy, math, logging, os.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("audio.metrics")

# Type alias для PCM сигналов
PCMSignal = NDArray[np.float32]


# =============================================================================
# УТИЛИТЫ
# =============================================================================

def _resample_linear(x: PCMSignal, sr_from: int, sr_to: int) -> PCMSignal:
    """Линейная интерполяция для приведения дискретизации test→reference.

    Алгоритм:
    1) Если частоты совпадают или сигнал пуст — вернуть x в float32 без копии.
    2) Вычислить новую длину new_len = round(len(x)*sr_to/sr_from).
    3) Построить нормализованные оси xp∈[0,1] и xnew∈[0,1] для старых/новых отсчётов.
    4) Вызвать np.interp(xnew, xp, x) и привести к float32.

    Параметры:
    - x: входной сигнал PCM float32
    - sr_from: исходная частота дискретизации
    - sr_to: целевая частота дискретизации

    Возвращает: ресемплированный сигнал в float32.
    """
    if sr_from == sr_to or len(x) == 0:
        return x.astype(np.float32, copy=False)
    new_len = max(1, int(round(len(x) * float(sr_to) / float(sr_from))))
    xp = np.linspace(0, 1, len(x), endpoint=True)
    xnew = np.linspace(0, 1, new_len, endpoint=True)
    y = np.interp(xnew, xp, x).astype(np.float32)
    return y


# =============================================================================
# МЕТРИКИ ВРЕМЕННОЙ ОБЛАСТИ
# =============================================================================

def compute_snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    """Вычислить SNR в дБ между опорным и тестовым сигналами.

    Алгоритм:
    1) Усечь оба сигнала до общей длины N = min(len(ref), len(test)).
    2) Вычислить вектор ошибки noise = ref - test.
    3) Оценить мощности Psig = mean(ref^2) и Pnoise = mean(noise^2) с защитой eps.
    4) Вернуть 10*log10(Psig/Pnoise) в дБ.

    Параметры:
    - reference: опорный PCM float32 [-1,1]
    - test: тестовый PCM float32 [-1,1]

    Возвращает: SNR в дБ (float); выше — лучше.
    """
    n = min(len(reference), len(test))
    if n == 0:
        return float("nan")
    ref = reference[:n]
    tst = test[:n]
    noise = ref - tst
    ref_power = float(np.mean(ref * ref) + 1e-12)
    noise_power = float(np.mean(noise * noise) + 1e-12)
    snr = 10.0 * math.log10(ref_power / noise_power)
    return snr


def compute_rmse(reference: np.ndarray, test: np.ndarray) -> float:
    """RMSE во временной области на общей части сигналов.

    Алгоритм:
    1) Усечь сигналы до общей длины N.
    2) Посчитать вектор ошибки e = ref - test (в float32).
    3) Вернуть sqrt(mean(e^2)).

    Параметры:
    - reference, test: PCM float32 сигналы

    Возвращает: корень из среднего квадрата ошибки; ниже — лучше.
    """
    n = min(len(reference), len(test))
    if n == 0:
        return float("nan")
    err = reference[:n].astype(np.float32) - test[:n].astype(np.float32)
    return float(np.sqrt(np.mean(err * err)))


def compute_si_sdr_db(reference: np.ndarray, test: np.ndarray) -> float:
    """Вычислить SI-SDR (scale‑invariant) в дБ.

    Алгоритм:
    1) Усечь до N и привести к float32.
    2) Оценить масштаб alpha = <s,y>/||s||^2 (проекция y на s).
    3) Целевой компонент y_hat = alpha*s; шум e = y - y_hat.
    4) Вернуть 10*log10(||y_hat||^2 / ||e||^2).

    Параметры:
    - reference: опорный сигнал PCM float32
    - test: тестовый сигнал PCM float32

    Возвращает: SI-SDR в дБ; инвариантен к масштабу; выше — лучше.
    """
    n = min(len(reference), len(test))
    if n == 0:
        return float("nan")
    s = reference[:n].astype(np.float32)
    y = test[:n].astype(np.float32)
    s_energy = float(np.sum(s * s) + 1e-12)
    alpha = float(np.dot(s, y) / s_energy)
    e_target = alpha * s
    e_noise = y - e_target
    num = float(np.sum(e_target * e_target) + 1e-12)
    den = float(np.sum(e_noise * e_noise) + 1e-12)
    return 10.0 * math.log10(num / den)


# =============================================================================
# МЕТРИКИ СПЕКТРАЛЬНОЙ ОБЛАСТИ
# =============================================================================

def compute_lsd_db(
    reference: np.ndarray,
    test: np.ndarray,
    sr_ref: int,
    sr_test: int,
    n_fft: int = 1024,
    hop: int = 512,
) -> float:
    """Log-Spectral Distance (дБ) — средняя по окнам; ниже — лучше.

    Алгоритм:
    1) Привести test к sr_ref линейной интерполяцией.
    2) Если N>len(x): допаддить и посчитать одно окно; иначе — бежать окнами по H.
    3) Для каждой рамки: применить окно, rFFT, взять лог-амплитуды Sa,Sb.
    4) Посчитать RMSE(Sa,Sb) и усреднить по окнам (игнорируя NaN/Inf).

    Параметры:
    - reference: опорный сигнал PCM float32
    - test: тестовый сигнал PCM float32
    - sr_ref: частота дискретизации опорного сигнала
    - sr_test: частота дискретизации тестового сигнала
    - n_fft: размер окна FFT (по умолчанию 1024)
    - hop: шаг окна (по умолчанию 512)

    Возвращает: среднее LSD в дБ; ниже — лучше.
    """
    t = _resample_linear(test, sr_test, sr_ref)
    n = min(len(reference), len(t))
    if n <= 0:
        return float("nan")

    ref = reference[:n].astype(np.float32)
    tst = t[:n].astype(np.float32)
    N = int(n_fft)
    H = int(hop) if hop else N // 2
    if H <= 0:
        H = max(1, N // 2)

    if N > n:
        # Короткий сигнал: одно окно с дополнением
        win = np.hanning(N).astype(np.float32) + 1e-12
        a = np.zeros(N, dtype=np.float32)
        a[:n] = ref * win[:n]
        b = np.zeros(N, dtype=np.float32)
        b[:n] = tst * win[:n]
        A = np.fft.rfft(a)
        B = np.fft.rfft(b)
        Sa = 10.0 * np.log10(np.abs(A) ** 2 + 1e-12)
        Sb = 10.0 * np.log10(np.abs(B) ** 2 + 1e-12)
        d = float(np.sqrt(np.mean((Sa - Sb) ** 2)))
        return d if np.isfinite(d) else float("nan")

    win = np.hanning(N).astype(np.float32) + 1e-12
    frames = max(1, 1 + (n - N) // H)
    lsd_vals = []

    for i in range(frames):
        s = i * H
        e = s + N
        if e > n:
            e = n
            s = max(0, e - N)
        a = ref[s:e]
        b = tst[s:e]
        if len(a) < N:
            pad = N - len(a)
            a = np.pad(a, (0, pad))
            b = np.pad(b, (0, pad))
        a = a * win
        b = b * win
        A = np.fft.rfft(a)
        B = np.fft.rfft(b)
        Sa = 10.0 * np.log10(np.abs(A) ** 2 + 1e-12)
        Sb = 10.0 * np.log10(np.abs(B) ** 2 + 1e-12)
        d = np.sqrt(np.mean((Sa - Sb) ** 2))
        if np.isfinite(d):
            lsd_vals.append(float(d))

    if not lsd_vals:
        return float("nan")
    return float(np.mean(lsd_vals))


def compute_spectral_convergence(
    reference: np.ndarray,
    test: np.ndarray,
    sr_ref: int,
    sr_test: int,
    n_fft: int = 1024,
    hop: int = 512,
) -> float:
    """Spectral Convergence: среднее по окнам |||X|-|Y|||_2 / (||X||_2 + eps).

    Параметры:
    - reference, test: PCM float32 сигналы ([-1,1])
    - sr_ref, sr_test: частоты дискретизации; test приводится к sr_ref
    - n_fft: размер окна rFFT
    - hop: шаг между окнами

    Возвращает: среднюю спектральную сходимость; ниже — лучше.
    """
    t = _resample_linear(test, sr_test, sr_ref)
    n = min(len(reference), len(t))
    if n <= 0:
        return float("nan")

    ref = reference[:n].astype(np.float32)
    tst = t[:n].astype(np.float32)
    N = int(n_fft)
    H = int(hop) if hop else N // 2
    if H <= 0:
        H = max(1, N // 2)

    win = np.hanning(N).astype(np.float32)
    vals = []
    i = 0

    while i + N <= n or (i < n and len(vals) == 0):
        s = i
        e = min(n, i + N)
        a = ref[s:e]
        b = tst[s:e]
        if len(a) < N:
            pad = N - len(a)
            a = np.pad(a, (0, pad))
            b = np.pad(b, (0, pad))
        A = np.fft.rfft(a * win)
        B = np.fft.rfft(b * win)
        magA = np.abs(A)
        magB = np.abs(B)
        num = np.linalg.norm(magA - magB)
        den = np.linalg.norm(magA) + 1e-12
        v = float(num / den)
        if np.isfinite(v):
            vals.append(v)
        i += H

    if not vals:
        return float("nan")
    return float(np.mean(vals))


def compute_spectral_centroid_diff_hz(
    reference: np.ndarray,
    test: np.ndarray,
    sr_ref: int,
    sr_test: int,
    n_fft: int = 1024,
    hop: int = 512,
) -> float:
    """Средняя абсолютная разница спектрального центроида (в Гц).

    Спектральный центроид — это "центр тяжести" спектра:
    centroid = Σ(k * |X_k|) / Σ|X_k|
    
    Где k — индекс частотного бина. Центроид измеряется в Гц
    и отражает "яркость" звука:
    - Низкий центроид (~500-2000 Гц): басовые, глухие звуки
    - Высокий центроид (~5000-15000 Гц): яркие, шипящие звуки
    
    ВАЖНО: Для сигналов с широкополосным шумом центроид может быть
    очень высоким (до 10000+ Гц), так как шум содержит высокочастотные
    компоненты. Это НЕ ошибка, а нормальное поведение метрики.

    Для каждого окна вычисляем центроид Σ(k*|X_k|)/Σ|X_k|,
    затем усредняем |centroid_ref - centroid_test|.

    Параметры:
    - reference, test: PCM float32 сигналы
    - sr_ref, sr_test: частоты дискретизации
    - n_fft: размер окна FFT
    - hop: шаг окна

    Возвращает: среднюю разницу центроидов в Гц; ниже — лучше.
    """
    t = _resample_linear(test, sr_test, sr_ref)
    n = min(len(reference), len(t))
    if n <= 0:
        return float("nan")

    ref = reference[:n].astype(np.float32)
    tst = t[:n].astype(np.float32)
    N = int(n_fft)
    H = int(hop) if hop else N // 2
    if H <= 0:
        H = max(1, N // 2)

    win = np.hanning(N).astype(np.float32)
    df = sr_ref / float(N)
    vals = []
    i = 0

    while i + N <= n or (i < n and len(vals) == 0):
        s = i
        e = min(n, i + N)
        a = ref[s:e]
        b = tst[s:e]
        if len(a) < N:
            pad = N - len(a)
            a = np.pad(a, (0, pad))
            b = np.pad(b, (0, pad))
        A = np.abs(np.fft.rfft(a * win))
        B = np.abs(np.fft.rfft(b * win))
        k = np.arange(len(A), dtype=np.float32)
        ca = float(np.sum(k * A) / (np.sum(A) + 1e-12)) * df
        cb = float(np.sum(k * B) / (np.sum(B) + 1e-12)) * df
        d = abs(ca - cb)
        if np.isfinite(d):
            vals.append(d)
        i += H

    if not vals:
        return float("nan")
    return float(np.mean(vals))


def compute_spectral_cosine_similarity(
    reference: np.ndarray,
    test: np.ndarray,
    sr_ref: int,
    sr_test: int,
    n_fft: int = 1024,
    hop: int = 512,
) -> float:
    """Средняя косинусная близость спектров (0..1).

    На каждом окне считаем cos_sim = <|X|, |Y|> / (||X||·||Y||) и усредняем.

    Параметры:
    - reference, test: PCM float32 сигналы
    - sr_ref, sr_test: частоты дискретизации
    - n_fft: размер окна FFT
    - hop: шаг окна

    Возвращает: среднюю косинусную схожесть (0..1); выше — лучше.
    """
    t = _resample_linear(test, sr_test, sr_ref)
    n = min(len(reference), len(t))
    if n <= 0:
        return float("nan")

    ref = reference[:n].astype(np.float32)
    tst = t[:n].astype(np.float32)
    N = int(n_fft)
    H = int(hop) if hop else N // 2
    if H <= 0:
        H = max(1, N // 2)

    win = np.hanning(N).astype(np.float32)
    vals = []
    i = 0

    while i + N <= n or (i < n and len(vals) == 0):
        s = i
        e = min(n, i + N)
        a = ref[s:e]
        b = tst[s:e]
        if len(a) < N:
            pad = N - len(a)
            a = np.pad(a, (0, pad))
            b = np.pad(b, (0, pad))
        A = np.abs(np.fft.rfft(a * win)).astype(np.float32)
        B = np.abs(np.fft.rfft(b * win)).astype(np.float32)
        num = float(np.dot(A, B))
        den = float(np.linalg.norm(A) * np.linalg.norm(B) + 1e-12)
        cs = num / den
        if np.isfinite(cs):
            vals.append(cs)
        i += H

    if not vals:
        return float("nan")
    return float(np.mean(vals))


# =============================================================================
# ПАКЕТНЫЙ РАСЧЁТ МЕТРИК
# =============================================================================

def compute_metrics_batch(
    original_wav: str,
    items: List[Tuple[str, str, float]],
    load_wav_func: Callable[[str], Tuple[np.ndarray, int]],
    decode_audio_func: Callable[[str], Tuple[np.ndarray, int]],
    get_meta_func: Callable[[str], Dict[str, int]],
) -> List[Dict]:
    """Посчитать метрики качества для набора результатов.

    Алгоритм:
    1) Пробинг метаданных исходника и загрузка опорного PCM (ref, sr_ref).
    2) Для каждого (variant, path, time):
       2.1) Пробинг метаданных файла (sr, channels, bitrate, size).
       2.2) Декод PCM; расчёт всех метрик.
       2.3) Формирование словаря результата.
    3) Нормировка метрик (min‑max) и расчёт агрегированного score.
    4) Вернуть список результатов.

    Параметры:
    - original_wav: путь к исходному WAV (референс)
    - items: список кортежей (variant, path_to_mp3, time_sec)
    - load_wav_func: функция загрузки WAV (из codecs)
    - decode_audio_func: функция декодирования аудио (из codecs)
    - get_meta_func: функция получения метаданных (из codecs)

    Возвращает: список словарей с полями размера, метрик, времени и score.
    """
    # Метаданные исходника
    orig = get_meta_func(original_wav)
    ref, sr_ref = load_wav_func(original_wav)

    def metrics_for(path: str) -> Tuple[Dict, float, float, float, float, float, float, float]:
        """Вычислить все метрики для одного файла."""
        meta = get_meta_func(path)
        sig, sr = decode_audio_func(path)
        lsd = compute_lsd_db(ref, sig, sr_ref, sr)
        snr = compute_snr_db(ref, sig)
        sc = compute_spectral_convergence(ref, sig, sr_ref, sr)
        rmse = compute_rmse(ref, sig)
        si_sdr = compute_si_sdr_db(ref, sig)
        sc_diff = compute_spectral_centroid_diff_hz(ref, sig, sr_ref, sr)
        cos_sim = compute_spectral_cosine_similarity(ref, sig, sr_ref, sr)
        return meta, float(lsd), float(snr), float(sc), float(rmse), float(si_sdr), float(sc_diff), float(cos_sim)

    results: List[Dict] = []
    for variant, path, time_s in items:
        meta, lsd, snr, sc, rmse, sisdr, scdiff, cossim = metrics_for(path)

        logger.info(
            "metrics_computed",
            extra={
                "variant": variant,
                "path": path,
                "lsd_db": float(lsd),
                "snr_db": float(snr),
                "spec_conv": float(sc),
                "rmse": float(rmse),
                "si_sdr_db": float(sisdr),
                "spec_centroid_diff_hz": float(scdiff),
                "spec_cosine": float(cossim),
            }
        )

        out = {
            "variant": variant,
            "path": path,
            "size_bytes": os.path.getsize(path),
            "sample_rate_hz": meta["sample_rate_hz"],
            "bit_depth_bits": meta["bit_depth_bits"],
            "bitrate_bps": meta["bitrate_bps"],
            "time_sec": float(time_s),
            "lsd_db": float(lsd),
            "snr_db": float(snr),
            "spec_conv": float(sc),
            "rmse": float(rmse),
            "si_sdr_db": float(sisdr),
            "spec_centroid_diff_hz": float(scdiff),
            "spec_cosine": float(cossim),
            "orig_sample_rate_hz": orig["sample_rate_hz"],
            "orig_bit_depth_bits": orig["bit_depth_bits"],
            "orig_bitrate_bps": orig["bitrate_bps"],
        }
        out["delta_sr"] = out["sample_rate_hz"] - out["orig_sample_rate_hz"]
        out["delta_bd"] = out["bit_depth_bits"] - out["orig_bit_depth_bits"]
        out["delta_br_bps"] = out["bitrate_bps"] - out["orig_bitrate_bps"]
        results.append(out)

    # Агрегированный балл (min-max нормировка)
    def _minmax(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
        vals_f = [v for v in vals if v == v]  # filter NaN
        if not vals_f:
            return None, None
        return min(vals_f), max(vals_f)

    eps = 1e-12
    lsd_min, lsd_max = _minmax([r["lsd_db"] for r in results])
    sc_min, sc_max = _minmax([r["spec_conv"] for r in results])
    t_min, t_max = _minmax([r["time_sec"] for r in results])
    snr_min, snr_max = _minmax([r["snr_db"] for r in results])
    rmse_min, rmse_max = _minmax([r.get("rmse") for r in results])
    sisdr_min, sisdr_max = _minmax([r.get("si_sdr_db") for r in results])
    scdiff_min, scdiff_max = _minmax([r.get("spec_centroid_diff_hz") for r in results])
    cos_min, cos_max = _minmax([r.get("spec_cosine") for r in results])

    for r in results:
        # Нормировка: все приводим к "выше-лучше" (1.0 = лучший результат)
        # Для метрик "ниже-лучше" инвертируем: (max - value) / (max - min)
        # Для метрик "выше-лучше" оставляем: (value - min) / (max - min)

        # Метрики "ниже-лучше" (инвертируем):
        lsd_n = 0.0 if lsd_min is None else (lsd_max - r["lsd_db"]) / ((lsd_max - lsd_min) + eps)
        sc_n = 0.0 if sc_min is None else (sc_max - r["spec_conv"]) / ((sc_max - sc_min) + eps)
        rmse_n = 0.0 if rmse_min is None else (rmse_max - r["rmse"]) / ((rmse_max - rmse_min) + eps)
        scdiff_n = 0.0 if scdiff_min is None else (scdiff_max - r["spec_centroid_diff_hz"]) / ((scdiff_max - scdiff_min) + eps)
        t_n = 0.0 if t_min is None else (t_max - r["time_sec"]) / ((t_max - t_min) + eps)

        # Метрики "выше-лучше" (оставляем как есть):
        snr_n = 0.0 if snr_min is None else (r["snr_db"] - snr_min) / ((snr_max - snr_min) + eps)
        sisdr_n = 0.0 if sisdr_min is None else (r["si_sdr_db"] - sisdr_min) / ((sisdr_max - sisdr_min) + eps)
        cos_n = 0.0 if cos_min is None else (r["spec_cosine"] - cos_min) / ((cos_max - cos_min) + eps)

        # Взвешенная сумма (все компоненты приведены к "выше-лучше")
        # Чем выше score, тем лучше метод
        r["score"] = float(
            0.20 * lsd_n +       # LSD: ниже лучше → инвертировано
            0.15 * sc_n +        # Spectral Conv: ниже лучше → инвертировано
            0.20 * snr_n +       # SNR: выше лучше
            0.15 * rmse_n +      # RMSE: ниже лучше → инвертировано
            0.15 * sisdr_n +     # SI-SDR: выше лучше
            0.05 * scdiff_n +    # Centroid Δ: ниже лучше → инвертировано
            0.05 * cos_n +       # Cosine: выше лучше
            0.05 * t_n           # Time: ниже лучше → инвертировано
        )

    return results


# =============================================================================
# ЭКСПОРТ ИМЁН
# =============================================================================

__all__ = [
    # Метрики временной области
    "compute_snr_db",
    "compute_rmse",
    "compute_si_sdr_db",
    # Метрики спектральной области
    "compute_lsd_db",
    "compute_spectral_convergence",
    "compute_spectral_centroid_diff_hz",
    "compute_spectral_cosine_similarity",
    # Пакетный расчёт
    "compute_metrics_batch",
]
