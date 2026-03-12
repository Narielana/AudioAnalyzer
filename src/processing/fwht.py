"""
FWHT: математика и блочная обработка (OLA) — без кода кодеков/файлов.

Содержимое:
- Чистые реализации FWHT/IFWHT, ортонормированные варианты.
- Блочная обработка с перекрытием/суммированием (OLA) и опциональным отбором коэффициентов.

Внешние библиотеки: numpy (векторизация), math (нормировка), logging.

Примечание по производительности:
FWHT должен быть быстрее FFT теоретически (только сложения/вычитания, без умножений),
но numpy FFT оптимизирован на уровне C/Fortran. Для достижения сопоставимой скорости
используем векторизованные операции вместо Python-циклов.
"""
from __future__ import annotations

import math
import logging
from typing import Callable, Optional

import numpy as np

from .utils import is_power_of_two

logger = logging.getLogger("audio.fwht")


def fwht(x: np.ndarray) -> np.ndarray:
    """Векторизованное быстрое преобразование Уолша–Адамара (FWHT).
    
    Алгоритм использует бабочку Уолша-Адамара с O(N log N) операциями.
    В отличие от FFT, FWHT использует только сложения и вычитания,
    что теоретически должно быть быстрее, но на практике numpy FFT 
    оптимизирован на низком уровне.
    
    Требование: длина массива — степень двойки.
    Возвращает массив коэффициентов без нормировки (см. fwht_ortho для орто‑варианта).
    
    Параметры:
    - x: входной массив (длина должна быть степенью двойки)
    
    Возвращает: массив коэффициентов Уолша-Адамара
    """
    n = x.shape[0]
    if not is_power_of_two(n):
        raise ValueError("Длина FWHT должна быть степенью двойки")
    
    y = np.asarray(x, dtype=np.float64).copy()
    
    # Векторизованный алгоритм бабочек
    # На каждом уровне reshaped массив и применяем операции к парам
    h = 1
    while h < n:
        # Reshape для векторизации: (n // (2*h), 2, h)
        # Это позволяет обрабатывать все пары одновременно
        y = y.reshape(-1, 2, h)
        a = y[:, 0, :].copy()  # первые элементы каждой пары
        b = y[:, 1, :].copy()  # вторые элементы каждой пары
        y[:, 0, :] = a + b     # суммы (верхняя половина бабочки)
        y[:, 1, :] = a - b     # разности (нижняя половина бабочки)
        y = y.reshape(-1)      # обратно в плоский массив
        h *= 2
    
    return y.astype(np.float32)


def ifwht(x: np.ndarray) -> np.ndarray:
    """Обратное FWHT в небезопасной нормировке: повторный FWHT и деление на N."""
    n = x.shape[0]
    return fwht(x) / n


def fwht_ortho(x: np.ndarray) -> np.ndarray:
    """Ортонормированное FWHT (деление на sqrt(N))."""
    n = x.shape[0]
    return fwht(x) / math.sqrt(n)


def ifwht_ortho(x: np.ndarray) -> np.ndarray:
    """Обратное ортонормированное FWHT (идентично fwht_ortho)."""
    n = x.shape[0]
    return fwht(x) / math.sqrt(n)


def fwht_ola(
    x: np.ndarray,
    *,
    block_size: int = 2048,
    window: Optional[np.ndarray] = None,
    select_mode: str = "none",  # none | energy | lowpass
    keep_energy_ratio: float = 1.0,
    sequency_keep_ratio: float = 1.0,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> np.ndarray:
    """Блочная FWHT‑обработка с OLA (Overlap-Add) и опциональным отбором коэффициентов.
    
    Метод OLA (Overlap-Add) позволяет обрабатывать сигналы произвольной длины:
    1. Сигнал разбивается на перекрывающиеся блоки (обычно 50% перекрытие)
    2. Каждый блок умножается на анализ-окно (sqrt-Hann)
    3. Применяется преобразование (FWHT)
    4. Выполняется обратное преобразование
    5. Блоки складываются обратно с учётом синтез-окна
    
    Окно sqrt-Hann обеспечивает идеальную реконструкцию при 50% перекрытии:
    w^2[n] + w^2[n+N/2] = 1 для всех n.

    Параметры:
    - x: входной PCM сигнал (float32, диапазон [-1, 1])
    - block_size: размер блока (должен быть степенью двойки, обычно 1024-4096)
    - window: опциональное окно анализа (по умолчанию sqrt-Hann)
    - select_mode: режим отбора коэффициентов:
        * 'none' - без отбора (идеальная реконструкция)
        * 'energy' - сохранение доли энергии (keep_energy_ratio)
        * 'lowpass' - сохранение низкочастотных компонент (sequency_keep_ratio)
    - keep_energy_ratio: доля энергии для сохранения (0..1), только для mode='energy'
    - sequency_keep_ratio: доля коэффициентов (0..1), только для mode='lowpass'
    - progress_cb: колбэк для отображения прогресса

    Возвращает: обработанный сигнал той же длины, что и входной.
    """
    N = int(block_size)
    if not is_power_of_two(N):
        raise ValueError("block_size должен быть степенью двойки")

    hop = N // 2  # 50% перекрытие - оптимально для sqrt-Hann окна
    if hop <= 0:
        hop = N

    # Окно sqrt-Hann: обеспечивает идеальную реконструкцию при 50% перекрытии
    # w^2[n] + w^2[n+hop] = 1 для всех n в пределах блока
    if window is None:
        window = np.sqrt(np.hanning(N) + 1e-12).astype(np.float32)
    else:
        if len(window) != N:
            raise ValueError("длина окна должна совпадать с block_size")

    n = len(x)
    # Число фреймов с 50% перекрытием
    frames = max(1, int(np.ceil(max(0, n - N) / hop)) + 1)
    total_len = (frames - 1) * hop + N
    pad = total_len - n
    
    # Дополнение нулями до полного числа фреймов
    x_padded = np.pad(x, (0, pad), mode="constant")

    # Накопители для OLA
    y_accum = np.zeros_like(x_padded)
    w_accum = np.zeros_like(x_padded)

    select_mode = (select_mode or "none").lower()
    use_energy = (select_mode == "energy") and keep_energy_ratio < 1.0
    use_lowpass = (select_mode == "lowpass") and sequency_keep_ratio < 1.0

    for fi in range(frames):
        i = fi * hop
        blk = x_padded[i : i + N]
        
        # Анализ: применение окна
        blk_w = blk * window
        
        # Прямое FWHT (ортонормированное)
        coeffs = fwht_ortho(blk_w)

        # Отбор коэффициентов (если задан)
        if use_energy:
            # Сохраняем минимальное число коэффициентов, чтобы набрать заданную долю энергии
            # Сортируем по убыванию энергии и берём первые k
            magsq = coeffs * coeffs
            order = np.argsort(magsq)[::-1]  # индексы по убыванию энергии
            cumsum = np.cumsum(magsq[order])
            total_e = cumsum[-1] + 1e-12
            need = keep_energy_ratio * total_e
            keep_n = int(np.searchsorted(cumsum, need, side="left")) + 1
            keep_idx = order[:keep_n]
            mask = np.zeros_like(coeffs, dtype=bool)
            mask[keep_idx] = True
            mask[0] = True  # DC-компонента (среднее) всегда сохраняется
            coeffs = np.where(mask, coeffs, 0.0)
        elif use_lowpass:
            # Сохраняем первые k коэффициентов в порядке Уолша
            # Это аналогично lowpass-фильтрации в частотной области
            k_lp = max(1, int(sequency_keep_ratio * N))
            mask = np.zeros_like(coeffs, dtype=bool)
            mask[:k_lp] = True
            coeffs = np.where(mask, coeffs, 0.0)

        # Синтез: обратное FWHT и применение окна
        rec = ifwht_ortho(coeffs) * window
        
        # OLA: накопление сигнала и весов окна
        y_accum[i : i + N] += rec
        w_accum[i : i + N] += window * window

        if progress_cb:
            progress_cb(min(0.95, (fi + 1) / frames), f"FWHT: блок {fi+1}/{frames}")

    # Нормировка OLA: деление на сумму квадратов окна
    # Это компенсирует перекрытие окон
    y = np.divide(y_accum, np.maximum(w_accum, 1e-8))[:n]
    
    # Защита от клиппинга
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / peak
        
    if progress_cb:
        progress_cb(1.0, "FWHT: готово")
    logger.debug("fwht_ola done n=%d N=%d frames=%d", n, N, frames)
    return y
