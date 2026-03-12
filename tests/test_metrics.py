#!/usr/bin/env python3
"""
Unit-тесты для модуля processing.metrics.

Тестирует:
- Метрики временной области (SNR, RMSE, SI-SDR)
- Метрики спектральной области (LSD, Spectral Convergence, Centroid Diff, Cosine Similarity)
- Граничные случаи (пустые сигналы, NaN значения)
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from processing.metrics import (
    compute_snr_db,
    compute_rmse,
    compute_si_sdr_db,
    compute_lsd_db,
    compute_spectral_convergence,
    compute_spectral_centroid_diff_hz,
    compute_spectral_cosine_similarity,
)


class TestTimeDomainMetrics(unittest.TestCase):
    """Тесты метрик временной области."""

    def test_snr_identical_signals(self):
        """SNR для идентичных сигналов должен быть бесконечным или очень высоким."""
        x = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
        snr = compute_snr_db(x, x)
        self.assertTrue(math.isinf(snr) or snr > 100, f"Expected very high SNR, got {snr}")

    def test_snr_different_amplitudes(self):
        """SNR для сигналов с разной амплитудой."""
        ref = np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float32)
        test = np.array([0.9, 0.45, -0.45, -0.9], dtype=np.float32)
        snr = compute_snr_db(ref, test)
        self.assertTrue(10 < snr < 50, f"SNR should be moderate, got {snr}")

    def test_snr_empty_signals(self):
        """SNR для пустых сигналов должен быть NaN."""
        empty = np.array([], dtype=np.float32)
        snr = compute_snr_db(empty, empty)
        self.assertTrue(math.isnan(snr))

    def test_rmse_identical_signals(self):
        """RMSE для идентичных сигналов должен быть 0."""
        x = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
        rmse = compute_rmse(x, x)
        self.assertAlmostEqual(rmse, 0.0, places=6)

    def test_rmse_known_difference(self):
        """RMSE для сигналов с известной разницей."""
        ref = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        test = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rmse = compute_rmse(ref, test)
        self.assertAlmostEqual(rmse, 1.0, places=5)

    def test_rmse_empty_signals(self):
        """RMSE для пустых сигналов должен быть NaN."""
        empty = np.array([], dtype=np.float32)
        rmse = compute_rmse(empty, empty)
        self.assertTrue(math.isnan(rmse))

    def test_si_sdr_scale_invariant(self):
        """SI-SDR должен быть инвариантен к масштабу."""
        ref = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
        test_scaled = ref * 0.5  # Тот же сигнал, но масштабированный
        si_sdr = compute_si_sdr_db(ref, test_scaled)
        # SI-SDR должен быть очень высоким для масштабированной версии того же сигнала
        self.assertTrue(si_sdr > 20, f"SI-SDR should be high for scaled signal, got {si_sdr}")

    def test_si_sdr_empty_signals(self):
        """SI-SDR для пустых сигналов должен быть NaN."""
        empty = np.array([], dtype=np.float32)
        si_sdr = compute_si_sdr_db(empty, empty)
        self.assertTrue(math.isnan(si_sdr))


class TestSpectralMetrics(unittest.TestCase):
    """Тесты метрик спектральной области."""

    def setUp(self):
        """Создание тестовых сигналов."""
        self.sr = 44100
        self.duration = 0.5
        self.t = np.linspace(0, self.duration, int(self.sr * self.duration), dtype=np.float32)
        # Синусоида 440 Гц
        self.ref = (0.5 * np.sin(2 * np.pi * 440 * self.t)).astype(np.float32)
        # Синусоида с немного другой амплитудой
        self.test = (0.45 * np.sin(2 * np.pi * 440 * self.t)).astype(np.float32)

    def test_lsd_identical_signals(self):
        """LSD для идентичных сигналов должен быть около 0."""
        lsd = compute_lsd_db(self.ref, self.ref, self.sr, self.sr)
        self.assertAlmostEqual(lsd, 0.0, places=2)

    def test_lsd_similar_signals(self):
        """LSD для похожих сигналов должен быть низким."""
        lsd = compute_lsd_db(self.ref, self.test, self.sr, self.sr)
        self.assertTrue(0 < lsd < 5, f"LSD should be low, got {lsd}")

    def test_lsd_empty_signals(self):
        """LSD для пустых сигналов должен быть NaN."""
        empty = np.array([], dtype=np.float32)
        lsd = compute_lsd_db(empty, empty, self.sr, self.sr)
        self.assertTrue(math.isnan(lsd))

    def test_spectral_convergence_identical(self):
        """Spectral Convergence для идентичных сигналов должен быть около 0."""
        sc = compute_spectral_convergence(self.ref, self.ref, self.sr, self.sr)
        self.assertAlmostEqual(sc, 0.0, places=3)

    def test_spectral_convergence_similar(self):
        """Spectral Convergence для похожих сигналов должен быть низким."""
        sc = compute_spectral_convergence(self.ref, self.test, self.sr, self.sr)
        self.assertTrue(0 <= sc < 0.5, f"SC should be low, got {sc}")

    def test_spectral_centroid_diff_identical(self):
        """Разница центроидов для идентичных сигналов должна быть около 0."""
        scd = compute_spectral_centroid_diff_hz(self.ref, self.ref, self.sr, self.sr)
        self.assertAlmostEqual(scd, 0.0, places=1)

    def test_spectral_cosine_similarity_identical(self):
        """Косинусная схожесть для идентичных сигналов должна быть около 1."""
        cs = compute_spectral_cosine_similarity(self.ref, self.ref, self.sr, self.sr)
        self.assertAlmostEqual(cs, 1.0, places=3)

    def test_spectral_cosine_similarity_similar(self):
        """Косинусная схожесть для похожих сигналов должна быть высокой."""
        cs = compute_spectral_cosine_similarity(self.ref, self.test, self.sr, self.sr)
        self.assertTrue(cs > 0.9, f"Cosine similarity should be high, got {cs}")


class TestEdgeCases(unittest.TestCase):
    """Тесты граничных случаев."""

    def test_different_lengths(self):
        """Метрики должны работать с сигналами разной длины."""
        short = np.ones(100, dtype=np.float32)
        long = np.ones(200, dtype=np.float32)
        
        snr = compute_snr_db(long, short)
        rmse = compute_rmse(long, short)
        
        # Должны вернуть валидные значения
        self.assertTrue(math.isfinite(snr))
        self.assertAlmostEqual(rmse, 0.0, places=5)

    def test_very_short_signal(self):
        """Метрики должны работать с очень короткими сигналами."""
        short = np.array([0.5, -0.5], dtype=np.float32)
        sr = 8000
        
        lsd = compute_lsd_db(short, short, sr, sr)
        self.assertTrue(math.isfinite(lsd) or math.isnan(lsd))

    def test_zero_signal(self):
        """Метрики для нулевого сигнала."""
        zero = np.zeros(1000, dtype=np.float32)
        nonzero = np.ones(1000, dtype=np.float32)
        
        # RMSE должен работать
        rmse = compute_rmse(zero, nonzero)
        self.assertAlmostEqual(rmse, 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
