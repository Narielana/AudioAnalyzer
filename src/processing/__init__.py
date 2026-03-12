"""
processing: пакет вычислительной части и кодеков.

Переэкспортирует часто используемые функции для удобного импорта в UI:
- fwht_transform_and_mp3 — пайплайн FWHT→MP3
- compare_results — метрики для нескольких вариантов (совместимость)
- standard_convert_to_mp3 — прямое кодирование WAV→MP3
- Метрики качества вынесены в metrics.py
- Утилиты вынесены в utils.py
"""
# Переэкспорт основных API для удобного импорта из UI
from .audio_ops import (
    fwht_transform_and_mp3,
    fft_transform_and_mp3,
    dct_transform_and_mp3,
    wavelet_transform_and_mp3,
    huffman_like_transform_and_mp3,
    rosenbrock_like_transform_and_mp3,
    standard_convert_to_mp3,
    compare_results,
    _compute_metrics_batch,
)
# Переэкспорт метрик для прямого доступа
from .metrics import (
    compute_snr_db,
    compute_rmse,
    compute_si_sdr_db,
    compute_lsd_db,
    compute_spectral_convergence,
    compute_spectral_centroid_diff_hz,
    compute_spectral_cosine_similarity,
    compute_metrics_batch,
)
# Переэкспорт утилит
from .utils import (
    is_power_of_two,
    normalize_ratio,
    parse_int,
    parse_float,
)

__all__ = [
    # Пайплайны
    "fwht_transform_and_mp3",
    "fft_transform_and_mp3",
    "dct_transform_and_mp3",
    "wavelet_transform_and_mp3",
    "huffman_like_transform_and_mp3",
    "rosenbrock_like_transform_and_mp3",
    "standard_convert_to_mp3",
    # Метрики
    "compare_results",
    "_compute_metrics_batch",
    "compute_snr_db",
    "compute_rmse",
    "compute_si_sdr_db",
    "compute_lsd_db",
    "compute_spectral_convergence",
    "compute_spectral_centroid_diff_hz",
    "compute_spectral_cosine_similarity",
    "compute_metrics_batch",
    # Утилиты
    "is_power_of_two",
    "normalize_ratio",
    "parse_int",
    "parse_float",
]
