"""
Централизованная конфигурация приложения AudioAnalyzer.

Назначение:
- Единая точка для всех настраиваемых параметров.
- Параметры по умолчанию для пайплайнов обработки.
- Пути к директориям проекта.

Использование:
    from config import Config
    block_size = Config.DEFAULT_BLOCK_SIZE
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# =============================================================================
# ПУТИ ПРОЕКТА
# =============================================================================

# Корневая директория проекта
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()

# Директория с тестовыми данными
DEFAULT_TEST_DATA_DIR: Path = PROJECT_ROOT / "default_test_data"

# Директория для выходных файлов
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# Директория для логов
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Директория с исходным кодом
SRC_DIR: Path = PROJECT_ROOT / "src"


# =============================================================================
# КОНФИГУРАЦИЯ ОБРАБОТКИ
# =============================================================================

@dataclass
class ProcessingConfig:
    """Параметры обработки аудио."""

    # Размер блока для блочных преобразований (должен быть степенью двойки)
    block_size: int = 2048

    # Битрейт MP3
    bitrate: str = "192k"

    # Режим отбора коэффициентов: "none", "energy", "lowpass"
    select_mode: str = "none"

    # Доля сохраняемой энергии (для режима "energy")
    keep_energy_ratio: float = 1.0

    # Доля сохраняемых низких частот (для режима "lowpass")
    sequency_keep_ratio: float = 1.0

    # Число уровней DWT (вейвлет-преобразование)
    dwt_levels: int = 4

    # Параметр μ для μ-law компандирования (Huffman-like)
    mu: float = 255.0

    # Число бит квантования (Huffman-like)
    bits: int = 8

    # Параметры Rosenbrock-like преобразования
    rosen_alpha: float = 0.2
    rosen_beta: float = 1.0

    def to_dict(self) -> Dict[str, any]:
        """Преобразовать в словарь."""
        return {
            "block_size": self.block_size,
            "bitrate": self.bitrate,
            "select_mode": self.select_mode,
            "keep_energy_ratio": self.keep_energy_ratio,
            "sequency_keep_ratio": self.sequency_keep_ratio,
            "levels": self.dwt_levels,
            "mu": self.mu,
            "bits": self.bits,
            "rosen_alpha": self.rosen_alpha,
            "rosen_beta": self.rosen_beta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, any]) -> "ProcessingConfig":
        """Создать из словаря."""
        return cls(
            block_size=d.get("block_size", 2048),
            bitrate=d.get("bitrate", "192k"),
            select_mode=d.get("select_mode", "none"),
            keep_energy_ratio=d.get("keep_energy_ratio", 1.0),
            sequency_keep_ratio=d.get("sequency_keep_ratio", 1.0),
            dwt_levels=d.get("levels", 4),
            mu=d.get("mu", 255.0),
            bits=d.get("bits", 8),
            rosen_alpha=d.get("rosen_alpha", 0.2),
            rosen_beta=d.get("rosen_beta", 1.0),
        )


# =============================================================================
# КОНФИГУРАЦИЯ МЕТРИК
# =============================================================================

@dataclass
class MetricsConfig:
    """Параметры вычисления метрик."""

    # Размер окна FFT для спектральных метрик
    n_fft: int = 1024

    # Шаг окна (hop size)
    hop: int = 512

    # Веса для агрегированного балла
    weight_lsd: float = 0.25
    weight_spectral_conv: float = 0.20
    weight_snr: float = 0.15
    weight_rmse: float = 0.15
    weight_si_sdr: float = 0.10
    weight_centroid_diff: float = 0.05
    weight_cosine_sim: float = 0.05
    weight_time: float = 0.05


# =============================================================================
# КОНФИГУРАЦИЯ UI
# =============================================================================

@dataclass
class UIConfig:
    """Параметры пользовательского интерфейса."""

    # Размеры окна
    window_width: int = 1100
    window_height: int = 650
    window_title: str = "Audio Transformer"

    # Настройки таблицы
    table_sorting_enabled: bool = True

    # Показывать логи по умолчанию
    show_logs_by_default: bool = True


# =============================================================================
# ПРЕСЕТЫ
# =============================================================================

PRESETS: Dict[str, ProcessingConfig] = {
    "Стандартный": ProcessingConfig(),
    "Быстрый": ProcessingConfig(
        block_size=1024,
        bitrate="160k",
        select_mode="none",
        dwt_levels=3,
    ),
    "Качество": ProcessingConfig(
        block_size=4096,
        bitrate="256k",
        select_mode="energy",
        keep_energy_ratio=0.95,
        dwt_levels=5,
        bits=12,
        rosen_alpha=0.15,
    ),
    "Лучший средний": ProcessingConfig(
        block_size=2048,
        bitrate="192k",
        select_mode="energy",
        keep_energy_ratio=0.9,
        sequency_keep_ratio=0.9,
        bits=10,
    ),
}


# =============================================================================
# ЭКСПОРТ
# =============================================================================

__all__ = [
    # Пути
    "PROJECT_ROOT",
    "DEFAULT_TEST_DATA_DIR",
    "OUTPUT_DIR",
    "LOGS_DIR",
    "SRC_DIR",
    # Конфигурации
    "ProcessingConfig",
    "MetricsConfig",
    "UIConfig",
    # Пресеты
    "PRESETS",
]
