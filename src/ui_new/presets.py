"""
Пресеты настроек для UI.

Назначение:
- Предустановленные конфигурации параметров обработки.
- Упрощение выбора оптимальных настроек для разных сценариев.
"""
from __future__ import annotations

from typing import Any, List

from PySide6.QtWidgets import QLineEdit, QComboBox

from .constants import SELECT_MODES


# =============================================================================
# ИМЕНА ПРЕСЕТОВ
# =============================================================================

PRESET_NAMES: List[str] = [
    "Стандартный",
    "Без сжатия (identity)",
    "Быстрый",
    "Качество",
    "Лучший средний",
    "Std — Скорость",
    "Std — Качество",
    "Std — Размер",
    "FWHT — Скорость",
    "FWHT — Качество",
    "FWHT — Размер",
    "FFT — Скорость",
    "FFT — Качество",
    "FFT — Размер",
    "DCT — Скорость",
    "DCT — Качество",
    "DCT — Размер",
    "DWT — Скорость",
    "DWT — Качество",
    "DWT — Размер",
    "Хаффман — Скорость",
    "Хаффман — Качество",
    "Хаффман — Размер",
    "Розенброк — Скорость",
    "Розенброк — Качество",
    "Розенброк — Размер",
]


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def _set_text(widget: Any, value: str) -> None:
    """Безопасно установить текст в QLineEdit."""
    try:
        if isinstance(widget, QLineEdit):
            widget.setText(value)
    except Exception:
        pass


def _set_combo_by_data(combo: Any, data_value: str) -> None:
    """Выбрать элемент QComboBox по userData."""
    try:
        if isinstance(combo, QComboBox):
            for i in range(combo.count()):
                if combo.itemData(i) == data_value:
                    combo.setCurrentIndex(i)
                    return
    except Exception:
        pass


def _set_combo_by_text(combo: Any, text: str) -> None:
    """Выбрать элемент QComboBox по тексту."""
    try:
        if isinstance(combo, QComboBox):
            for i in range(combo.count()):
                if combo.itemText(i) == text:
                    combo.setCurrentIndex(i)
                    return
    except Exception:
        pass


# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ПРИМЕНЕНИЯ ПРЕСЕТА
# =============================================================================

def apply_preset(owner: Any, preset_name: str) -> None:
    """Применить именованный пресет к виджетам настроек.

    Автоматически определяет имена виджетов (поддерживает два стиля):
    - Стиль v1: setting_block_size_edit, setting_bitrate_edit, ...
    - Стиль v2: ed_block, ed_bitrate, ...

    Параметры:
    - owner: объект, содержащий виджеты настроек
    - preset_name: имя пресета из списка PRESET_NAMES
    """
    name = preset_name or ""

    # Определяем виджеты (поддерживаем оба стиля имён)
    ed_block = getattr(owner, 'ed_block', None) or getattr(owner, 'setting_block_size_edit', None)
    ed_bitrate = getattr(owner, 'ed_bitrate', None) or getattr(owner, 'setting_bitrate_edit', None)
    cb_select = getattr(owner, 'cb_select', None) or getattr(owner, 'setting_select_mode_combo', None)
    ed_keep_energy = getattr(owner, 'ed_keep_energy', None) or getattr(owner, 'setting_keep_energy_edit', None)
    ed_seq_keep = getattr(owner, 'ed_seq_keep', None) or getattr(owner, 'setting_sequency_keep_edit', None)
    ed_levels = getattr(owner, 'ed_levels', None) or getattr(owner, 'setting_levels_edit', None)
    ed_mu = getattr(owner, 'ed_mu', None) or getattr(owner, 'setting_mu_edit', None)
    ed_bits = getattr(owner, 'ed_bits', None) or getattr(owner, 'setting_bits_edit', None)
    ed_ra = getattr(owner, 'ed_ra', None) or getattr(owner, 'setting_rosen_alpha_edit', None)
    ed_rb = getattr(owner, 'ed_rb', None) or getattr(owner, 'setting_rosen_beta_edit', None)

    # -------------------------------------------------------------------------
    # ПРЕСЕТЫ
    # -------------------------------------------------------------------------

    if name == "Без сжатия (identity)":
        # Без сжатия - все методы дадут одинаковый результат (identity transform)
        # Полезно для проверки корректности реализации
        _set_text(ed_block, "2048")
        _set_text(ed_bitrate, "192k")
        _set_combo_by_data(cb_select, "none")  # Без отбора - identity
        _set_text(ed_keep_energy, "1.0")
        _set_text(ed_seq_keep, "1.0")
        _set_text(ed_levels, "4")
        _set_text(ed_mu, "255")
        _set_text(ed_bits, "16")  # 16 бит = без потерь
        _set_text(ed_ra, "0.0")   # α=0 = без искажений
        _set_text(ed_rb, "1.0")

    elif name == "Быстрый":
        # Приоритет скорости
        _set_text(ed_block, "1024")
        _set_text(ed_bitrate, "160k")
        _set_combo_by_data(cb_select, "none")
        _set_text(ed_keep_energy, "1.0")
        _set_text(ed_seq_keep, "1.0")
        _set_text(ed_levels, "3")
        _set_text(ed_mu, "255")
        _set_text(ed_bits, "8")
        _set_text(ed_ra, "0.2")
        _set_text(ed_rb, "1.0")

    elif name == "Качество":
        # Приоритет качества
        _set_text(ed_block, "4096")
        _set_text(ed_bitrate, "256k")
        _set_combo_by_data(cb_select, "energy")
        _set_text(ed_keep_energy, "0.95")
        _set_text(ed_seq_keep, "1.0")
        _set_text(ed_levels, "5")
        _set_text(ed_mu, "255")
        _set_text(ed_bits, "12")
        _set_text(ed_ra, "0.15")
        _set_text(ed_rb, "1.0")

    elif name == "Лучший средний":
        # Баланс качества и скорости
        _set_text(ed_block, "2048")
        _set_text(ed_bitrate, "192k")
        _set_combo_by_data(cb_select, "energy")
        _set_text(ed_keep_energy, "0.9")
        _set_text(ed_seq_keep, "0.9")
        _set_text(ed_levels, "4")
        _set_text(ed_mu, "255")
        _set_text(ed_bits, "10")
        _set_text(ed_ra, "0.2")
        _set_text(ed_rb, "1.0")

    # Std пресеты
    elif name == "Std — Скорость":
        _set_text(ed_bitrate, "160k")

    elif name == "Std — Качество":
        _set_text(ed_bitrate, "256k")

    elif name == "Std — Размер":
        _set_text(ed_bitrate, "128k")

    # FWHT пресеты
    elif name == "FWHT — Скорость":
        _set_text(ed_block, "1024")
        _set_combo_by_data(cb_select, "none")
        _set_text(ed_keep_energy, "1.0")
        _set_text(ed_seq_keep, "1.0")
        _set_text(ed_bitrate, "160k")

    elif name == "FWHT — Качество":
        _set_text(ed_block, "4096")
        _set_combo_by_data(cb_select, "energy")
        _set_text(ed_keep_energy, "0.98")
        _set_text(ed_seq_keep, "1.0")
        _set_text(ed_bitrate, "256k")

    elif name == "FWHT — Размер":
        _set_text(ed_block, "2048")
        _set_combo_by_data(cb_select, "lowpass")
        _set_text(ed_keep_energy, "0.90")
        _set_text(ed_seq_keep, "0.5")
        _set_text(ed_bitrate, "128k")

    # FFT пресеты
    elif name == "FFT — Скорость":
        _set_text(ed_block, "1024")
        _set_combo_by_data(cb_select, "none")
        _set_text(ed_keep_energy, "1.0")
        _set_text(ed_seq_keep, "1.0")
        _set_text(ed_bitrate, "160k")

    elif name == "FFT — Качество":
        _set_text(ed_block, "4096")
        _set_combo_by_data(cb_select, "energy")
        _set_text(ed_keep_energy, "0.98")
        _set_text(ed_seq_keep, "1.0")
        _set_text(ed_bitrate, "256k")

    elif name == "FFT — Размер":
        _set_text(ed_block, "2048")
        _set_combo_by_data(cb_select, "lowpass")
        _set_text(ed_keep_energy, "0.90")
        _set_text(ed_seq_keep, "0.5")
        _set_text(ed_bitrate, "128k")

    # DCT пресеты
    elif name == "DCT — Скорость":
        _set_text(ed_block, "1024")
        _set_combo_by_data(cb_select, "none")
        _set_text(ed_keep_energy, "1.0")
        _set_text(ed_seq_keep, "1.0")
        _set_text(ed_bitrate, "160k")

    elif name == "DCT — Качество":
        _set_text(ed_block, "4096")
        _set_combo_by_data(cb_select, "energy")
        _set_text(ed_keep_energy, "0.95")
        _set_text(ed_seq_keep, "1.0")
        _set_text(ed_bitrate, "256k")

    elif name == "DCT — Размер":
        _set_text(ed_block, "2048")
        _set_combo_by_data(cb_select, "lowpass")
        _set_text(ed_keep_energy, "0.90")
        _set_text(ed_seq_keep, "0.6")
        _set_text(ed_bitrate, "128k")

    # DWT пресеты
    elif name == "DWT — Скорость":
        _set_text(ed_block, "1024")
        _set_combo_by_data(cb_select, "none")
        _set_text(ed_keep_energy, "1.0")
        _set_text(ed_seq_keep, "1.0")
        _set_text(ed_levels, "3")
        _set_text(ed_bitrate, "160k")

    elif name == "DWT — Качество":
        _set_text(ed_block, "4096")
        _set_combo_by_data(cb_select, "energy")
        _set_text(ed_keep_energy, "0.95")
        _set_text(ed_seq_keep, "1.0")
        _set_text(ed_levels, "5")
        _set_text(ed_bitrate, "256k")

    elif name == "DWT — Размер":
        _set_text(ed_block, "2048")
        _set_combo_by_data(cb_select, "lowpass")
        _set_text(ed_keep_energy, "0.90")
        _set_text(ed_seq_keep, "0.6")
        _set_text(ed_levels, "6")
        _set_text(ed_bitrate, "128k")

    # Huffman пресеты
    elif name == "Хаффман — Скорость":
        _set_text(ed_mu, "255")
        _set_text(ed_bits, "8")
        _set_text(ed_bitrate, "160k")

    elif name == "Хаффман — Качество":
        _set_text(ed_mu, "255")
        _set_text(ed_bits, "12")
        _set_text(ed_bitrate, "256k")

    elif name == "Хаффман — Размер":
        _set_text(ed_mu, "255")
        _set_text(ed_bits, "8")
        _set_text(ed_bitrate, "128k")

    # Rosenbrock пресеты
    elif name == "Розенброк — Скорость":
        _set_text(ed_ra, "0.2")
        _set_text(ed_rb, "1.0")
        _set_text(ed_bitrate, "160k")

    elif name == "Розенброк — Качество":
        _set_text(ed_ra, "0.15")
        _set_text(ed_rb, "1.0")
        _set_text(ed_bitrate, "256k")

    elif name == "Розенброк — Размер":
        _set_text(ed_ra, "0.2")
        _set_text(ed_rb, "1.0")
        _set_text(ed_bitrate, "128k")

    else:
        # Стандартный пресет (по умолчанию) - с умеренным сжатием для демонстрации разницы методов
        _set_text(ed_block, "2048")
        _set_text(ed_bitrate, "192k")
        _set_combo_by_data(cb_select, "energy")  # Включаем отбор по энергии
        _set_text(ed_keep_energy, "0.95")  # Сохраняем 95% энергии - покажет разницу
        _set_text(ed_seq_keep, "1.0")
        _set_text(ed_levels, "4")
        _set_text(ed_mu, "255")
        _set_text(ed_bits, "8")
        _set_text(ed_ra, "0.2")
        _set_text(ed_rb, "1.0")
