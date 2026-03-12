"""
УСТАРЕВШИЙ фасад совместимости.

Назначение:
- Сохранить обратную совместимость публичного API ранних версий.
- Переэкспортировать актуальные функции из processing.audio_ops без дубликата логики.

Используемые библиотеки: стандартные typing для подсказок типов.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, List, Dict

# Переэкспорт актуальных функций (сохраняем старые точки входа для внешнего кода)
from .audio_ops import (
    fwht_transform_and_mp3,
    compare_results,
    standard_convert_to_mp3,
)
