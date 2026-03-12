"""
Общие утилиты для модуля processing.

Назначение:
- Разделяемые функции, используемые в нескольких модулях.
- Избежание дублирования кода.

Содержимое:
- is_power_of_two: проверка числа на степень двойки
- normalize_ratio: нормализация параметра в диапазон [0.0, 1.0]
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("audio.processing.utils")


def is_power_of_two(n: int) -> bool:
    """Проверка: n — степень двойки (>0).

    Параметры:
    - n: проверяемое число

    Возвращает: True если n является степенью двойки, False иначе.
    """
    return (n & (n - 1) == 0) and n > 0


def normalize_ratio(value: float, default: float = 1.0) -> float:
    """Нормализация параметра в диапазон [0.0, 1.0].

    Параметры:
    - value: входное значение
    - default: значение по умолчанию при ошибке

    Возвращает: нормализованное значение в диапазоне [0.0, 1.0].
    """
    try:
        return float(max(0.0, min(1.0, float(value))))
    except (TypeError, ValueError) as e:
        logger.debug("normalize_ratio_error: %s, using default %s", e, default)
        return default


def parse_int(value: any, default: int = 0, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """Безопасный парсинг целого числа с ограничениями.

    Параметры:
    - value: входное значение
    - default: значение по умолчанию при ошибке
    - min_val: минимальное допустимое значение (опционально)
    - max_val: максимальное допустимое значение (опционально)

    Возвращает: распарсенное целое число.
    """
    try:
        result = int(value)
    except (TypeError, ValueError) as e:
        logger.debug("parse_int_error: %s, using default %s", e, default)
        result = default

    if min_val is not None:
        result = max(min_val, result)
    if max_val is not None:
        result = min(max_val, result)

    return result


def parse_float(value: any, default: float = 0.0, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Безопасный парсинг числа с плавающей точкой с ограничениями.

    Параметры:
    - value: входное значение
    - default: значение по умолчанию при ошибке
    - min_val: минимальное допустимое значение (опционально)
    - max_val: максимальное допустимое значение (опционально)

    Возвращает: распарсенное число.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as e:
        logger.debug("parse_float_error: %s, using default %s", e, default)
        result = default

    if min_val is not None:
        result = max(min_val, result)
    if max_val is not None:
        result = min(max_val, result)

    return result


# =============================================================================
# ЭКСПОРТ ИМЁН
# =============================================================================

__all__ = [
    "is_power_of_two",
    "normalize_ratio",
    "parse_int",
    "parse_float",
]
