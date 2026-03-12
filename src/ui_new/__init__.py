"""
ui_new — единый модуль пользовательского интерфейса AudioAnalyzer.

Содержимое:
- main_window.py: Главное окно приложения
- worker.py: Фоновая обработка аудио
- constants.py: Константы UI (VARIANTS, METRIC_KEYS, ...)
- presets.py: Пресеты настроек
- log_handler.py: Логирование в UI

Использование:
    from ui_new.main_window import MainWindow
    # или
    from ui_new import MainWindow
"""

from .main_window import MainWindow
from .worker import Worker, ResultRow
from .constants import VARIANTS, METRIC_KEYS, TABLE_HEADERS
from .presets import PRESET_NAMES
from .presets import apply_preset
from .log_handler import QtLogHandler, UiLogEmitter

__all__ = [
    "MainWindow",
    "Worker",
    "ResultRow",
    "VARIANTS",
    "METRIC_KEYS",
    "TABLE_HEADERS",
    "PRESET_NAMES",
    "apply_preset",
    "QtLogHandler",
    "UiLogEmitter",
]
