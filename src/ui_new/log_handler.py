"""
Логирование в UI: Qt-совместимый обработчик логов.

Назначение:
- Перенаправление Python logging в Qt-виджеты через сигналы.
- Форматирование лог-сообщений для отображения в панели UI.

Внешние зависимости: PySide6 (QObject, Signal), logging, datetime.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from PySide6.QtCore import QObject, Signal


class UiLogEmitter(QObject):
    """Qt-эмиттер для передачи лог-сообщений в UI.

    Используется как мост между Python logging и Qt-виджетами.
    Подключите сигнал log_line к слоту текстового виджета.
    """
    log_line = Signal(str)


class QtLogHandler(logging.Handler):
    """Обработчик логов для вывода в Qt UI.

    Форматирует записи лога в читаемые строки и отправляет их
    через UiLogEmitter в текстовую панель интерфейса.

    Пример использования:
        emitter = UiLogEmitter()
        emitter.log_line.connect(text_edit.appendPlainText)
        handler = QtLogHandler(emitter)
        logging.getLogger().addHandler(handler)
    """

    def __init__(self, emitter: UiLogEmitter, *, show_timestamp: bool = True):
        """Инициализация обработчика.

        Параметры:
        - emitter: UiLogEmitter для отправки сообщений в UI
        - show_timestamp: показывать ли временную метку в логах
        """
        super().__init__()
        self.emitter = emitter
        self._show_timestamp = show_timestamp
        self._formatter: Optional[logging.Formatter] = None

    def emit(self, record: logging.LogRecord) -> None:
        """Обработать запись лога и отправить в UI.

        Формат: "HH:MM:SS [LEVEL] logger: message"
        При наличии исключения добавляется трассировка.
        """
        try:
            # Формируем временную метку
            if self._show_timestamp:
                ts = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
                line = f"{ts} [{record.levelname}] {record.name}: {record.getMessage()}"
            else:
                line = f"[{record.levelname}] {record.name}: {record.getMessage()}"

            # Добавляем трассировку исключения, если есть
            if record.exc_info:
                try:
                    if self._formatter is None:
                        self._formatter = logging.Formatter()
                    exc_text = self._formatter.formatException(record.exc_info)
                    line += "\n" + exc_text
                except Exception:
                    pass

            # Отправляем в UI через сигнал
            self.emitter.log_line.emit(line)

        except Exception:
            # При любой ошибке пытаемся отправить хотя бы базовое сообщение
            try:
                self.emitter.log_line.emit(record.getMessage())
            except Exception:
                pass

    def setTimestampVisible(self, visible: bool) -> None:
        """Установить видимость временной метки."""
        self._show_timestamp = visible
