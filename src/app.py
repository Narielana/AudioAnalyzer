"""
Точка входа настольного приложения Audio Transformer.

Назначение:
- Инициализировать структурированное логирование (JSON) и каталоги логов.
- Загрузить и запустить класс главного окна Qt (ui_new).
- Показать понятные сообщения об ошибках при проблемах.

Внешние библиотеки:
- PySide6: графический интерфейс (QApplication, QMessageBox).
- logging: корневое логирование.

Переменные окружения:
- APP_LOG_DIR, APP_LOG_PATH — выставляются в utils.logging_setup.
"""
from PySide6.QtWidgets import QApplication, QMessageBox
import logging
import os
import sys
from importlib import import_module


# =============================================================================
# ИМПОРТ MAINWINDOW
# =============================================================================

# Прямой импорт из нового модуля ui_new
try:
    from ui_new.main_window import MainWindow  # type: ignore
except ImportError:
    try:
        from src.ui_new.main_window import MainWindow  # type: ignore
    except ImportError as e:
        MainWindow = None  # Будет обработано в main()


# =============================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# =============================================================================

def _setup_logging() -> None:
    """Подготовить каталог логов и сконфигурировать корневой логгер.

    Пробует несколько вариантов расположения логов:
    1. ./logs (рядом с приложением)
    2. LocalAppData/AudioTransformer/logs (Windows)
    3. TEMP/AudioTransformer/logs
    """
    # Импорт модуля конфигурации логов
    _logging_setup = None
    for module_path in ['utils.logging_setup', 'src.utils.logging_setup']:
        try:
            _logging_setup = import_module(module_path)
            break
        except ImportError:
            continue

    if _logging_setup is None:
        return  # Продолжаем без файлового логирования

    # Определение базовой директории
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.getcwd()

    # Кандидаты для логов
    candidates = [
        os.path.join(base_dir, 'logs'),
        os.path.join(
            os.environ.get('LOCALAPPDATA', os.path.expanduser('~')),
            'AudioTransformer', 'logs'
        ),
        os.path.join(
            os.environ.get('TMP', os.environ.get('TEMP', os.getcwd())),
            'AudioTransformer', 'logs'
        ),
    ]

    # Пробуем создать директорию и настроить логирование
    for log_dir in candidates:
        try:
            os.makedirs(log_dir, exist_ok=True)
            # Проверка записи
            probe_file = os.path.join(log_dir, 'probe.txt')
            with open(probe_file, 'a', encoding='utf-8') as f:
                f.write('')
            # Настройка логирования
            _logging_setup.setup_logging(log_dir, json_logs=True)
            os.environ['APP_LOG_DIR'] = log_dir
            return
        except Exception:
            continue


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

def main() -> int:
    """Запуск QApplication и отображение MainWindow.

    Возвращает код выхода приложения (int).
    """
    _setup_logging()
    log = logging.getLogger("app")
    log.info("app_start")

    app = QApplication(sys.argv)

    # Проверяем, что MainWindow загружен
    if MainWindow is None:
        log.error("main_window_import_failed")
        QMessageBox.critical(
            None,
            "Ошибка импорта",
            "Не удалось загрузить модуль UI. Проверьте установку PySide6."
        )
        return 1

    # Создаём и отображаем окно
    try:
        win = MainWindow()
        win.show()
        win.raise_()
        win.activateWindow()
    except Exception as e:
        log.exception("main_window_creation_error")
        QMessageBox.critical(
            None,
            "Ошибка создания окна",
            f"Не удалось создать главное окно:\n{e}"
        )
        return 1

    # Запуск цикла событий
    rc = app.exec()
    log.info("app_exit", extra={"exit_code": rc})
    return rc


if __name__ == "__main__":
    sys.exit(main())
