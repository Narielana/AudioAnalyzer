"""
Конфигурация логирования приложения.

Назначение:
- Инициализация корневого логгера с JSON-форматом для файла и консоли.
- Безопасный выбор пути лог-файла с несколькими резервными сценариями.

Внешние библиотеки:
- logging (RotatingFileHandler, Formatter), json: вывод JSON-строк в лог.
- os/tempfile/uuid/datetime/socket: пути, уникальные run_id и контекст среды.

Переменные окружения:
- APP_LOG_PATH — путь к активному лог-файлу, выставляется после инициализации.
"""
import json
import logging
import os
import socket
import time
import uuid
from datetime import datetime
from logging import LogRecord
from logging.handlers import RotatingFileHandler
import tempfile


class JsonFormatter(logging.Formatter):
    """Форматтер: преобразует LogRecord в компактную JSON-строку."""
    def format(self, record: LogRecord) -> str:
        """Собрать словарь полей и сериализовать в JSON (ensure_ascii=False)."""
        data = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "ts_ms": int(record.created * 1000),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "file": record.pathname,
            "line": record.lineno,
            "func": record.funcName,
            "pid": record.process,
            "thread": record.threadName,
        }
        for k, v in getattr(record, "__dict__", {}).items():
            if k not in data and k not in (
                "name","msg","args","levelname","levelno","pathname","filename","module",
                "exc_info","exc_text","stack_info","lineno","funcName","created","msecs",
                "relativeCreated","thread","threadName","processName","process","message",
            ):
                try:
                    json.dumps(v)
                    data[k] = v
                except Exception:
                    data[k] = str(v)
        return json.dumps(data, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """Фильтр: добавляет run_id и host ко всем записям лога."""
    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id
        self.host = socket.gethostname()

    def filter(self, record: LogRecord) -> bool:
        """Внедрить контекст в LogRecord и разрешить запись (True)."""
        record.run_id = self.run_id
        record.host = self.host
        return True


def setup_logging(path_or_dir: str, *, json_logs: bool = True) -> None:
    """Настроить корневое логирование в файл и консоль.

    Поведение:
    - Если дан каталог — создаёт уникальный файл logs/app_YYYY-MM-DD_HHMMSS_<run>.log
    - Если дан путь к .log — использует его напрямую, создавая каталоги по необходимости
    - Ротация файла: 5MB × 3
    - Консоль: INFO+; файл: DEBUG+

    If `path_or_dir` is a directory, a unique file name is created.
    If it looks like a file (.log), it's used as-is.
    - File: DEBUG+ as JSON lines (rotating 5MB x 3)
    - Console: INFO+ as JSON lines (or text if json_logs=False)
    """
    root = logging.getLogger()
    # Reconfigure even if handlers exist
    try:
        for h in list(root.handlers):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            root.removeHandler(h)
    except Exception:
        pass
    root.setLevel(logging.DEBUG)

    run_id = str(uuid.uuid4())
    run_short = run_id.split('-')[0]

    # Decide path with probing and fallbacks
    def _mk(base_dir: str) -> str:
        os.makedirs(base_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        return os.path.join(base_dir, f'app_{ts}_{run_short}.log')

    candidates = []
    try:
        is_dir = os.path.isdir(path_or_dir) or not os.path.splitext(path_or_dir)[1]
    except Exception:
        is_dir = True
    if is_dir:
        candidates.append(_mk(path_or_dir))
    else:
        try:
            os.makedirs(os.path.dirname(path_or_dir), exist_ok=True)
            candidates.append(path_or_dir)
        except Exception:
            pass
    # LocalAppData fallback
    candidates.append(_mk(os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'AudioTransformer', 'logs')))
    # Temp dir fallback
    candidates.append(_mk(os.path.join(tempfile.gettempdir(), 'AudioTransformer', 'logs')))

    log_path = None
    for p in candidates:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'a', encoding='utf-8') as f:
                f.write('')
            log_path = p
            break
        except Exception:
            continue
    if not log_path:
        # as absolute last resort, don't crash; use temp without writing test
        log_path = os.path.join(tempfile.gettempdir(), f'app_{run_short}.log')

    # Context
    filt = ContextFilter(run_id)
    root.addFilter(filt)

    file_fmt = JsonFormatter() if json_logs else logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', '%H:%M:%S')
    stream_fmt = JsonFormatter() if json_logs else file_fmt

    try:
        fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_fmt)
        root.addHandler(fh)
    except Exception:
        # Try temp file directly
        try:
            temp_path = os.path.join(tempfile.gettempdir(), f'app_{run_short}.log')
            fh = RotatingFileHandler(temp_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(file_fmt)
            root.addHandler(fh)
            log_path = temp_path
        except Exception:
            pass

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(stream_fmt)
    root.addHandler(sh)

    os.environ['APP_LOG_PATH'] = log_path
    logging.getLogger("app").info("logging_initialized", extra={"log_path": log_path, "run_id": run_id})
