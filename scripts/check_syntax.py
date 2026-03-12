#!/usr/bin/env python3
"""
Проверка синтаксиса Python файлов проекта.

Использование:
    python check_syntax.py                    # проверить основные файлы
    python check_syntax.py --all              # проверить все .py файлы
    python check_syntax.py path/to/file.py    # проверить конкретный файл
"""
import argparse
import os
import py_compile
import sys
from pathlib import Path


def get_default_files() -> list[str]:
    """Получить список основных файлов для проверки по умолчанию."""
    # Определяем базовую директорию проекта (родитель scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    src_dir = project_root / "src"
    
    files = [
        # Точка входа
        src_dir / "app.py",
        # Processing модуль
        src_dir / "processing" / "audio_ops.py",
        src_dir / "processing" / "codecs.py",
        src_dir / "processing" / "metrics.py",
        src_dir / "processing" / "fwht.py",
        src_dir / "processing" / "api.py",
        # UI модуль (новый)
        src_dir / "ui_new" / "main_window.py",
        src_dir / "ui_new" / "worker.py",
        src_dir / "ui_new" / "presets.py",
        src_dir / "ui_new" / "constants.py",
        src_dir / "ui_new" / "log_handler.py",
        # Utils
        src_dir / "utils" / "logging_setup.py",
    ]
    
    # Фильтруем существующие файлы
    return [str(f) for f in files if f.exists()]


def get_all_python_files() -> list[str]:
    """Получить список всех Python файлов в проекте."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    src_dir = project_root / "src"
    
    files = []
    for pattern in ["**/*.py"]:
        for f in src_dir.glob(pattern):
            # Исключаем __pycache__
            if "__pycache__" in str(f):
                continue
            files.append(str(f))
    
    return sorted(files)


def check_syntax(filepath: str) -> bool:
    """Проверить синтаксис одного файла.
    
    Возвращает True если синтаксис корректный, False иначе.
    """
    try:
        py_compile.compile(filepath, doraise=True)
        print(f"✓ OK: {filepath}")
        return True
    except py_compile.PyCompileError as e:
        print(f"✗ ERROR: {filepath}", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"✗ NOT FOUND: {filepath}", file=sys.stderr)
        return False


def main() -> int:
    """Главная функция проверки синтаксиса.
    
    Возвращает количество файлов с ошибками.
    """
    parser = argparse.ArgumentParser(
        description="Проверка синтаксиса Python файлов проекта AudioAnalyzer"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Файлы для проверки (по умолчанию основные файлы проекта)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Проверить все Python файлы в проекте"
    )
    
    args = parser.parse_args()
    
    # Определяем список файлов для проверки
    if args.files:
        files = args.files
    elif args.all:
        files = get_all_python_files()
    else:
        files = get_default_files()
    
    if not files:
        print("Нет файлов для проверки", file=sys.stderr)
        return 1
    
    print(f"Проверка синтаксиса {len(files)} файлов...\n")
    
    errors = 0
    for filepath in files:
        if not check_syntax(filepath):
            errors += 1
    
    print(f"\n{'=' * 50}")
    print(f"Проверено: {len(files)}, ошибок: {errors}")
    
    return errors


if __name__ == "__main__":
    sys.exit(main())
