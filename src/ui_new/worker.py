"""
Фоновый обработчик аудио для UI.

Назначение:
- Выполнение пайплайнов обработки аудио в отдельном потоке.
- Сбор метрик качества для всех методов.
- Прогресс и ETA для пользовательского интерфейса.

Внешние зависимости: PySide6 (QObject, Signal, Slot), processing.audio_ops.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Signal, Slot

# Импорт функций обработки с поддержкой разных режимов запуска
try:
    from processing.audio_ops import (
        fwht_transform_and_mp3,
        fft_transform_and_mp3,
        dct_transform_and_mp3,
        wavelet_transform_and_mp3,
        huffman_like_transform_and_mp3,
        rosenbrock_like_transform_and_mp3,
        _compute_metrics_batch,
        standard_convert_to_mp3,
    )
except ImportError:
    from src.processing.audio_ops import (
        fwht_transform_and_mp3,
        fft_transform_and_mp3,
        dct_transform_and_mp3,
        wavelet_transform_and_mp3,
        huffman_like_transform_and_mp3,
        rosenbrock_like_transform_and_mp3,
        _compute_metrics_batch,
        standard_convert_to_mp3,
    )


logger = logging.getLogger("ui_new.worker")


# =============================================================================
# РЕЗУЛЬТАТ ОБРАБОТКИ
# =============================================================================

@dataclass
class ResultRow:
    """Структурированный результат обработки одного метода."""
    source: str
    genre: Optional[str]
    variant: str
    path: str
    size_bytes: int
    lsd_db: float
    snr_db: float
    spec_conv: float
    rmse: float
    si_sdr_db: float
    spec_centroid_diff_hz: float
    spec_cosine: float
    score: float
    time_sec: float


# =============================================================================
# РАБОЧИЙ ПОТОК
# =============================================================================

class Worker(QObject):
    """Фоновая обработка WAV-файлов с запуском всех методов и сбором метрик.

    Сигналы:
    - result(object): результат обработки файла
    - error(str): сообщение об ошибке
    - status(str): строка статуса с ETA
    - progress_file(int): прогресс текущего файла (0-100)
    - progress_total(int): прогресс всего набора (0-100)
    - finished(): завершение всех задач
    """

    # Сигналы
    result = Signal(object)       # payload: {source, genre, results}
    error = Signal(str)           # сообщение об ошибке
    status = Signal(str)          # статус с ETA
    progress_file = Signal(int)   # 0-100 по текущему файлу
    progress_total = Signal(int)  # 0-100 по набору
    finished = Signal()           # завершение

    def __init__(
        self,
        wav_paths: List[str],
        out_dir: str,
        dataset_root: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ):
        """Инициализация Worker.

        Параметры:
        - wav_paths: список путей к WAV-файлам
        - out_dir: директория для вывода MP3
        - dataset_root: корень набора (для определения жанров) или None
        - settings: параметры пайплайнов
        """
        super().__init__()
        self.wav_paths = wav_paths
        self.out_dir = out_dir
        self.dataset_root = dataset_root
        self.settings = settings or {}
        self._total = max(1, len(wav_paths))
        self._is_batch = bool(dataset_root)

        # Логирование
        self._log = logging.getLogger("ui_new.worker")

        # Флаг отмены
        self._cancelled = False

        # Время
        self._batch_t0: Optional[float] = None
        self._cur_file_t0: Optional[float] = None

        # Статистика для ETA (7 методов = 7 стадий)
        self._stage_total = 7
        self._stage_stats: Dict[int, Tuple[float, int]] = {}

    def cancel(self) -> None:
        """Запросить отмену обработки."""
        self._cancelled = True
        self._log.info("cancel_requested")

    def is_cancelled(self) -> bool:
        """Проверить, был ли запрошен отмену."""
        return self._cancelled
    # =========================================================================
    # ETA ФОРМАТИРОВАНИЕ
    # =========================================================================

    def _fmt_eta(self, seconds: float) -> str:
        """Форматировать секунды в строку ETA (H:MM:SS или MM:SS)."""
        try:
            if seconds != seconds or seconds < 0:  # NaN check
                return "—"
            s = int(seconds + 0.5)
            h = s // 3600
            m = (s % 3600) // 60
            sec = s % 60
            if h > 0:
                return f"{h}:{m:02d}:{sec:02d}"
            else:
                return f"{m:02d}:{sec:02d}"
        except Exception:
            return "—"

    def _status_with_eta(
        self,
        base: str,
        file_frac: float,
        processed_done: int,
        total_files: int,
    ) -> str:
        """Строка статуса с ETA (упрощённый API)."""
        try:
            return self._status_with_eta_cycle(base, 0, file_frac, processed_done, total_files)
        except Exception:
            return base

    def _status_with_eta_cycle(
        self,
        base: str,
        stage_idx: int,
        stage_frac: float,
        processed_done: int,
        total_files: int,
    ) -> str:
        """Расчёт ETA на основе статистики стадий."""
        try:
            n = max(1, self._stage_total)
            sf = max(0.0, min(1.0, float(stage_frac)))
            si = max(0, min(n - 1, int(stage_idx)))

            # Средние длительности стадий
            avg = []
            for i in range(n):
                s = self._stage_stats.get(i)
                if s and s[1] > 0:
                    avg.append(max(1e-6, s[0] / float(s[1])))
                else:
                    avg.append(float('nan'))

            # Фоллбек: равномерная оценка из текущего elapsed
            now = time.perf_counter()
            t_file_elapsed = max(1e-6, (now - (self._cur_file_t0 or now)))

            if not any(v == v for v in avg):
                base_unit = t_file_elapsed / max(1.0, (si + max(sf, 1e-3)))
                avg = [base_unit for _ in range(n)]
            else:
                known = [v for v in avg if v == v]
                fill = (sum(known) / len(known)) if known else t_file_elapsed / max(1.0, si + max(sf, 1e-3))
                avg = [v if v == v else fill for v in avg]

            # Остаток по файлу
            rem_file = max(0.0, (1.0 - sf) * avg[si]) + sum(avg[si + 1:])
            t_file_left = rem_file

            # ETA по набору (только в пакетном режиме)
            tail = ""
            if self._is_batch:
                total = max(1, int(total_files))
                done_files = max(0, int(processed_done))
                avg_per_file = sum(avg)
                rem_batch = rem_file + max(0, total - (done_files + 1)) * avg_per_file
                tail = f", набор ~ {self._fmt_eta(rem_batch)}"

            return f"{base} | Осталось: файл ~ {self._fmt_eta(t_file_left)}{tail}"
        except Exception:
            return base

    # =========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # =========================================================================

    def _genre_of(self, wav_path: str) -> Optional[str]:
        """Определить жанр как первую подпапку относительно dataset_root."""
        if not self.dataset_root:
            return None
        try:
            rel = os.path.relpath(os.path.dirname(wav_path), self.dataset_root)
            parts = rel.split(os.sep)
            return parts[0] if parts and parts[0] not in ('.', '') else None
        except Exception:
            return None

    def _parse_settings(self) -> Tuple:
        """Разобрать настройки с значениями по умолчанию."""
        s = self.settings

        bs = int(s.get('block_size', 2048) or 2048)
        sel_mode = str(s.get('select_mode', 'none') or 'none')
        keep_energy = float(s.get('keep_energy_ratio', 1.0) or 1.0)
        seq_keep = float(s.get('sequency_keep_ratio', 1.0) or 1.0)
        bitrate = str(s.get('bitrate', '192k') or '192k')
        levels = int(s.get('levels', 4) or 4)
        mu = float(s.get('mu', 255.0) or 255.0)
        bits = int(s.get('bits', 8) or 8)
        alpha = float(s.get('rosen_alpha', 0.2) or 0.2)
        beta = float(s.get('rosen_beta', 1.0) or 1.0)

        return bs, sel_mode, keep_energy, seq_keep, bitrate, levels, mu, bits, alpha, beta

    # =========================================================================
    # ОСНОВНОЙ ЦИКЛ
    # =========================================================================

    @Slot()
    def run(self) -> None:
        """Выполнить обработку всех WAV-файлов.

        Для каждого файла:
        1. Запустить 7 методов обработки
        2. Собрать метрики
        3. Отправить результат через сигнал
        """
        self._log.info("worker_start", extra={"files": len(self.wav_paths)})

        processed = 0
        total = self._total
        self._batch_t0 = time.perf_counter()

        # Гарантируем существование выходной директории
        os.makedirs(self.out_dir, exist_ok=True)

        for wav_path in self.wav_paths:
            # Проверка отмены
            if self._cancelled:
                self._log.info("worker_cancelled", extra={"processed": processed, "total": total})
                self.status.emit("Отменено пользователем")
                break

            try:
                wav_path = os.path.normpath(wav_path)
                self._log.info("file_start", extra={"path": wav_path, "idx": processed + 1, "total": total})

                if not os.path.exists(wav_path):
                    raise FileNotFoundError(wav_path)

                self._cur_file_t0 = time.perf_counter()
                self.status.emit(f"Обработка: {os.path.basename(wav_path)} ({processed + 1}/{total})…")
                self.progress_file.emit(1)
                self.progress_total.emit(max(1, int(100 * processed / total)))

                # Настройки
                bs, sel_mode, keep_energy, seq_keep, bitrate, levels, mu, bits, alpha, beta = self._parse_settings()

                # =====================================================================
                # FWHT
                # =====================================================================
                def cb_fwht(frac: float, msg: str):
                    p = 5 + int(max(0.0, min(1.0, frac)) * 50)
                    self.progress_file.emit(p)
                    self.status.emit(self._status_with_eta_cycle(msg, 0, frac, processed, total))
                    self._log.debug("progress", extra={"file": wav_path, "stage": 0, "frac": frac})

                t_s0 = time.perf_counter()
                fwht_mp3, t_fwht = fwht_transform_and_mp3(
                    wav_path, self.out_dir,
                    block_size=bs,
                    select_mode=sel_mode,
                    keep_energy_ratio=keep_energy,
                    sequency_keep_ratio=seq_keep,
                    bitrate=bitrate,
                    progress_cb=cb_fwht,
                )
                self._stage_stats[0] = (
                    self._stage_stats.get(0, (0.0, 0))[0] + (time.perf_counter() - t_s0),
                    self._stage_stats.get(0, (0.0, 0))[1] + 1
                )
                self.progress_file.emit(55)
                self._log.info("fwht_done", extra={"file": wav_path, "out": fwht_mp3, "time_s": t_fwht})

                # =====================================================================
                # FFT
                # =====================================================================
                def cb_fft(frac: float, msg: str):
                    p = 55 + int(max(0.0, min(1.0, frac)) * 10)
                    self.progress_file.emit(p)
                    self.status.emit(self._status_with_eta_cycle(msg, 1, frac, processed, total))

                t_s1 = time.perf_counter()
                fft_mp3, t_fft = fft_transform_and_mp3(
                    wav_path, self.out_dir,
                    block_size=bs,
                    select_mode=sel_mode,
                    keep_energy_ratio=keep_energy,
                    sequency_keep_ratio=seq_keep,
                    bitrate=bitrate,
                    progress_cb=cb_fft,
                )
                self._stage_stats[1] = (
                    self._stage_stats.get(1, (0.0, 0))[0] + (time.perf_counter() - t_s1),
                    self._stage_stats.get(1, (0.0, 0))[1] + 1
                )
                self.progress_file.emit(65)
                self._log.info("fft_done", extra={"file": wav_path, "out": fft_mp3, "time_s": t_fft})

                # =====================================================================
                # DCT
                # =====================================================================
                def cb_dct(frac: float, msg: str):
                    p = 65 + int(max(0.0, min(1.0, frac)) * 10)
                    self.progress_file.emit(p)
                    self.status.emit(self._status_with_eta_cycle(msg, 2, frac, processed, total))

                t_s2 = time.perf_counter()
                dct_mp3, t_dct = dct_transform_and_mp3(
                    wav_path, self.out_dir,
                    block_size=bs,
                    select_mode=sel_mode,
                    keep_energy_ratio=keep_energy,
                    sequency_keep_ratio=seq_keep,
                    bitrate=bitrate,
                    progress_cb=cb_dct,
                )
                self._stage_stats[2] = (
                    self._stage_stats.get(2, (0.0, 0))[0] + (time.perf_counter() - t_s2),
                    self._stage_stats.get(2, (0.0, 0))[1] + 1
                )
                self.progress_file.emit(75)
                self._log.info("dct_done", extra={"file": wav_path, "out": dct_mp3, "time_s": t_dct})

                # =====================================================================
                # DWT (Haar)
                # =====================================================================
                def cb_dwt(frac: float, msg: str):
                    p = 75 + int(max(0.0, min(1.0, frac)) * 10)
                    self.progress_file.emit(p)
                    self.status.emit(self._status_with_eta_cycle(msg, 3, frac, processed, total))

                t_s3 = time.perf_counter()
                dwt_mp3, t_dwt = wavelet_transform_and_mp3(
                    wav_path, self.out_dir,
                    block_size=bs,
                    select_mode=sel_mode,
                    keep_energy_ratio=keep_energy,
                    sequency_keep_ratio=seq_keep,
                    levels=levels,
                    bitrate=bitrate,
                    progress_cb=cb_dwt,
                )
                self._stage_stats[3] = (
                    self._stage_stats.get(3, (0.0, 0))[0] + (time.perf_counter() - t_s3),
                    self._stage_stats.get(3, (0.0, 0))[1] + 1
                )
                self.progress_file.emit(85)
                self._log.info("dwt_done", extra={"file": wav_path, "out": dwt_mp3, "time_s": t_dwt})

                # =====================================================================
                # Huffman-like
                # =====================================================================
                def cb_huff(frac: float, msg: str):
                    p = 85 + int(max(0.0, min(1.0, frac)) * 5)
                    self.progress_file.emit(p)
                    self.status.emit(self._status_with_eta_cycle(msg, 4, frac, processed, total))

                t_s4 = time.perf_counter()
                huff_mp3, t_huff = huffman_like_transform_and_mp3(
                    wav_path, self.out_dir,
                    block_size=bs,
                    bitrate=bitrate,
                    mu=mu,
                    bits=bits,
                    progress_cb=cb_huff,
                )
                self._stage_stats[4] = (
                    self._stage_stats.get(4, (0.0, 0))[0] + (time.perf_counter() - t_s4),
                    self._stage_stats.get(4, (0.0, 0))[1] + 1
                )
                self.progress_file.emit(90)
                self._log.info("huffman_done", extra={"file": wav_path, "out": huff_mp3, "time_s": t_huff})

                # =====================================================================
                # Rosenbrock-like
                # =====================================================================
                def cb_rb(frac: float, msg: str):
                    p = 90 + int(max(0.0, min(1.0, frac)) * 5)
                    self.progress_file.emit(p)
                    self.status.emit(self._status_with_eta_cycle(msg, 5, frac, processed, total))

                t_s5 = time.perf_counter()
                rb_mp3, t_rb = rosenbrock_like_transform_and_mp3(
                    wav_path, self.out_dir,
                    alpha=alpha,
                    beta=beta,
                    bitrate=bitrate,
                    progress_cb=cb_rb,
                )
                self._stage_stats[5] = (
                    self._stage_stats.get(5, (0.0, 0))[0] + (time.perf_counter() - t_s5),
                    self._stage_stats.get(5, (0.0, 0))[1] + 1
                )
                self.progress_file.emit(93)
                self._log.info("rosenbrock_done", extra={"file": wav_path, "out": rb_mp3, "time_s": t_rb})

                # =====================================================================
                # Стандартный MP3
                # =====================================================================
                self.status.emit(self._status_with_eta_cycle("Стандартный: кодирование MP3", 6, 0.0, processed, total))
                t_s6 = time.perf_counter()
                std_mp3, t_std = standard_convert_to_mp3(wav_path, self.out_dir, bitrate=bitrate)
                self._stage_stats[6] = (
                    self._stage_stats.get(6, (0.0, 0))[0] + (time.perf_counter() - t_s6),
                    self._stage_stats.get(6, (0.0, 0))[1] + 1
                )
                self.progress_file.emit(97)
                self.status.emit(self._status_with_eta_cycle("Стандартный: готово", 6, 1.0, processed, total))
                self._log.info("std_done", extra={"file": wav_path, "out": std_mp3, "time_s": t_std})

                # =====================================================================
                # Вычисление метрик
                # =====================================================================
                items: List[Tuple[str, str, float]] = [
                    ("Стандартный MP3", std_mp3, t_std),
                    ("FWHT MP3", fwht_mp3, t_fwht),
                    ("FFT MP3", fft_mp3, t_fft),
                    ("DCT MP3", dct_mp3, t_dct),
                    ("DWT MP3", dwt_mp3, t_dwt),
                    ("Хаффман MP3", huff_mp3, t_huff),
                    ("Розенброк MP3", rb_mp3, t_rb),
                ]

                results = _compute_metrics_batch(wav_path, items)

                # Логирование результатов
                for r in results:
                    self._log.info(
                        "result",
                        extra={
                            "file": wav_path,
                            "variant": r.get("variant"),
                            "lsd_db": r.get("lsd_db"),
                            "time_sec": r.get("time_sec"),
                            "size_bytes": r.get("size_bytes"),
                        }
                    )

                # Отправка результата
                payload = {
                    "source": os.path.basename(wav_path),
                    "genre": self._genre_of(wav_path),
                    "results": results,
                }
                self.result.emit(payload)

                processed += 1
                self.progress_file.emit(100)
                self.progress_total.emit(int(100 * processed / total))
                self._log.info("file_done", extra={"path": wav_path})

            except Exception as e:
                self._log.exception("file_error", extra={"path": wav_path})
                self.error.emit(f"{os.path.basename(wav_path)}: {e}")
                continue

        # Завершение
        try:
            self.finished.emit()
            self._log.info("worker_finished")
        except Exception:
            self._log.exception("worker_emit_finished_error")
