"""
Главное окно приложения AudioAnalyzer.

Назначение:
- Единая реализация UI без дублирования кода.
- Таблица результатов, графики сравнения, настройки, логи.
- Поддержка одиночной и пакетной обработки.

Внешние зависимости: PySide6, processing.audio_ops.
"""
from __future__ import annotations

import glob
import logging
import math
import os
import sys
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QLineSeries,
    QValueAxis,
)
from PySide6.QtCore import Qt, QThread, QUrl, Slot
from PySide6.QtGui import QColor, QFont, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .constants import COLUMN_TOOLTIPS, METRIC_KEYS, SCOPE_OPTIONS, TABLE_HEADERS, VARIANTS
from .presets import PRESET_NAMES
from .log_handler import QtLogHandler, UiLogEmitter
from .presets import apply_preset
from .worker import ResultRow, Worker
from .export_xlsx import export_results_to_xlsx, generate_export_filename, is_export_available


logger = logging.getLogger("ui_new.main_window")

# =============================================================================
# ПУТИ ПРОЕКТА
# =============================================================================

# Корневая директория проекта (относительно расположения этого файла)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_TEST_DATA_DIR = PROJECT_ROOT / "default_test_data"

# Для запуска из exe используем директорию exe, иначе - директорию проекта
if getattr(sys, 'frozen', False):
    OUTPUT_DIR = Path(sys.executable).parent / "output"
else:
    OUTPUT_DIR = PROJECT_ROOT / "output"


# =============================================================================
# ДИАЛОГ ВЫБОРА ИСХОДНЫХ ФАЙЛОВ
# =============================================================================

class SourceFilesDialog(QMessageBox):
    """Диалог для выбора исходного файла из списка доступных."""
    
    def __init__(self, parent, files: List[Tuple[str, str]]):
        super().__init__(parent)
        self.setWindowTitle("Выберите исходный файл")
        self.setText("Доступные исходные файлы:")
        self._selected_path: Optional[str] = None
        self._files = files
        
        # Создаём список файлов
        list_widget = QWidget()
        layout = QVBoxLayout()
        
        # Инструкция
        layout.addWidget(QLabel("Выберите файл для спектрального анализа:"))
        
        # Список файлов
        self.files_list = QTableWidget(len(files), 2)
        self.files_list.setHorizontalHeaderLabels(["Файл", "Путь"])
        self.files_list.horizontalHeader().setStretchLastSection(True)
        self.files_list.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.files_list.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        
        for i, (name, path) in enumerate(files):
            self.files_list.setItem(i, 0, QTableWidgetItem(name))
            # Показываем только часть пути для краткости
            display_path = path if len(path) < 60 else "..." + path[-57:]
            self.files_list.setItem(i, 1, QTableWidgetItem(display_path))
            self.files_list.item(i, 0).setData(Qt.UserRole, path)
        
        self.files_list.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self.files_list)
        
        list_widget.setLayout(layout)
        
        # Добавляем виджет в диалог
        self.layout().addWidget(list_widget, 1, 0, 1, self.layout().columnCount())
        
        # Кнопки
        self.addButton("Выбрать", QMessageBox.AcceptRole)
        self.addButton("Выбрать другой файл...", QMessageBox.ActionRole)
        self.addButton("Отмена", QMessageBox.RejectRole)
    
    def _on_double_click(self) -> None:
        """Двойной клик - выбрать файл."""
        self._selected_path = self._get_selected_path_internal()
        if self._selected_path:
            self.accept()
    
    def _get_selected_path_internal(self) -> Optional[str]:
        """Получить путь из выбранной строки."""
        selected = self.files_list.selectedItems()
        if selected:
            row = selected[0].row()
            item = self.files_list.item(row, 0)
            if item:
                return item.data(Qt.UserRole)
        return None
    
    def get_selected_path(self) -> Optional[str]:
        """Возвращает выбранный путь или '__BROWSE__' для выбора из ФС."""
        clicked = self.clickedButton()
        buttons = self.buttons()
        
        # Кнопка "Выбрать другой файл"
        browse_btn = None
        for btn in buttons:
            if "другой" in btn.text().lower():
                browse_btn = btn
                break
        
        if browse_btn and clicked == browse_btn:
            return "__BROWSE__"
        
        # Кнопка "Выбрать"
        select_btn = None
        for btn in buttons:
            if btn.text() == "Выбрать":
                select_btn = btn
                break
        
        if select_btn and clicked == select_btn:
            return self._get_selected_path_internal()
        
        return None


# =============================================================================
# ГЛАВНОЕ ОКНО
# =============================================================================

class MainWindow(QMainWindow):
    """Главное окно приложения AudioAnalyzer.

    Содержит:
    - Вкладку "Таблица" с результатами
    - Вкладку "Сравнение" с графиками и heatmap
    - Вкладку "Настройки" с параметрами методов
    - Панель логов
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Audio Transformer")
        self.resize(1100, 650)

        # Состояние
        self._thread: Optional[QThread] = None
        self._worker: Optional[Worker] = None
        self._results: List[ResultRow] = []
        self._variant_visible: Dict[str, bool] = {v: True for v in VARIANTS}
        
        # Путь к папке с данными для пакетной обработки
        self._dataset_folder: Optional[str] = str(DEFAULT_TEST_DATA_DIR) if DEFAULT_TEST_DATA_DIR.exists() else None
        
        # Плеер
        self._current_player_file: Optional[str] = None

        # Логирование в UI
        self._log_emitter = UiLogEmitter()
        self._qt_log_handler = QtLogHandler(self._log_emitter)
        self._qt_log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self._qt_log_handler)
        logging.getLogger().setLevel(logging.INFO)

        # Построение UI
        self._build_ui()

    # =========================================================================
    # ПОСТРОЕНИЕ UI
    # =========================================================================

    def _build_ui(self) -> None:
        """Построить интерфейс приложения."""
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout()
        central.setLayout(root)

        # -------------------------------------------------------------------------
        # Верхняя панель
        # -------------------------------------------------------------------------
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)

        btn_browse = QPushButton("Выбрать .wav…")
        btn_browse.clicked.connect(self.on_browse)

        self.btn_convert = QPushButton("Запустить выбранный файл")
        self.btn_convert.setEnabled(False)
        self.btn_convert.clicked.connect(self.on_convert)

        # Папка с данными
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setReadOnly(True)
        self.dataset_edit.setPlaceholderText("Папка с WAV-файлами...")
        if self._dataset_folder:
            self.dataset_edit.setText(self._dataset_folder)
        
        btn_dataset_browse = QPushButton("Выбрать папку…")
        btn_dataset_browse.setToolTip("Выбрать папку с WAV-файлами для пакетной обработки")
        btn_dataset_browse.clicked.connect(self.on_browse_dataset)
        
        self.btn_batch = QPushButton("Запустить набор")
        self.btn_batch.setToolTip("Обработать WAV-файлы из выбранной папки рекурсивно")
        self.btn_batch.clicked.connect(self.on_run_dataset)

        self.show_logs_cb = QCheckBox("Показывать логи")
        self.show_logs_cb.setChecked(True)
        self.show_logs_cb.toggled.connect(self._on_toggle_logs)

        self.status_label = QLabel("")

        # Первая строка: выбор файла
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Исходный WAV:"))
        row1.addWidget(self.path_edit, 1)
        row1.addWidget(btn_browse)
        row1.addWidget(self.btn_convert)
        row1.addWidget(self.show_logs_cb)
        root.addLayout(row1)
        
        # Вторая строка: выбор папки с данными
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Папка с данными:"))
        row2.addWidget(self.dataset_edit, 1)
        row2.addWidget(btn_dataset_browse)
        row2.addWidget(self.btn_batch)
        root.addLayout(row2)

        # -------------------------------------------------------------------------
        # Центральная область (вкладки + логи)
        # -------------------------------------------------------------------------
        center = QHBoxLayout()
        root.addLayout(center, 1)

        # Вкладки
        self.tabs = QTabWidget()
        center.addWidget(self.tabs, 3)

        # Панель логов
        self.logs_tabs = QTabWidget()
        center.addWidget(self.logs_tabs, 2)

        self.logs_edit = QPlainTextEdit()
        self.logs_edit.setReadOnly(True)
        self.logs_tabs.addTab(self.logs_edit, "Логи")
        self._log_emitter.log_line.connect(lambda s: self.logs_edit.appendPlainText(s))

        # -------------------------------------------------------------------------
        # Вкладка: Таблица
        # -------------------------------------------------------------------------
        page_table = QWidget()
        lay_table = QVBoxLayout()
        page_table.setLayout(lay_table)

        self.table = QTableWidget(0, len(TABLE_HEADERS))
        self.table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)

        # Tooltips для столбцов
        for col, tooltip in COLUMN_TOOLTIPS.items():
            item = self.table.horizontalHeaderItem(col)
            if item:
                item.setToolTip(tooltip)

        # Масштабирование колонок
        try:
            hh = self.table.horizontalHeader()
            hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            hh.setStretchLastSection(True)
        except Exception:
            pass

        lay_table.addWidget(self.table, 1)

        # Панель кнопок под таблицей
        btn_row = QHBoxLayout()
        
        self.btn_export_xlsx = QPushButton("📄 Экспорт в Excel")
        self.btn_export_xlsx.setToolTip("Сохранить таблицу результатов в файл .xlsx")
        self.btn_export_xlsx.clicked.connect(self.on_export_xlsx)
        self.btn_export_xlsx.setEnabled(False)
        btn_row.addWidget(self.btn_export_xlsx)
        
        self.btn_clear_output = QPushButton("🗑️ Очистить output")
        self.btn_clear_output.setToolTip("Удалить все обработанные файлы из папки output")
        self.btn_clear_output.clicked.connect(self.on_clear_output)
        btn_row.addWidget(self.btn_clear_output)
        
        btn_row.addStretch(1)
        lay_table.addLayout(btn_row)

        # Прогресс
        self.progress_total = QProgressBar()
        self.progress_total.setRange(0, 100)
        self.progress_total.setFormat("Набор: %p%")
        self.progress_total.setVisible(False)
        lay_table.addWidget(self.progress_total)

        self.progress_file = QProgressBar()
        self.progress_file.setRange(0, 100)
        self.progress_file.setFormat("Файл: %p%")
        self.progress_file.setVisible(False)
        lay_table.addWidget(self.progress_file)

        lay_table.addWidget(self.status_label)

        self.tabs.addTab(page_table, "Таблица")

        # -------------------------------------------------------------------------
        # Вкладка: Сравнение
        # -------------------------------------------------------------------------
        page_compare = QWidget()
        lay_compare = QVBoxLayout()
        page_compare.setLayout(lay_compare)

        # Панель управления
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Сравнить:"))

        self.combo_scope = QComboBox()
        for text, data in SCOPE_OPTIONS:
            self.combo_scope.addItem(text, data)
        self.combo_scope.currentIndexChanged.connect(self._refresh_chart)
        self.combo_scope.currentIndexChanged.connect(self._refresh_heatmap)
        ctrl.addWidget(self.combo_scope)

        ctrl.addWidget(QLabel("Метрика:"))
        self.combo_metric = QComboBox()
        for key, (title, _, _) in METRIC_KEYS.items():
            self.combo_metric.addItem(title, key)
        self.combo_metric.currentIndexChanged.connect(self._refresh_chart)
        self.combo_metric.currentIndexChanged.connect(self._refresh_heatmap)
        ctrl.addWidget(self.combo_metric)

        self.cb_heatmap = QCheckBox("Heatmap")
        self.cb_heatmap.setChecked(True)
        self.cb_heatmap.toggled.connect(self._toggle_heatmap)
        ctrl.addWidget(self.cb_heatmap)

        self.cb_hints = QCheckBox("Подсказки")
        self.cb_hints.setChecked(True)
        self.cb_hints.toggled.connect(self._toggle_hints)
        ctrl.addWidget(self.cb_hints)

        ctrl.addStretch(1)
        lay_compare.addLayout(ctrl)

        # Панель методов
        left_w = QWidget()
        left_l = QVBoxLayout()
        left_w.setLayout(left_l)
        left_l.addWidget(QLabel("Методы:"))

        self._variant_cbs: Dict[str, QCheckBox] = {}
        for v in VARIANTS:
            cb = QCheckBox(v)
            cb.setChecked(True)
            cb.toggled.connect(self._on_variant_visibility)
            self._variant_cbs[v] = cb
            left_l.addWidget(cb)
        left_l.addStretch(1)

        # График
        self.chart = QChart()
        self.chart.setTitle("Сравнение методов")
        self.chart.legend().setVisible(True)
        self.chart_view = QChartView(self.chart)

        row_chart = QHBoxLayout()
        row_chart.addWidget(left_w)
        row_chart.addWidget(self.chart_view, 1)
        lay_compare.addLayout(row_chart, 1)

        # Heatmap
        self.table_heatmap = QTableWidget(0, 0)
        lay_compare.addWidget(self.table_heatmap)

        # Подсказки
        self.hints_table = QTableWidget(0, 3)
        self.hints_table.setHorizontalHeaderLabels(["Метрика", "Краткое описание", "Что отражает"])
        self.hints_table.horizontalHeader().setStretchLastSection(True)
        lay_compare.addWidget(self.hints_table)

        self.tabs.addTab(page_compare, "Сравнение")

        # -------------------------------------------------------------------------
        # Вкладка: Настройки
        # -------------------------------------------------------------------------
        page_settings = QWidget()
        lay_settings = QHBoxLayout()
        page_settings.setLayout(lay_settings)

        # Форма настроек
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)

        # Поля
        self.ed_block = QLineEdit("2048")
        self.ed_block.setMaximumWidth(160)
        self.ed_block.setToolTip("Размер блока для OLA (Overlap-Add). Должен быть степенью двойки (512, 1024, 2048, 4096). Больше блок = лучше частотное разрешение, но медленнее.")
        
        self.ed_bitrate = QLineEdit("192k")
        self.ed_bitrate.setMaximumWidth(160)
        self.ed_bitrate.setToolTip("Битрейт выходного MP3 файла. Чем выше, тем лучше качество, но больше размер. Типичные значения: 128k, 192k, 256k, 320k.")

        self.cb_select = QComboBox()
        self.cb_select.addItem("Без отбора", "none")
        self.cb_select.addItem("По энергии", "energy")
        self.cb_select.addItem("Низкочастотный", "lowpass")
        self.cb_select.setToolTip("Режим отбора коэффициентов:\n• Без отбора — полная реконструкция\n• По энергии — сохранение доли энергии (сжатие)\n• Низкочастотный — обрезание высоких частот (фильтрация)")

        self.ed_keep_energy = QLineEdit("1.0")
        self.ed_keep_energy.setMaximumWidth(160)
        self.ed_keep_energy.setToolTip("Доля сохраняемой энергии (0.0-1.0) для режима 'По энергии'. Например, 0.95 = 95% энергии.")

        self.ed_seq_keep = QLineEdit("1.0")
        self.ed_seq_keep.setMaximumWidth(160)
        self.ed_seq_keep.setToolTip("Доля сохраняемых низких частот (0.0-1.0) для режима 'Низкочастотный'. Например, 0.5 = первые 50% частот.")

        self.ed_levels = QLineEdit("4")
        self.ed_levels.setMaximumWidth(160)
        self.ed_levels.setToolTip("Число уровней вейвлет-декомпозиции для DWT (Haar). Типичные значения: 3-6.")

        self.ed_mu = QLineEdit("255")
        self.ed_mu.setMaximumWidth(160)
        self.ed_mu.setToolTip("Параметр μ для μ-law компандирования (Хаффман-подобный метод). Типично 255.")

        self.ed_bits = QLineEdit("8")
        self.ed_bits.setMaximumWidth(160)
        self.ed_bits.setToolTip("Число бит квантования (Хаффман-подобный метод). Больше бит = выше качество. Типично 8-12.")

        self.ed_ra = QLineEdit("0.2")
        self.ed_ra.setMaximumWidth(160)
        self.ed_ra.setToolTip("Параметр α для Розенброк-преобразования. Контролирует сглаживание.")

        self.ed_rb = QLineEdit("1.0")
        self.ed_rb.setMaximumWidth(160)
        self.ed_rb.setToolTip("Параметр β для Розенброк-преобразования. Контролирует сдвиг.")

        # Пресеты
        self.cb_preset = QComboBox()
        self.cb_preset.addItems(PRESET_NAMES)
        self.cb_preset.currentIndexChanged.connect(lambda _: apply_preset(self, self.cb_preset.currentText()))
        form.addRow(QLabel("Пресет:"), self.cb_preset)

        # Добавляем поля
        form.addRow(QLabel("Размер блока (2^n):"), self.ed_block)
        form.addRow(QLabel("Битрейт MP3:"), self.ed_bitrate)
        form.addRow(QLabel("Режим отбора:"), self.cb_select)
        form.addRow(QLabel("Доля энергии (0..1):"), self.ed_keep_energy)
        form.addRow(QLabel("Доля частот (0..1):"), self.ed_seq_keep)
        form.addRow(QLabel("DWT уровни:"), self.ed_levels)
        form.addRow(QLabel("μ (Хаффман-подобн.):"), self.ed_mu)
        form.addRow(QLabel("Биты (Хаффман-подобн.):"), self.ed_bits)
        form.addRow(QLabel("Rosenbrock α:"), self.ed_ra)
        form.addRow(QLabel("Rosenbrock β:"), self.ed_rb)

        left_panel = QWidget()
        left_panel.setLayout(form)
        left_panel.setMinimumWidth(380)
        lay_settings.addWidget(left_panel)

        # Матрица настроек
        self._build_settings_matrix(lay_settings)

        self.tabs.addTab(page_settings, "Настройки")

        # -------------------------------------------------------------------------
        # Вкладка: Плеер
        # -------------------------------------------------------------------------
        page_player = QWidget()
        lay_player = QVBoxLayout()
        page_player.setLayout(lay_player)

        # Заголовок
        player_header = QLabel("🎵 Плеер аудиофайлов")
        player_header.setStyleSheet("font-weight: bold; font-size: 14px; margin: 5px;")
        lay_player.addWidget(player_header)

        # Панель выбора файла
        file_panel = QHBoxLayout()
        
        file_panel.addWidget(QLabel("Файл:"))
        self.player_file_edit = QLineEdit()
        self.player_file_edit.setReadOnly(True)
        self.player_file_edit.setPlaceholderText("Выберите файл для воспроизведения...")
        file_panel.addWidget(self.player_file_edit, 1)
        
        self.btn_browse_player = QPushButton("Обзор...")
        self.btn_browse_player.setToolTip("Выбрать аудиофайл (WAV, MP3)")
        self.btn_browse_player.clicked.connect(self.on_browse_player_file)
        file_panel.addWidget(self.btn_browse_player)
        
        lay_player.addLayout(file_panel)

        # Информация о файле
        self.player_info_label = QLabel("Файл не выбран")
        self.player_info_label.setStyleSheet("color: gray; margin: 5px;")
        lay_player.addWidget(self.player_info_label)

        # Плеер контролы
        controls_panel = QHBoxLayout()
        
        self.btn_play = QPushButton("▶️ Воспроизвести")
        self.btn_play.setToolTip("Начать воспроизведение")
        self.btn_play.clicked.connect(self.on_player_play)
        self.btn_play.setEnabled(False)
        controls_panel.addWidget(self.btn_play)
        
        self.btn_pause = QPushButton("⏸️ Пауза")
        self.btn_pause.setToolTip("Приостановить воспроизведение")
        self.btn_pause.clicked.connect(self.on_player_pause)
        self.btn_pause.setEnabled(False)
        controls_panel.addWidget(self.btn_pause)
        
        self.btn_stop = QPushButton("⏹️ Стоп")
        self.btn_stop.setToolTip("Остановить воспроизведение")
        self.btn_stop.clicked.connect(self.on_player_stop)
        self.btn_stop.setEnabled(False)
        controls_panel.addWidget(self.btn_stop)
        
        # Громкость
        controls_panel.addWidget(QLabel("🔊"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setMaximumWidth(150)
        self.volume_slider.setToolTip("Громкость (0-100%)")
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        controls_panel.addWidget(self.volume_slider)
        
        self.volume_label = QLabel("80%")
        self.volume_label.setMinimumWidth(40)
        controls_panel.addWidget(self.volume_label)
        
        controls_panel.addStretch(1)
        lay_player.addLayout(controls_panel)

        # Прогресс воспроизведения
        progress_panel = QHBoxLayout()
        self.position_label = QLabel("00:00")
        self.position_label.setMinimumWidth(50)
        progress_panel.addWidget(self.position_label)
        
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.setValue(0)
        self.position_slider.setToolTip("Позиция воспроизведения")
        self.position_slider.sliderMoved.connect(self.on_position_slider_moved)
        progress_panel.addWidget(self.position_slider, 1)
        
        self.duration_label = QLabel("00:00")
        self.duration_label.setMinimumWidth(50)
        progress_panel.addWidget(self.duration_label)
        
        lay_player.addLayout(progress_panel)

        # Инициализация медиаплеера
        self._media_player = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._media_player.setAudioOutput(self._audio_output)
        self._audio_output.setVolume(0.8)
        
        # Подключение сигналов плеера
        self._media_player.positionChanged.connect(self.on_player_position_changed)
        self._media_player.durationChanged.connect(self.on_player_duration_changed)
        self._media_player.playbackStateChanged.connect(self.on_player_state_changed)
        self._media_player.errorChanged.connect(self.on_player_error)

        # Разделение на две колонки: исходные и обработанные файлы
        files_splitter = QHBoxLayout()
        
        # Левая колонка: исходные файлы
        source_column = QVBoxLayout()
        source_header = QLabel("📁 Исходные файлы:")
        source_header.setStyleSheet("font-weight: bold;")
        source_column.addWidget(source_header)
        
        self.source_files_list = QTableWidget(0, 2)
        self.source_files_list.setHorizontalHeaderLabels(["Файл", "Размер"])
        self.source_files_list.horizontalHeader().setStretchLastSection(True)
        self.source_files_list.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.source_files_list.doubleClicked.connect(self.on_source_file_double_clicked)
        self.source_files_list.itemSelectionChanged.connect(self.on_source_file_selected)
        source_column.addWidget(self.source_files_list, 1)
        
        # Кнопки для исходных файлов
        source_buttons = QHBoxLayout()
        btn_refresh_source = QPushButton("🔄 Обновить")
        btn_refresh_source.clicked.connect(self.refresh_source_files_list)
        source_buttons.addWidget(btn_refresh_source)
        
        btn_add_source = QPushButton("➕ Добавить файл")
        btn_add_source.clicked.connect(self.on_add_source_file)
        source_buttons.addWidget(btn_add_source)
        source_column.addLayout(source_buttons)
        
        source_widget = QWidget()
        source_widget.setLayout(source_column)
        files_splitter.addWidget(source_widget, 1)
        
        # Правая колонка: обработанные файлы
        output_column = QVBoxLayout()
        output_header = QLabel("📂 Обработанные файлы:")
        output_header.setStyleSheet("font-weight: bold;")
        output_column.addWidget(output_header)
        
        self.output_files_list = QTableWidget(0, 3)
        self.output_files_list.setHorizontalHeaderLabels(["Метод", "Размер", "Дата"])
        self.output_files_list.horizontalHeader().setStretchLastSection(True)
        self.output_files_list.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.output_files_list.doubleClicked.connect(self.on_output_file_double_clicked)
        output_column.addWidget(self.output_files_list, 1)
        
        # Кнопки для обработанных файлов
        output_buttons = QHBoxLayout()
        btn_refresh_output = QPushButton("🔄 Обновить")
        btn_refresh_output.clicked.connect(self.refresh_output_files_list)
        output_buttons.addWidget(btn_refresh_output)
        
        btn_open_folder = QPushButton("📂 Открыть папку")
        btn_open_folder.clicked.connect(self.on_open_output_folder)
        output_buttons.addWidget(btn_open_folder)
        output_column.addLayout(output_buttons)
        
        output_widget = QWidget()
        output_widget.setLayout(output_column)
        files_splitter.addWidget(output_widget, 1)
        
        lay_player.addLayout(files_splitter, 1)

        self.tabs.addTab(page_player, "Плеер")

        # -------------------------------------------------------------------------
        # Вкладка: Спектр
        # -------------------------------------------------------------------------
        page_spectrum = QWidget()
        lay_spectrum = QVBoxLayout()
        page_spectrum.setLayout(lay_spectrum)

        # Заголовок
        spectrum_header = QLabel("📊 Спектральный анализ")
        spectrum_header.setStyleSheet("font-weight: bold; font-size: 14px; margin: 5px;")
        lay_spectrum.addWidget(spectrum_header)

        # Панель выбора файлов
        spectrum_files_panel = QHBoxLayout()
        
        # Исходный файл
        spectrum_files_panel.addWidget(QLabel("Исходный:"))
        self.spectrum_source_edit = QLineEdit()
        self.spectrum_source_edit.setReadOnly(True)
        self.spectrum_source_edit.setPlaceholderText("Выберите исходный файл...")
        spectrum_files_panel.addWidget(self.spectrum_source_edit, 1)
        
        btn_browse_spectrum_source = QPushButton("Обзор...")
        btn_browse_spectrum_source.clicked.connect(self.on_browse_spectrum_source)
        spectrum_files_panel.addWidget(btn_browse_spectrum_source)
        
        lay_spectrum.addLayout(spectrum_files_panel)

        # Таблица обработки для сравнения
        spectrum_compare_label = QLabel("Выберите методы для сравнения:")
        lay_spectrum.addWidget(spectrum_compare_label)
        
        self.spectrum_files_table = QTableWidget(0, 4)
        self.spectrum_files_table.setHorizontalHeaderLabels(
            ["✓", "Метод", "Файл", "Размер"]
        )
        self.spectrum_files_table.horizontalHeader().setStretchLastSection(True)
        self.spectrum_files_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        lay_spectrum.addWidget(self.spectrum_files_table, 1)

        # Кнопки управления
        spectrum_buttons = QHBoxLayout()
        
        btn_refresh_spectrum = QPushButton("🔄 Обновить список")
        btn_refresh_spectrum.clicked.connect(self.refresh_spectrum_files_list)
        spectrum_buttons.addWidget(btn_refresh_spectrum)
        
        btn_compare_spectrum = QPushButton("📊 Сравнить спектры")
        btn_compare_spectrum.clicked.connect(self.on_compare_spectrum)
        spectrum_buttons.addWidget(btn_compare_spectrum)
        
        btn_compare_all = QPushButton("✅ Выбрать все")
        btn_compare_all.clicked.connect(self.on_select_all_spectrum)
        spectrum_buttons.addWidget(btn_compare_all)
        
        spectrum_buttons.addStretch(1)
        lay_spectrum.addLayout(spectrum_buttons)

        # График спектра
        self.spectrum_chart = QChart()
        self.spectrum_chart.setTitle("Спектральное сравнение")
        self.spectrum_chart.legend().setVisible(True)
        self.spectrum_chart_view = QChartView(self.spectrum_chart)
        lay_spectrum.addWidget(self.spectrum_chart_view, 2)

        self.tabs.addTab(page_spectrum, "Спектр")

        # -------------------------------------------------------------------------
        # Инициализация
        # -------------------------------------------------------------------------
        apply_preset(self, "Стандартный")
        self._update_settings_matrix_table()
        self._fill_metric_hints()
        
        # Инициализация списка файлов output
        self.refresh_output_files_list()

        # Реакция на изменения
        for ed in (
            self.ed_block, self.ed_bitrate, self.ed_keep_energy,
            self.ed_seq_keep, self.ed_levels, self.ed_mu,
            self.ed_bits, self.ed_ra, self.ed_rb
        ):
            try:
                ed.editingFinished.connect(self._update_settings_matrix_table)
            except Exception:
                pass

        self.cb_select.currentIndexChanged.connect(lambda _: self._update_settings_matrix_table())
        self.cb_preset.currentIndexChanged.connect(lambda _: self._update_settings_matrix_table())

        # Изначальная видимость
        self._toggle_heatmap(self.cb_heatmap.isChecked())
        self._toggle_hints(self.cb_hints.isChecked())

    def _build_settings_matrix(self, layout: QHBoxLayout) -> None:
        """Построить матрицу влияния: методы × метрики.
        
        Матрица показывает качественное влияние параметров на метрики:
        ↑↑ - существенное улучшение качества
        ↑  - улучшение
        →  - без изменений (стандартные параметры)
        ↓  - ухудшение
        ↓↓ - существенное ухудшение
        
        Цветовая кодировка: зелёный = улучшение, серый = без изменений, красный = ухудшение.
        """
        self._metrics_cols = self._get_metrics_cols()
        methods = self._method_headers()

        # Создаём вертикальный контейнер для матрицы и легенды
        matrix_container = QVBoxLayout()
        matrix_container.setSpacing(5)
        
        # Заголовок матрицы
        matrix_title = QLabel("📊 Матрица влияния параметров")
        matrix_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        matrix_container.addWidget(matrix_title)

        # Таблица: строки = методы, столбцы = метрики
        self.table_settings_matrix = QTableWidget(len(methods) + 1, len(self._metrics_cols) + 1)
        
        # Подсказка для всей таблицы
        self.table_settings_matrix.setToolTip(
            "Матрица влияния параметров на метрики качества.\n\n"
            "Символы:\n"
            "↑↑ — существенное улучшение (зелёный)\n"
            "↑ — улучшение (светло-зелёный)\n"
            "→ — без изменений (серый)\n"
            "↓ — ухудшение (оранжевый)\n"
            "↓↓ — существенное ухудшение (красный)\n\n"
            "Наведите на ячейку для просмотра факторов влияния."
        )

        # Заголовки метрик (столбцы)
        for ci, (_, mlabel, _) in enumerate(self._metrics_cols, start=1):
            it = QTableWidgetItem(mlabel)
            it.setTextAlignment(Qt.AlignCenter)
            self.table_settings_matrix.setItem(0, ci, it)

        # Заголовки методов (строки)
        for ri, (_, mlabel) in enumerate(methods, start=1):
            it = QTableWidgetItem(mlabel)
            it.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.table_settings_matrix.setItem(ri, 0, it)

        try:
            self.table_settings_matrix.verticalHeader().setVisible(False)
            self.table_settings_matrix.setCornerButtonEnabled(False)
            # Ширина столбцов
            self.table_settings_matrix.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.ResizeToContents
            )
            for i in range(1, len(self._metrics_cols) + 1):
                self.table_settings_matrix.horizontalHeader().setSectionResizeMode(
                    i, QHeaderView.ResizeMode.ResizeToContents
                )
        except Exception:
            pass

        matrix_container.addWidget(self.table_settings_matrix, 1)

        # Чекбокс для показа легенды (по аналогии с подсказками на вкладке Сравнение)
        self.cb_matrix_legend = QCheckBox("Показать легенду матрицы")
        self.cb_matrix_legend.setChecked(False)
        self.cb_matrix_legend.toggled.connect(self._toggle_matrix_legend)
        matrix_container.addWidget(self.cb_matrix_legend)

        # Таблица легенды (изначально скрыта)
        self.matrix_legend_table = QTableWidget(5, 3)
        self.matrix_legend_table.setHorizontalHeaderLabels(["Символ", "Значение", "Цвет"])
        self.matrix_legend_table.setVisible(False)
        self.matrix_legend_table.setMaximumHeight(150)
        
        legend_data = [
            ("↑↑", "Существенное улучшение", "🟢 Зелёный"),
            ("↑", "Улучшение", "🟢 Светло-зелёный"),
            ("→", "Без изменений", "⚫ Серый"),
            ("↓", "Ухудшение", "🟠 Оранжевый"),
            ("↓↓", "Существенное ухудшение", "🔴 Красный"),
        ]
        
        for row, (symbol, meaning, color) in enumerate(legend_data):
            self.matrix_legend_table.setItem(row, 0, QTableWidgetItem(symbol))
            self.matrix_legend_table.setItem(row, 1, QTableWidgetItem(meaning))
            self.matrix_legend_table.setItem(row, 2, QTableWidgetItem(color))
        
        self.matrix_legend_table.horizontalHeader().setStretchLastSection(True)
        self.matrix_legend_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        matrix_container.addWidget(self.matrix_legend_table)

        # Добавляем контейнер в layout
        matrix_widget = QWidget()
        matrix_widget.setLayout(matrix_container)
        layout.addWidget(matrix_widget, 1)

    def _toggle_matrix_legend(self, visible: bool) -> None:
        """Показать/скрыть легенду матрицы."""
        self.matrix_legend_table.setVisible(visible)

    def _method_headers(self) -> List[Tuple[str, str]]:
        """Заголовки методов для матрицы."""
        return [
            ("std", "Стандартный"),
            ("fwht", "FWHT"),
            ("fft", "FFT"),
            ("dct", "DCT"),
            ("dwt", "DWT"),
            ("huff", "Хаффман"),
            ("rb", "Розенброк"),
        ]

    def _get_metrics_cols(self) -> List[Tuple[str, str, str]]:
        """Метрики для матрицы: (key, label, direction)."""
        return [
            ("lsd", "LSD↓", "Ниже лучше"),
            ("snr", "SNR↑", "Выше лучше"),
            ("rmse", "RMSE↓", "Ниже лучше"),
            ("sisdr", "SI-SDR↑", "Выше лучше"),
            ("spec_conv", "Спектр↓", "Ниже лучше"),
            ("centroid", "Центр↓", "Ниже лучше"),
            ("cosine", "Косин↑", "Выше лучше"),
            ("time", "Время↓", "Ниже лучше"),
            ("size", "Размер↓", "Ниже лучше"),
        ]

    def _get_params_rows(self) -> List[Tuple[str, str, str]]:
        """Параметры для матрицы влияния: (key, label, description)."""
        return [
            ("block_size", "Размер блока", "Размер блока для OLA (2^n)"),
            ("bitrate", "Битрейт MP3", "Битрейт кодирования"),
            ("select_mode", "Режим отбора", "Метод отбора коэффициентов"),
            ("keep_energy", "Доля энергии", "Для режима 'energy'"),
            ("seq_keep", "Доля частот", "Для режима 'lowpass'"),
            ("levels", "DWT уровни", "Число уровней вейвлета"),
            ("mu", "μ (Хаффман)", "Параметр μ-law компандирования"),
            ("bits", "Биты (Хаффман)", "Биты квантования"),
            ("alpha", "α (Розенброк)", "Параметр сглаживания"),
            ("beta", "β (Розенброк)", "Параметр сдвига"),
        ]

    def _calculate_impact_score(self, method: str, metric: str, settings: Dict) -> Tuple[str, str, QColor]:
        """Рассчитать влияние параметров на метрику для метода.

        Возвращает: (символ, текстовое описание, цвет)
        """
        block_size = settings.get("block_size", 2048)
        bitrate = str(settings.get("bitrate", "192k"))
        select_mode = settings.get("select_mode", "none")
        keep_energy = float(settings.get("keep_energy_ratio", 1.0))
        seq_keep = float(settings.get("sequency_keep_ratio", 1.0))
        levels = int(settings.get("levels", 4))
        mu = float(settings.get("mu", 255))
        bits = int(settings.get("bits", 8))
        alpha = float(settings.get("rosen_alpha", 0.2))
        beta = float(settings.get("rosen_beta", 1.0))

        # Базовые значения
        base_block = 2048
        base_bitrate_kbps = 192

        # Парсинг битрейта
        try:
            bitrate_kbps = int(bitrate.replace('k', '').replace('K', ''))
        except:
            bitrate_kbps = 192

        # Инициализация влияния
        impact = 0.0
        factors = []

        # ========================================
        # РАСЧЁТ ВЛИЯНИЯ ДЛЯ КАЖДОЙ МЕТРИКИ
        # ========================================

        # --- LSD (Log-Spectral Distance) - ниже лучше ---
        if metric == "lsd":
            # Битрейт: выше битрейт → ниже LSD (лучше)
            if method != "std":
                bitrate_impact = (bitrate_kbps - base_bitrate_kbps) / base_bitrate_kbps * -1.5
                impact += bitrate_impact
                if abs(bitrate_impact) > 0.1:
                    factors.append(f"битрейт:{bitrate_kbps}k")

            # Размер блока: больше блок → лучше спектральное разрешение → ниже LSD
            if method in ("fwht", "fft", "dct", "dwt"):
                block_impact = (block_size - base_block) / base_block * -0.5
                impact += block_impact
                if abs(block_impact) > 0.1:
                    factors.append(f"блок:{block_size}")

                # Режим отбора
                if select_mode == "energy" and keep_energy < 1.0:
                    energy_impact = (1.0 - keep_energy) * 2.0  # потери энергии → выше LSD
                    impact += energy_impact
                    factors.append(f"энергия:{keep_energy:.0%}")
                elif select_mode == "lowpass" and seq_keep < 1.0:
                    lowpass_impact = (1.0 - seq_keep) * 2.5  # удаление частот → выше LSD
                    impact += lowpass_impact
                    factors.append(f"частоты:{seq_keep:.0%}")

            # DWT уровни
            if method == "dwt" and levels != 4:
                level_impact = (levels - 4) * -0.1
                impact += level_impact
                if abs(level_impact) > 0.1:
                    factors.append(f"уровни:{levels}")

            # Хаффман μ и биты
            if method == "huff":
                if mu != 255:
                    mu_impact = abs(mu - 255) / 255 * 0.3
                    impact += mu_impact
                    factors.append(f"μ:{mu:.0f}")
                if bits != 8:
                    bits_impact = (8 - bits) * 0.2
                    impact += bits_impact
                    factors.append(f"биты:{bits}")

            # Розенброк
            if method == "rb":
                if alpha != 0.2:
                    alpha_impact = abs(alpha - 0.2) * 0.5
                    impact += alpha_impact
                    factors.append(f"α:{alpha:.1f}")
                if beta != 1.0:
                    beta_impact = abs(beta - 1.0) * 0.3
                    impact += beta_impact
                    factors.append(f"β:{beta:.1f}")

        # --- SNR (Signal-to-Noise Ratio) - выше лучше ---
        elif metric == "snr":
            # Битрейт: выше битрейт → выше SNR
            bitrate_impact = (bitrate_kbps - base_bitrate_kbps) / base_bitrate_kbps * 1.5
            impact += bitrate_impact
            if abs(bitrate_impact) > 0.1:
                factors.append(f"битрейт:{bitrate_kbps}k")

            # Размер блока: больше → выше SNR
            if method in ("fwht", "fft", "dct", "dwt"):
                block_impact = (block_size - base_block) / base_block * 0.3
                impact += block_impact
                if abs(block_impact) > 0.1:
                    factors.append(f"блок:{block_size}")

                # Режим отбора (инвертированное влияние - потери уменьшают SNR)
                if select_mode == "energy" and keep_energy < 1.0:
                    energy_impact = (keep_energy - 1.0) * 2.0
                    impact += energy_impact
                    factors.append(f"энергия:{keep_energy:.0%}")
                elif select_mode == "lowpass" and seq_keep < 1.0:
                    lowpass_impact = (seq_keep - 1.0) * 2.5
                    impact += lowpass_impact
                    factors.append(f"частоты:{seq_keep:.0%}")

            # Хаффман
            if method == "huff":
                if bits != 8:
                    bits_impact = (bits - 8) * 0.3
                    impact += bits_impact
                    factors.append(f"биты:{bits}")
                if mu != 255:
                    mu_impact = -abs(mu - 255) / 255 * 0.2
                    impact += mu_impact
                    factors.append(f"μ:{mu:.0f}")

        # --- RMSE - ниже лучше ---
        elif metric == "rmse":
            bitrate_impact = (bitrate_kbps - base_bitrate_kbps) / base_bitrate_kbps * -1.2
            impact += bitrate_impact

            if method in ("fwht", "fft", "dct", "dwt"):
                block_impact = (block_size - base_block) / base_block * -0.3
                impact += block_impact

                if select_mode == "energy" and keep_energy < 1.0:
                    impact += (1.0 - keep_energy) * 1.8
                    factors.append(f"энергия:{keep_energy:.0%}")
                elif select_mode == "lowpass" and seq_keep < 1.0:
                    impact += (1.0 - seq_keep) * 2.0
                    factors.append(f"частоты:{seq_keep:.0%}")

            if method == "huff" and bits != 8:
                impact += (8 - bits) * 0.15
                factors.append(f"биты:{bits}")

        # --- SI-SDR - выше лучше ---
        elif metric == "sisdr":
            bitrate_impact = (bitrate_kbps - base_bitrate_kbps) / base_bitrate_kbps * 1.3
            impact += bitrate_impact

            if method in ("fwht", "fft", "dct", "dwt"):
                if select_mode == "energy" and keep_energy < 1.0:
                    impact += (keep_energy - 1.0) * 1.5
                    factors.append(f"энергия:{keep_energy:.0%}")
                elif select_mode == "lowpass" and seq_keep < 1.0:
                    impact += (seq_keep - 1.0) * 1.8
                    factors.append(f"частоты:{seq_keep:.0%}")

        # --- Spectral Convergence - ниже лучше ---
        elif metric == "spec_conv":
            if method in ("fwht", "fft", "dct", "dwt"):
                if select_mode == "energy" and keep_energy < 1.0:
                    impact += (1.0 - keep_energy) * 1.5
                    factors.append(f"энергия:{keep_energy:.0%}")
                elif select_mode == "lowpass" and seq_keep < 1.0:
                    impact += (1.0 - seq_keep) * 2.0
                    factors.append(f"частоты:{seq_keep:.0%}")

        # --- Centroid Δ - ниже лучше ---
        elif metric == "centroid":
            if method in ("fwht", "fft", "dct", "dwt"):
                if select_mode == "lowpass" and seq_keep < 1.0:
                    # Lowpass искусственно сдвигает центроид вниз
                    impact += (seq_keep - 1.0) * -0.5
                    factors.append(f"lowpass:{seq_keep:.0%}")

        # --- Cosine Similarity - выше лучше ---
        elif metric == "cosine":
            bitrate_impact = (bitrate_kbps - base_bitrate_kbps) / base_bitrate_kbps * 0.5
            impact += bitrate_impact

            if method in ("fwht", "fft", "dct", "dwt"):
                if select_mode == "energy" and keep_energy < 1.0:
                    impact += (keep_energy - 1.0) * 1.0
                    factors.append(f"энергия:{keep_energy:.0%}")

        # --- Время обработки - ниже лучше ---
        elif metric == "time":
            if method in ("fwht", "fft", "dct", "dwt"):
                # Больше блок → больше время
                block_impact = (block_size - base_block) / base_block * 0.8
                impact += block_impact
                if block_impact > 0.1:
                    factors.append(f"блок:{block_size}")

                # DWT уровни
                if method == "dwt" and levels != 4:
                    level_impact = (levels - 4) * 0.15
                    impact += level_impact
                    factors.append(f"уровни:{levels}")

        # --- Размер файла - ниже лучше ---
        elif metric == "size":
            # Битрейт прямо влияет на размер
            bitrate_impact = (bitrate_kbps - base_bitrate_kbps) / base_bitrate_kbps * 2.0
            impact += bitrate_impact
            if abs(bitrate_impact) > 0.05:
                factors.append(f"битрейт:{bitrate_kbps}k")

        # ========================================
        # ФОРМАТИРОВАНИЕ РЕЗУЛЬТАТА
        # ========================================

        # Определяем пороги для классификации
        if impact > 0.5:
            symbol = "↑↑"
            color = QColor(0, 150, 0)  # Зелёный
            desc = "Существенное улучшение"
        elif impact > 0.15:
            symbol = "↑"
            color = QColor(50, 180, 50)  # Светло-зелёный
            desc = "Улучшение"
        elif impact > -0.15:
            symbol = "→"
            color = QColor(100, 100, 100)  # Серый
            desc = "Без изменений"
        elif impact > -0.5:
            symbol = "↓"
            color = QColor(200, 100, 0)  # Оранжевый
            desc = "Ухудшение"
        else:
            symbol = "↓↓"
            color = QColor(200, 0, 0)  # Красный
            desc = "Существенное ухудшение"

        # Для метрик где "выше лучше" - инвертируем отображение
        higher_better = metric in ("snr", "sisdr", "cosine")
        if higher_better:
            # Инвертируем символы для метрик "выше лучше"
            if symbol == "↑↑":
                symbol = "↓↓"  # положительное влияние = лучше = ниже значение в таблице
                color = QColor(0, 150, 0)
            elif symbol == "↑":
                symbol = "↓"
                color = QColor(50, 180, 50)
            elif symbol == "↓↓":
                symbol = "↑↑"
                color = QColor(200, 0, 0)
            elif symbol == "↓":
                symbol = "↑"
                color = QColor(200, 100, 0)
            # → остаётся как есть

        # Формируем tooltip
        tooltip = f"{desc}\nВлияние: {impact:+.2f}"
        if factors:
            tooltip += f"\nФакторы: {', '.join(factors)}"

        return symbol, tooltip, color

    def _update_settings_matrix_table(self) -> None:
        """Обновить таблицу влияния параметров на метрики.

        Показывает качественное влияние текущих параметров на метрики для каждого метода:
        ↑↑ - существенное улучшение качества
        ↑  - улучшение
        →  - без изменений (стандартные параметры)
        ↓  - ухудшение
        ↓↓ - существенное ухудшение
        """
        settings = self._current_settings()
        methods = self._method_headers()
        metrics = self._metrics_cols

        # Проверяем, есть ли отличия от стандартных настроек
        is_default = (
            settings.get("block_size", 2048) == 2048 and
            str(settings.get("bitrate", "192k")) == "192k" and
            settings.get("select_mode", "none") == "none" and
            float(settings.get("keep_energy_ratio", 1.0)) == 1.0 and
            float(settings.get("sequency_keep_ratio", 1.0)) == 1.0 and
            int(settings.get("levels", 4)) == 4 and
            float(settings.get("mu", 255)) == 255 and
            int(settings.get("bits", 8)) == 8 and
            float(settings.get("rosen_alpha", 0.2)) == 0.2 and
            float(settings.get("rosen_beta", 1.0)) == 1.0
        )

        # Заполняем таблицу
        for ri, (method_key, _) in enumerate(methods, start=1):
            for ci, (metric_key, _, _) in enumerate(metrics, start=1):
                if is_default:
                    # При стандартных настройках - все нейтрально
                    symbol = "→"
                    color = QColor(100, 100, 100)
                    tooltip = "Стандартные параметры\nВлияние нейтральное"
                else:
                    symbol, tooltip, color = self._calculate_impact_score(
                        method_key, metric_key, settings
                    )

                it = QTableWidgetItem(symbol)
                it.setTextAlignment(Qt.AlignCenter)
                it.setForeground(color)
                it.setToolTip(tooltip)
                self.table_settings_matrix.setItem(ri, ci, it)

    def _fill_metric_hints(self) -> None:
        """Заполнить таблицу подсказок по метрикам."""
        hints = [
            ("LSD (дБ)", "Log-Spectral Distance", "Расстояние между спектрами, ниже лучше"),
            ("SNR (дБ)", "Signal-to-Noise Ratio", "Отношение сигнал/шум, выше лучше"),
            ("Спектр. сх.", "Spectral Convergence", "Ошибка амплитуд спектра, ниже лучше"),
            ("RMSE", "Root Mean Square Error", "Ошибка во временной области, ниже лучше"),
            ("SI-SDR (дБ)", "Scale-Invariant SDR", "Устойчивый к масштабу SDR, выше лучше"),
            ("Центроид Δ (Гц)", "Spectral Centroid Difference", "Разница центров спектра (высокие значения для сигналов с ВЧ шумом)"),
            ("Косин. сход.", "Cosine Similarity", "Схожесть спектров (0-1), выше лучше"),
            ("Общий балл", "Aggregate Score", "Комплексная оценка качества, выше лучше"),
        ]

        self.hints_table.setRowCount(len(hints))
        for i, (metric, desc, meaning) in enumerate(hints):
            self.hints_table.setItem(i, 0, QTableWidgetItem(metric))
            self.hints_table.setItem(i, 1, QTableWidgetItem(desc))
            self.hints_table.setItem(i, 2, QTableWidgetItem(meaning))

    # =========================================================================
    # ОБРАБОТЧИКИ СОБЫТИЙ
    # =========================================================================

    def on_browse(self) -> None:
        """Диалог выбора WAV-файла."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите WAV-файл", str(PROJECT_ROOT), "WAV файлы (*.wav)"
        )
        if path:
            self.path_edit.setText(path)
            self.btn_convert.setEnabled(True)

    def on_browse_dataset(self) -> None:
        """Диалог выбора папки с данными."""
        start_dir = self._dataset_folder or str(PROJECT_ROOT)
        folder = QFileDialog.getExistingDirectory(
            self, 
            "Выберите папку с WAV-файлами",
            start_dir,
            QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            self._dataset_folder = folder
            self.dataset_edit.setText(folder)
            # Подсчитать количество WAV-файлов
            wavs = glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True)
            self.btn_batch.setToolTip(f"Обработать {len(wavs)} WAV-файлов из {folder}")

    def on_convert(self) -> None:
        """Запустить обработку выбранного файла."""
        path = self.path_edit.text().strip()
        if not path:
            return
        if not os.path.exists(path):
            QMessageBox.warning(self, "Ошибка", "Файл не найден")
            return
        self._start_worker([path], dataset_root=None)

    def on_run_dataset(self) -> None:
        """Запустить пакетную обработку."""
        # Используем выбранную папку или дефолтную
        dataset_root = self._dataset_folder
        if not dataset_root:
            dataset_root = str(DEFAULT_TEST_DATA_DIR)
        
        if not os.path.isdir(dataset_root):
            QMessageBox.warning(
                self, 
                "Папка не найдена", 
                f"Папка '{dataset_root}' не существует.\nВыберите папку с WAV-файлами."
            )
            return
        
        # Поиск WAV-файлов рекурсивно
        wavs = sorted(glob.glob(os.path.join(dataset_root, "**", "*.wav"), recursive=True))
        
        if not wavs:
            QMessageBox.information(
                self, 
                "Нет файлов", 
                f"В папке '{dataset_root}' не найдено WAV-файлов.\n\n"
                "Структура должна быть:\n"
                "  папка/\n"
                "  ├── жанр1/\n"
                "  │   ├── track1.wav\n"
                "  │   └── track2.wav\n"
                "  └── жанр2/\n"
                "      └── track3.wav"
            )
            return
        
        # Показать информацию о найденных файлах
        genres = set()
        for w in wavs:
            rel = os.path.relpath(w, dataset_root)
            parts = rel.split(os.sep)
            if len(parts) > 1:
                genres.add(parts[0])
        
        genre_info = f" ({len(genres)} жанров)" if genres else ""
        self._log_emitter.log_line.emit(
            f"Найдено {len(wavs)} WAV-файлов{genre_info} в {dataset_root}"
        )
        
        self._start_worker(wavs, dataset_root=dataset_root)

    def on_export_xlsx(self) -> None:
        """Экспортировать результаты в Excel файл."""
        if not self._results:
            QMessageBox.information(
                self,
                "Нет данных",
                "Нет результатов для экспорта.\nСначала обработайте аудио файлы."
            )
            return

        if not is_export_available():
            QMessageBox.warning(
                self,
                "Недоступно",
                "Для экспорта в Excel необходимо установить библиотеку openpyxl.\n\n"
                "Установите: pip install openpyxl"
            )
            return

        # Диалог сохранения файла
        default_name = generate_export_filename("audio_analysis")
        default_path = os.path.join(str(OUTPUT_DIR), default_name)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результаты",
            default_path,
            "Excel файлы (*.xlsx)"
        )

        if not file_path:
            return

        # Убеждаемся что расширение .xlsx
        if not file_path.lower().endswith(".xlsx"):
            file_path += ".xlsx"

        # Экспорт
        success = export_results_to_xlsx(self._results, file_path)

        if success:
            QMessageBox.information(
                self,
                "Успешно",
                f"Результаты сохранены в:\n{file_path}"
            )
            self._log_emitter.log_line.emit(f"Экспорт завершён: {file_path}")
        else:
            QMessageBox.warning(
                self,
                "Ошибка",
                f"Не удалось сохранить файл.\nПодробнее см. в логах."
            )

    def closeEvent(self, event) -> None:
        """Аккуратно завершить поток при закрытии."""
        try:
            if self._thread and self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(1500)
        except Exception:
            pass
        super().closeEvent(event)

    def _on_toggle_logs(self, checked: bool) -> None:
        """Показать/скрыть панель логов."""
        self.logs_tabs.setVisible(bool(checked))

    def _on_variant_visibility(self) -> None:
        """Обновить видимость методов на графиках."""
        for v, cb in self._variant_cbs.items():
            self._variant_visible[v] = cb.isChecked()
        self._refresh_chart()
        self._refresh_heatmap()

    def _toggle_heatmap(self, visible: bool) -> None:
        """Показать/скрыть heatmap."""
        self.table_heatmap.setVisible(visible)

    def _toggle_hints(self, visible: bool) -> None:
        """Показать/скрыть подсказки."""
        self.hints_table.setVisible(visible)

    # =========================================================================
    # УПРАВЛЕНИЕ WORKER
    # =========================================================================

    def _current_settings(self) -> Dict[str, Any]:
        """Получить текущие настройки из виджетов."""
        def fnum(ed, cast):
            try:
                return cast(ed.text().strip())
            except Exception:
                return cast(type(cast())())

        return {
            "block_size": fnum(self.ed_block, int),
            "bitrate": self.ed_bitrate.text().strip() or "192k",
            "select_mode": self.cb_select.currentData() or "none",
            "keep_energy_ratio": fnum(self.ed_keep_energy, float),
            "sequency_keep_ratio": fnum(self.ed_seq_keep, float),
            "levels": fnum(self.ed_levels, int),
            "mu": fnum(self.ed_mu, float),
            "bits": fnum(self.ed_bits, int),
            "rosen_alpha": fnum(self.ed_ra, float),
            "rosen_beta": fnum(self.ed_rb, float),
        }

    def _start_worker(self, wav_paths: List[str], dataset_root: Optional[str]) -> None:
        """Запустить фоновую обработку."""
        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "Занято", "Обработка уже идёт")
            return

        self._results = []
        out_dir = str(OUTPUT_DIR)
        os.makedirs(out_dir, exist_ok=True)

        # Скрываем прогресс до начала
        self.progress_total.setValue(0)
        self.progress_total.setVisible(True)
        self.progress_file.setValue(0)
        self.progress_file.setVisible(True)
        self.table.setSortingEnabled(False)

        # Создаём поток
        self._thread = QThread(self)
        self._worker = Worker(wav_paths, out_dir, dataset_root, self._current_settings())
        self._worker.moveToThread(self._thread)

        # Связываем сигналы
        self._thread.started.connect(self._worker.run)
        self._worker.result.connect(self._on_worker_result)
        self._worker.error.connect(lambda m: self._append_log(f"Ошибка: {m}"))
        self._worker.status.connect(self.status_label.setText)
        self._worker.progress_file.connect(self._on_progress_file)
        self._worker.progress_total.connect(self._on_progress_total)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._thread.quit)

        self._thread.start()

    def _on_worker_finished(self) -> None:
        """Завершение обработки."""
        self.table.setSortingEnabled(True)
        self.status_label.setText("Готово")
        self.progress_file.setVisible(False)
        self.progress_total.setVisible(False)
        # Обновляем список файлов в плеере
        self.refresh_output_files_list()

    def _append_log(self, text: str) -> None:
        """Добавить строку в лог."""
        self.logs_edit.appendPlainText(text)

    def _on_progress_file(self, value: int) -> None:
        """Обновить прогресс файла."""
        self.progress_file.setValue(value)

    def _on_progress_total(self, value: int) -> None:
        """Обновить прогресс набора."""
        self.progress_total.setValue(value)

    @Slot(object)
    def _on_worker_result(self, payload: object) -> None:
        """Обработать результат от Worker."""
        try:
            d = dict(payload)  # type: ignore
        except Exception:
            return

        source = str(d.get("source", ""))
        genre = d.get("genre")
        rows: List[ResultRow] = []

        for item in d.get("results", []) or []:
            try:
                rows.append(ResultRow(
                    source=source,
                    genre=genre if isinstance(genre, str) else None,
                    variant=str(item.get("variant")),
                    path=str(item.get("path")),
                    size_bytes=int(item.get("size_bytes", 0)),
                    lsd_db=float(item.get("lsd_db", float("nan"))),
                    snr_db=float(item.get("snr_db", float("nan"))),
                    spec_conv=float(item.get("spec_conv", float("nan"))),
                    rmse=float(item.get("rmse", float("nan"))),
                    si_sdr_db=float(item.get("si_sdr_db", float("nan"))),
                    spec_centroid_diff_hz=float(item.get("spec_centroid_diff_hz", float("nan"))),
                    spec_cosine=float(item.get("spec_cosine", float("nan"))),
                    score=float(item.get("score", float("nan"))),
                    time_sec=float(item.get("time_sec", float("nan"))),
                ))
            except Exception:
                continue

        if not rows:
            return

        for r in rows:
            self._results.append(r)
            self._append_table_row(r)

        self._refresh_chart()
        self._refresh_heatmap()

    def _append_table_row(self, r: ResultRow) -> None:
        """Добавить строку в таблицу результатов."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        # Включаем кнопку экспорта при появлении данных
        if not self.btn_export_xlsx.isEnabled():
            self.btn_export_xlsx.setEnabled(True)

        def set_col(col: int, text: str, align_right: bool = False):
            it = QTableWidgetItem(text)
            if align_right:
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, col, it)

        size_mb = f"{(r.size_bytes or 0) / (1024 * 1024):.3f}"

        set_col(0, r.source)
        set_col(1, r.variant)
        set_col(2, size_mb, True)
        set_col(3, f"{r.lsd_db:.3f}", True)
        set_col(4, f"{r.snr_db:.3f}", True)
        set_col(5, f"{r.spec_conv:.3f}", True)
        set_col(6, f"{r.rmse:.5f}", True)
        set_col(7, f"{r.si_sdr_db:.3f}", True)
        set_col(8, f"{r.spec_centroid_diff_hz:.3f}", True)
        set_col(9, f"{r.spec_cosine:.4f}", True)
        set_col(10, f"{r.score:.4f}", True)
        set_col(11, f"{r.time_sec:.3f}", True)
        self.table.setItem(row, 12, QTableWidgetItem(r.path))

    # =========================================================================
    # ВИЗУАЛИЗАЦИЯ
    # =========================================================================

    def _refresh_chart(self) -> None:
        """Перестроить график сравнения.
        
        Создаёт столбчатую диаграмму для сравнения методов по выбранной метрике.
        Группировка данных: сводка (все треки), по жанрам, по отдельным трекам.
        """
        metric_key = self.combo_metric.currentData() or "lsd"
        title, key, _ = METRIC_KEYS.get(metric_key, METRIC_KEYS["lsd"])
        scope = self.combo_scope.currentData() or "summary"

        # Удаляем старые оси
        for ax in list(self.chart.axes()):
            try:
                self.chart.removeAxis(ax)
            except Exception:
                pass

        # Группируем данные по категориям (сводка/жанры/треки)
        groups: Dict[str, Dict[str, List[float]]] = {}
        for r in self._results:
            if not self._variant_visible.get(r.variant, True):
                continue
            grp = "Все треки" if scope == "summary" else (r.genre or "—") if scope == "genres" else r.source
            groups.setdefault(grp, {}).setdefault(r.variant, []).append(getattr(r, key))

        # Строим диаграмму
        self.chart.removeAllSeries()
        categories = list(groups.keys())
        
        if not categories:
            # Нет данных - показываем пустой график
            self.chart.setTitle(f"Сравнение: {title} (нет данных)")
            return
            
        series = QBarSeries()

        variants_in_data = {v for g in groups.values() for v in g.keys()}

        for v in VARIANTS:
            if v not in variants_in_data:
                continue
            if not self._variant_visible.get(v, True):
                continue

            bs = QBarSet(v)
            for g in categories:
                vals = groups.get(g, {}).get(v, [])
                vals_f = [float(x) for x in vals if isinstance(x, (int, float)) and math.isfinite(float(x))]
                avg = sum(vals_f) / len(vals_f) if vals_f else 0.0
                bs << avg
            series.append(bs)

        self.chart.addSeries(series)

        # Включаем метки значений на столбцах
        try:
            series.setLabelsVisible(True)
        except Exception:
            pass

        # Ось X (категории)
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        self.chart.addAxis(axis_x, Qt.AlignBottom)
        self.chart.setAxisX(axis_x, series)

        # Ось Y - вычисляем диапазон
        all_vals = []
        for s in series.barSets():
            try:
                all_vals += [s.at(i) for i in range(len(categories))]
            except Exception:
                pass

        if not all_vals:
            all_vals = [0.0, 1.0]
            
        ymin = min(all_vals) if all_vals else 0.0
        ymax = max(all_vals) if all_vals else 1.0
        if ymin == ymax:
            ymax = ymin + 1.0

        # Добавляем отступ для меток
        padding = 0.15 * abs(ymax - ymin) if ymax != ymin else 0.1
        axis_y = QValueAxis()
        axis_y.setRange(ymin - 0.05 * abs(ymax - ymin), ymax + padding)
        axis_y.setLabelsVisible(True)

        self.chart.addAxis(axis_y, Qt.AlignLeft)
        self.chart.setAxisY(axis_y, series)

        self.chart.setTitle(f"Сравнение: {title}")
        self.chart.legend().setVisible(True)

    def _refresh_heatmap(self) -> None:
        """Перестроить тепловую карту."""
        scope = self.combo_scope.currentData() or "summary"
        metric_key = self.combo_metric.currentData() or "lsd"
        _, key, _ = METRIC_KEYS.get(metric_key, METRIC_KEYS["lsd"])

        # Определяем категории
        categories = set()
        for r in self._results:
            if scope == "summary":
                categories.add("Все треки")
            elif scope == "genres":
                categories.add(r.genre or "—")
            else:
                categories.add(r.source)

        categories = sorted(categories)
        variants = [v for v in VARIANTS if self._variant_visible.get(v, True)]

        # Настраиваем таблицу
        self.table_heatmap.setRowCount(len(variants))
        self.table_heatmap.setColumnCount(len(categories))
        self.table_heatmap.setHorizontalHeaderLabels(categories)
        self.table_heatmap.setVerticalHeaderLabels(variants)

        # Заполняем значениями
        for vi, variant in enumerate(variants):
            for ci, cat in enumerate(categories):
                vals = []
                for r in self._results:
                    if r.variant != variant:
                        continue
                    if scope == "summary":
                        vals.append(getattr(r, key))
                    elif scope == "genres":
                        if (r.genre or "—") == cat:
                            vals.append(getattr(r, key))
                    else:
                        if r.source == cat:
                            vals.append(getattr(r, key))

                vals_f = [float(x) for x in vals if isinstance(x, (int, float)) and math.isfinite(float(x))]
                avg = sum(vals_f) / len(vals_f) if vals_f else float("nan")

                it = QTableWidgetItem(f"{avg:.3f}" if math.isfinite(avg) else "—")
                it.setTextAlignment(Qt.AlignCenter)
                self.table_heatmap.setItem(vi, ci, it)

    # =========================================================================
    # ПЛЕЕР
    # =========================================================================

    def on_browse_player_file(self) -> None:
        """Диалог выбора аудиофайла для воспроизведения."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите аудиофайл",
            str(PROJECT_ROOT),
            "Аудиофайлы (*.wav *.mp3);;WAV файлы (*.wav);;MP3 файлы (*.mp3)"
        )
        if file_path:
            self._load_player_file(file_path)

    def _load_player_file(self, file_path: str) -> None:
        """Загрузить файл в плеер."""
        import datetime
        self._current_player_file = file_path
        self.player_file_edit.setText(file_path)
        
        # Информация о файле
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            self.player_info_label.setText(
                f"📁 {os.path.basename(file_path)} | {size_mb:.2f} МБ | Изменён: {mtime.strftime('%Y-%m-%d %H:%M')}"
            )
            self.player_info_label.setStyleSheet("color: #333; margin: 5px;")
        except Exception as e:
            self.player_info_label.setText(f"Ошибка: {e}")
            self.player_info_label.setStyleSheet("color: red; margin: 5px;")
        
        # Загружаем в плеер
        self._media_player.setSource(QUrl.fromLocalFile(file_path))
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

    def on_player_play(self) -> None:
        """Начать воспроизведение."""
        self._media_player.play()

    def on_player_pause(self) -> None:
        """Приостановить воспроизведение."""
        if self._media_player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
            self._media_player.play()
        else:
            self._media_player.pause()

    def on_player_stop(self) -> None:
        """Остановить воспроизведение."""
        self._media_player.stop()
        self.position_slider.setValue(0)
        self.position_label.setText("00:00")

    def on_volume_changed(self, value: int) -> None:
        """Изменить громкость."""
        self._audio_output.setVolume(value / 100.0)
        self.volume_label.setText(f"{value}%")

    def on_player_position_changed(self, position: int) -> None:
        """Обновить позицию воспроизведения."""
        duration = self._media_player.duration()
        if duration > 0:
            self.position_slider.setValue(int(position / duration * 1000))
        
        # Форматирование времени
        seconds = position // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        self.position_label.setText(f"{minutes:02d}:{seconds:02d}")

    def on_player_duration_changed(self, duration: int) -> None:
        """Обновить длительность трека."""
        seconds = duration // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        self.duration_label.setText(f"{minutes:02d}:{seconds:02d}")

    def on_player_state_changed(self, state) -> None:
        """Изменение состояния плеера."""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play.setText("▶️ Играет...")
            self.btn_play.setEnabled(False)
            self.btn_pause.setEnabled(True)
        elif state == QMediaPlayer.PlaybackState.PausedState:
            self.btn_play.setText("▶️ Воспроизвести")
            self.btn_play.setEnabled(True)
            self.btn_pause.setText("▶️ Продолжить")
        else:  # StoppedState
            self.btn_play.setText("▶️ Воспроизвести")
            self.btn_play.setEnabled(True)
            self.btn_pause.setText("⏸️ Пауза")
            self.btn_pause.setEnabled(True)

    def on_player_error(self) -> None:
        """Обработка ошибки плеера."""
        error = self._media_player.errorString()
        if error:
            self.player_info_label.setText(f"❌ Ошибка: {error}")
            self.player_info_label.setStyleSheet("color: red; margin: 5px;")

    def on_position_slider_moved(self, value: int) -> None:
        """Перемотка по слайдеру."""
        duration = self._media_player.duration()
        if duration > 0:
            position = int(value / 1000 * duration)
            self._media_player.setPosition(position)

    def refresh_output_files_list(self) -> None:
        """Обновить список файлов в папке output."""
        import datetime
        self.output_files_list.setRowCount(0)
        
        output_dir = str(OUTPUT_DIR)
        if not os.path.isdir(output_dir):
            return
        
        # Получаем список файлов
        files = []
        for f in os.listdir(output_dir):
            if f.endswith(('.mp3', '.wav')):
                path = os.path.join(output_dir, f)
                try:
                    size = os.path.getsize(path) / (1024 * 1024)
                    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
                    files.append((f, size, mtime, path))
                except Exception:
                    continue
        
        # Сортируем по дате (новые первыми)
        files.sort(key=lambda x: x[2], reverse=True)
        
        # Заполняем таблицу
        for f, size, mtime, path in files:
            row = self.output_files_list.rowCount()
            self.output_files_list.insertRow(row)
            self.output_files_list.setItem(row, 0, QTableWidgetItem(f))
            self.output_files_list.setItem(row, 1, QTableWidgetItem(f"{size:.2f} МБ"))
            self.output_files_list.setItem(row, 2, QTableWidgetItem(mtime.strftime('%Y-%m-%d %H:%M')))
            # Сохраняем путь в data
            self.output_files_list.item(row, 0).setData(Qt.UserRole, path)

    def on_output_file_double_clicked(self, index) -> None:
        """Двойной клик по файлу в списке - воспроизвести."""
        row = index.row()
        item = self.output_files_list.item(row, 0)
        if item:
            path = item.data(Qt.UserRole)
            if path and os.path.exists(path):
                self._load_player_file(path)

    # =========================================================================
    # ОЧИСТКА OUTPUT
    # =========================================================================

    def on_clear_output(self) -> None:
        """Очистить папку output."""
        output_dir = str(OUTPUT_DIR)
        if not os.path.isdir(output_dir):
            QMessageBox.information(self, "Папка не найдена", "Папка output не существует.")
            return
        
        # Подсчитываем файлы
        files = [f for f in os.listdir(output_dir) if f.endswith(('.mp3', '.wav'))]
        if not files:
            QMessageBox.information(self, "Пусто", "Папка output уже пуста.")
            return
        
        # Подтверждение
        reply = QMessageBox.question(
            self,
            "Подтверждение",
            f"Удалить {len(files)} файл(ов) из папки output?\nЭто действие нельзя отменить.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Удаляем файлы
        deleted = 0
        for f in files:
            try:
                os.remove(os.path.join(output_dir, f))
                deleted += 1
            except Exception as e:
                self._log_emitter.log_line.emit(f"Ошибка удаления {f}: {e}")
        
        # Обновляем список
        self.refresh_output_files_list()
        
        QMessageBox.information(
            self,
            "Готово",
            f"Удалено {deleted} файл(ов) из папки output."
        )
        self._log_emitter.log_line.emit(f"Очищена папка output: удалено {deleted} файлов")

    # =========================================================================
    # РАБОТА С ИСХОДНЫМИ ФАЙЛАМИ
    # =========================================================================

    def refresh_source_files_list(self) -> None:
        """Обновить список исходных файлов из результатов обработки."""
        self.source_files_list.setRowCount(0)
        
        # Собираем уникальные исходные файлы из результатов
        source_files = {}  # name -> (size, path)
        for r in self._results:
            if r.source not in source_files:
                # Ищем исходный файл
                source_path = r.path.replace('_std.mp3', '.wav').replace('_fwht.mp3', '.wav')
                source_path = source_path.replace('_fft.mp3', '.wav').replace('_dct.mp3', '.wav')
                source_path = source_path.replace('_dwt.mp3', '.wav').replace('_huffman.mp3', '.wav')
                source_path = source_path.replace('_rosenbrock.mp3', '.wav')
                
                # Пробуем разные варианты
                possible_paths = [
                    os.path.join(os.path.dirname(r.path), r.source),
                    source_path,
                ]
                
                actual_path = None
                for p in possible_paths:
                    if os.path.exists(p):
                        actual_path = p
                        break
                
                if actual_path:
                    try:
                        size = os.path.getsize(actual_path) / (1024 * 1024)
                        source_files[r.source] = (size, actual_path)
                    except Exception:
                        source_files[r.source] = (0, actual_path)
        
        # Заполняем таблицу
        for name, (size, path) in sorted(source_files.items()):
            row = self.source_files_list.rowCount()
            self.source_files_list.insertRow(row)
            self.source_files_list.setItem(row, 0, QTableWidgetItem(name))
            self.source_files_list.setItem(row, 1, QTableWidgetItem(f"{size:.2f} МБ"))
            self.source_files_list.item(row, 0).setData(Qt.UserRole, path)
        
        # Добавляем папку с данными если есть
        if self._dataset_folder and os.path.isdir(self._dataset_folder):
            self._add_source_files_from_dir(self._dataset_folder)

    def _add_source_files_from_dir(self, directory: str) -> None:
        """Добавить исходные файлы из директории."""
        existing = set()
        for row in range(self.source_files_list.rowCount()):
            item = self.source_files_list.item(row, 0)
            if item:
                existing.add(item.text())
        
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith('.wav') and f not in existing:
                    path = os.path.join(root, f)
                    try:
                        size = os.path.getsize(path) / (1024 * 1024)
                        row = self.source_files_list.rowCount()
                        self.source_files_list.insertRow(row)
                        self.source_files_list.setItem(row, 0, QTableWidgetItem(f))
                        self.source_files_list.setItem(row, 1, QTableWidgetItem(f"{size:.2f} МБ"))
                        self.source_files_list.item(row, 0).setData(Qt.UserRole, path)
                        existing.add(f)
                    except Exception:
                        continue

    def on_source_file_selected(self) -> None:
        """При выборе исходного файла - фильтруем список обработанных."""
        selected = self.source_files_list.selectedItems()
        if not selected:
            self.refresh_output_files_list()
            return
        
        row = selected[0].row()
        item = self.source_files_list.item(row, 0)
        if item:
            source_name = item.text()
            self._filter_output_files_by_source(source_name)

    def _filter_output_files_by_source(self, source_name: str) -> None:
        """Фильтровать обработанные файлы по исходному."""
        import datetime
        
        self.output_files_list.setRowCount(0)
        
        # Извлекаем базовое имя без расширения
        base_name = os.path.splitext(source_name)[0]
        
        output_dir = str(OUTPUT_DIR)
        if not os.path.isdir(output_dir):
            return
        
        # Ищем файлы с совпадающим базовым именем
        files = []
        for f in os.listdir(output_dir):
            if f.startswith(base_name + '_') and f.endswith('.mp3'):
                path = os.path.join(output_dir, f)
                try:
                    size = os.path.getsize(path) / (1024 * 1024)
                    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
                    
                    # Извлекаем метод из имени файла
                    parts = f.replace(base_name + '_', '').replace('.mp3', '')
                    method_map = {
                        'std': 'Стандартный',
                        'fwht': 'FWHT',
                        'fft': 'FFT',
                        'dct': 'DCT',
                        'dwt': 'DWT',
                        'huffman': 'Хаффман',
                        'rosenbrock': 'Розенброк',
                    }
                    method = method_map.get(parts, parts)
                    
                    files.append((method, size, mtime, path))
                except Exception:
                    continue
        
        # Сортируем по дате
        files.sort(key=lambda x: x[2], reverse=True)
        
        # Заполняем таблицу
        for method, size, mtime, path in files:
            row = self.output_files_list.rowCount()
            self.output_files_list.insertRow(row)
            self.output_files_list.setItem(row, 0, QTableWidgetItem(method))
            self.output_files_list.setItem(row, 1, QTableWidgetItem(f"{size:.2f} МБ"))
            self.output_files_list.setItem(row, 2, QTableWidgetItem(mtime.strftime('%Y-%m-%d %H:%M')))
            self.output_files_list.item(row, 0).setData(Qt.UserRole, path)

    def on_source_file_double_clicked(self, index) -> None:
        """Двойной клик по исходному файлу - воспроизвести."""
        row = index.row()
        item = self.source_files_list.item(row, 0)
        if item:
            path = item.data(Qt.UserRole)
            if path and os.path.exists(path):
                self._load_player_file(path)

    def on_add_source_file(self) -> None:
        """Добавить исходный файл вручную."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите исходный аудиофайл",
            str(PROJECT_ROOT),
            "Аудиофайлы (*.wav *.mp3);;WAV файлы (*.wav);;MP3 файлы (*.mp3)"
        )
        if path:
            self._add_source_file_to_list(path)

    def _add_source_file_to_list(self, path: str) -> None:
        """Добавить файл в список исходных."""
        name = os.path.basename(path)
        
        # Проверяем на дубликаты
        for row in range(self.source_files_list.rowCount()):
            item = self.source_files_list.item(row, 0)
            if item and item.text() == name:
                return  # Уже есть
        
        try:
            size = os.path.getsize(path) / (1024 * 1024)
            row = self.source_files_list.rowCount()
            self.source_files_list.insertRow(row)
            self.source_files_list.setItem(row, 0, QTableWidgetItem(name))
            self.source_files_list.setItem(row, 1, QTableWidgetItem(f"{size:.2f} МБ"))
            self.source_files_list.item(row, 0).setData(Qt.UserRole, path)
        except Exception as e:
            self._log_emitter.log_line.emit(f"Ошибка добавления файла: {e}")

    def on_open_output_folder(self) -> None:
        """Открыть папку output в файловом менеджере."""
        output_dir = str(OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        
        import subprocess
        import platform
        
        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", output_dir])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])
        except Exception as e:
            QMessageBox.information(
                self,
                "Путь к папке",
                f"Папка output:\n{output_dir}"
            )

    # =========================================================================
    # СПЕКТРАЛЬНЫЙ АНАЛИЗ
    # =========================================================================

    def on_browse_spectrum_source(self) -> None:
        """Выбрать исходный файл для спектрального анализа.
        
        Сначала предлагает выбрать из уже обработанных файлов,
        если нет - открывает диалог выбора из файловой системы.
        """
        # Собираем доступные исходные файлы
        available_sources = self._get_available_source_files()
        
        if available_sources:
            # Показываем диалог выбора из доступных файлов
            dialog = SourceFilesDialog(self, available_sources)
            if dialog.exec():
                selected_path = dialog.get_selected_path()
                if selected_path == "__BROWSE__":
                    # Пользователь выбрал "Выбрать другой файл"
                    self._browse_spectrum_from_filesystem()
                elif selected_path:
                    self.spectrum_source_edit.setText(selected_path)
                    self.spectrum_source_edit.setToolTip(selected_path)
                    self.refresh_spectrum_files_list()
        else:
            # Нет доступных файлов - открываем диалог
            self._browse_spectrum_from_filesystem()
    
    def _get_available_source_files(self) -> List[Tuple[str, str]]:
        """Получить список доступных исходных файлов.
        
        Возвращает список кортежей (имя_файла, путь).
        """
        sources = {}  # name -> path
        
        # Из результатов обработки
        for r in self._results:
            if r.source and r.source not in sources:
                # Пытаемся найти исходный файл
                base_name = os.path.splitext(r.source)[0]
                
                # Ищем в директории обработанного файла
                output_dir = os.path.dirname(r.path)
                possible_paths = [
                    os.path.join(output_dir, r.source),
                    os.path.join(str(OUTPUT_DIR), r.source),
                ]
                
                # Также ищем в директории данных
                if self._dataset_folder:
                    for root, dirs, files in os.walk(self._dataset_folder):
                        if r.source in files:
                            possible_paths.append(os.path.join(root, r.source))
                
                for p in possible_paths:
                    if os.path.exists(p):
                        sources[r.source] = p
                        break
        
        # Из выбранного файла
        current_path = self.path_edit.text().strip()
        if current_path and os.path.exists(current_path):
            name = os.path.basename(current_path)
            sources[name] = current_path
        
        # Из папки с данными
        if self._dataset_folder and os.path.isdir(self._dataset_folder):
            for root, dirs, files in os.walk(self._dataset_folder):
                for f in files:
                    if f.endswith('.wav') and f not in sources:
                        path = os.path.join(root, f)
                        sources[f] = path
        
        return list(sources.items())
    
    def _browse_spectrum_from_filesystem(self) -> None:
        """Открыть диалог выбора файла из файловой системы."""
        start_dir = self._dataset_folder or str(PROJECT_ROOT)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите исходный аудиофайл",
            start_dir,
            "Аудиофайлы (*.wav *.mp3)"
        )
        if path:
            self.spectrum_source_edit.setText(path)
            self.spectrum_source_edit.setToolTip(path)
            self.refresh_spectrum_files_list()

    def refresh_spectrum_files_list(self) -> None:
        """Обновить список обработанных файлов для спектрального сравнения."""
        self.spectrum_files_table.setRowCount(0)
        
        source_path = self.spectrum_source_edit.text()
        if not source_path or not os.path.exists(source_path):
            return
        
        # Получаем базовое имя исходного файла
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        
        # Ищем обработанные файлы
        output_dir = str(OUTPUT_DIR)
        if not os.path.isdir(output_dir):
            return
        
        import datetime
        
        # Методы для сравнения
        method_names = {
            'std': 'Стандартный',
            'fwht': 'FWHT',
            'fft': 'FFT',
            'dct': 'DCT',
            'dwt': 'DWT',
            'huffman': 'Хаффман',
            'rosenbrock': 'Розенброк',
        }
        
        files = []
        for f in os.listdir(output_dir):
            if f.startswith(base_name + '_') and f.endswith('.mp3'):
                # Извлекаем метод из имени
                method_key = f.replace(base_name + '_', '').replace('.mp3', '')
                method_name = method_names.get(method_key, method_key)
                
                path = os.path.join(output_dir, f)
                try:
                    size = os.path.getsize(path) / (1024 * 1024)
                    files.append((method_key, method_name, f, size, path))
                except Exception:
                    continue
        
        # Сортируем по методу
        files.sort(key=lambda x: x[0])
        
        # Заполняем таблицу
        for method_key, method_name, filename, size, path in files:
            row = self.spectrum_files_table.rowCount()
            self.spectrum_files_table.insertRow(row)
            
            # Чекбокс
            chk = QCheckBox()
            chk.setChecked(True)
            self.spectrum_files_table.setCellWidget(row, 0, chk)
            
            # Метод
            self.spectrum_files_table.setItem(row, 1, QTableWidgetItem(method_name))
            
            # Файл
            self.spectrum_files_table.setItem(row, 2, QTableWidgetItem(filename))
            
            # Размер
            self.spectrum_files_table.setItem(row, 3, QTableWidgetItem(f"{size:.2f} МБ"))
            
            # Сохраняем путь
            self.spectrum_files_table.item(row, 1).setData(Qt.UserRole, path)
            self.spectrum_files_table.item(row, 1).setData(Qt.UserRole + 1, method_key)

    def on_compare_spectrum(self) -> None:
        """Сравнить спектры выбранных файлов."""
        source_path = self.spectrum_source_edit.text()
        if not source_path or not os.path.exists(source_path):
            QMessageBox.warning(self, "Ошибка", "Выберите исходный файл")
            return
        
        # Собираем выбранные файлы
        selected_files = []
        for row in range(self.spectrum_files_table.rowCount()):
            chk = self.spectrum_files_table.cellWidget(row, 0)
            if chk and chk.isChecked():
                item = self.spectrum_files_table.item(row, 1)
                if item:
                    path = item.data(Qt.UserRole)
                    method = item.data(Qt.UserRole + 1)
                    if path and os.path.exists(path):
                        selected_files.append((method, path))
        
        if not selected_files:
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы один метод для сравнения")
            return
        
        # Очищаем график
        self.spectrum_chart.removeAllSeries()
        for axis in list(self.spectrum_chart.axes()):
            self.spectrum_chart.removeAxis(axis)
        
        # Загружаем и анализируем исходный файл
        try:
            from processing.codecs import load_wav_mono, decode_audio_to_mono
            self._log_emitter.log_line.emit(f"Загрузка исходного файла: {source_path}")
            try:
                source_signal, source_sr = load_wav_mono(source_path)
            except Exception:
                source_signal, source_sr = decode_audio_to_mono(source_path)
            
            self._log_emitter.log_line.emit(f"Загружено: {len(source_signal)} сэмплов, {source_sr} Гц")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить исходный файл: {e}")
            self._log_emitter.log_line.emit(f"Ошибка загрузки: {e}")
            return
        
        # Вычисляем спектр исходного файла
        source_spectrum = self._compute_spectrum(source_signal, source_sr)
        self._log_emitter.log_line.emit(f"Спектр исходного: {len(source_spectrum)} точек")
        
        # Цвета для графиков
        colors = [
            QColor(0, 0, 0),      # Исходный - черный
            QColor(255, 0, 0),     # Красный
            QColor(0, 128, 0),     # Зелёный
            QColor(0, 0, 255),     # Синий
            QColor(255, 165, 0),   # Оранжевый
            QColor(128, 0, 128),   # Фиолетовый
            QColor(0, 128, 128),   # Бирюзовый
            QColor(255, 192, 203), # Розовый
        ]
        
        # Добавляем исходный спектр
        source_series = QLineSeries()
        source_series.setName("Исходный")
        source_series.setColor(colors[0])
        pen = QPen(colors[0])
        pen.setWidth(2)
        source_series.setPen(pen)
        
        # Нормализуем частоты для отображения
        max_points = 500  # Ограничиваем количество точек
        step = max(1, len(source_spectrum) // max_points)
        
        points_added = 0
        for i in range(0, len(source_spectrum), step):
            freq = i * source_sr / (2 * len(source_spectrum))
            if freq < 20000:  # Ограничиваем до 20 кГц
                try:
                    val = 20 * math.log10(max(source_spectrum[i], 1e-10))
                    source_series.append(freq, val)
                    points_added += 1
                except Exception:
                    pass
        
        self._log_emitter.log_line.emit(f"Исходный спектр: добавлено {points_added} точек")
        
        if points_added > 0:
            self.spectrum_chart.addSeries(source_series)
        
        # Добавляем спектры обработанных файлов
        for idx, (method, path) in enumerate(selected_files):
            try:
                self._log_emitter.log_line.emit(f"Загрузка {method}: {path}")
                try:
                    signal, sr = decode_audio_to_mono(path)
                except Exception:
                    signal, sr = load_wav_mono(path)
                
                spectrum = self._compute_spectrum(signal, sr)
                
                series = QLineSeries()
                series.setName(method.upper())
                color = colors[(idx + 1) % len(colors)]
                series.setColor(color)
                pen = QPen(color)
                pen.setWidth(2)
                series.setPen(pen)
                
                # Ресемплируем спектр если нужно
                if len(spectrum) != len(source_spectrum):
                    try:
                        from scipy.interpolate import interp1d
                        old_freqs = np.linspace(0, sr/2, len(spectrum))
                        new_freqs = np.linspace(0, source_sr/2, len(source_spectrum))
                        interp = interp1d(old_freqs, spectrum, kind='linear', fill_value='extrapolate')
                        spectrum = interp(new_freqs)
                    except ImportError:
                        # scipy не установлен - используем простое усреднение
                        self._log_emitter.log_line.emit("scipy не установлен, используется простая интерполяция")
                        spectrum = np.interp(
                            np.linspace(0, sr/2, len(source_spectrum)),
                            np.linspace(0, sr/2, len(spectrum)),
                            spectrum
                        )
                
                points_added = 0
                for i in range(0, min(len(spectrum), len(source_spectrum)), step):
                    freq = i * source_sr / (2 * len(source_spectrum))
                    if freq < 20000:
                        try:
                            val = 20 * math.log10(max(spectrum[i], 1e-10))
                            series.append(freq, val)
                            points_added += 1
                        except Exception:
                            pass
                
                self._log_emitter.log_line.emit(f"{method}: добавлено {points_added} точек")
                
                if points_added > 0:
                    self.spectrum_chart.addSeries(series)
                
            except Exception as e:
                self._log_emitter.log_line.emit(f"Ошибка при анализе {method}: {e}")
        
        # Настраиваем оси только если есть серии
        if self.spectrum_chart.series():
            axis_x = QValueAxis()
            axis_x.setTitleText("Частота (Гц)")
            axis_x.setRange(20, 20000)
            self.spectrum_chart.addAxis(axis_x, Qt.AlignBottom)
            
            axis_y = QValueAxis()
            axis_y.setTitleText("Амплитуда (дБ)")
            # Автоопределение диапазона Y
            all_vals = []
            for series in self.spectrum_chart.series():
                for i in range(series.count()):
                    point = series.at(i)
                    all_vals.append(point.y())
            
            if all_vals:
                y_min = min(all_vals)
                y_max = max(all_vals)
                margin = 0.1 * (y_max - y_min) if y_max != y_min else 10
                axis_y.setRange(y_min - margin, y_max + margin)
            
            self.spectrum_chart.addAxis(axis_y, Qt.AlignLeft)
            
            for series in self.spectrum_chart.series():
                series.attachAxis(axis_x)
                series.attachAxis(axis_y)
            
            self.spectrum_chart.setTitle("Спектральное сравнение")
            self._log_emitter.log_line.emit("Спектр построен успешно")
        else:
            self._log_emitter.log_line.emit("Нет данных для построения спектра")
            QMessageBox.warning(self, "Ошибка", "Не удалось построить спектр - нет данных")

    def _compute_spectrum(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Вычислить спектр сигнала."""
        # FFT
        n = len(signal)
        fft_result = np.fft.rfft(signal)
        spectrum = np.abs(fft_result)
        
        # Нормализация
        spectrum = spectrum / n * 2
        
        return spectrum

    def on_select_all_spectrum(self) -> None:
        """Выбрать все файлы для спектрального сравнения."""
        for row in range(self.spectrum_files_table.rowCount()):
            chk = self.spectrum_files_table.cellWidget(row, 0)
            if chk:
                chk.setChecked(True)
