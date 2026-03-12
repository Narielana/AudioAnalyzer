"""
src — пакет исходного кода AudioAnalyzer.

Структура пакета:
================

processing/ — модуль обработки аудио:
- audio_ops.py: пайплайны преобразований (FWHT, FFT, DCT, DWT, Huffman, Rosenbrock)
- fwht.py: быстрое преобразование Уолша-Адамара
- metrics.py: метрики качества (SNR, LSD, RMSE, SI-SDR, Spectral Convergence, etc.)
- codecs.py: кодирование/декодирование аудио через FFmpeg
- utils.py: вспомогательные функции

ui_new/ — модуль пользовательского интерфейса:
- main_window.py: главное окно приложения
- worker.py: фоновая обработка в отдельном потоке
- constants.py: константы UI (варианты методов, метрики, заголовки)
- presets.py: предустановленные конфигурации параметров
- export_xlsx.py: экспорт результатов в Excel
- log_handler.py: перенаправление логов в UI

utils/ — утилиты приложения:
- logging_setup.py: конфигурация логирования (JSON format)

config.py — централизованная конфигурация параметров

app.py — точка входа приложения
"""