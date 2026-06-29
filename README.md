# YOLO Helmet Detection

Проект по теме: дообучение и оптимизация YOLO для нового класса объектов.

Модель YOLO11n была дообучена для детекции касок, класс `helmet`.
После обучения модель была экспортирована и сравнена в разных runtime:

- PyTorch
- ONNX Runtime CUDA
- TensorRT FP16
- TensorRT INT8

## Как открыть результаты

На Windows достаточно запустить файл `start.bat` в корне проекта.

Он сам открывает главную страницу:

`demo/runtime_dashboard.html`

На этой странице есть:

- краткий вывод;
- железо, на котором проводились замеры;
- методология замеров;
- таблица benchmark;
- сравнение FP16 и INT8;
- объяснение, почему INT8 не быстрее FP16;
- анализ runtime-графов;
- ссылки на CSV и файлы анализа.

## Что было добавлено для воспроизводимости

Код обучения:

`scripts/train_helmet.py`

Код экспорта и валидации:

`scripts/export_validate.py`

Повторный benchmark:

`scripts/benchmark_runtimes.py`

Сводка benchmark:

`scripts/summarize_benchmark.py`

Сохранение ONNX Runtime optimized graph:

`scripts/save_ort_optimized_graph.py`

Анализ runtime-графов:

`scripts/inspect_runtime_graphs.py`

Генерация HTML-дашборда:

`scripts/build_dashboard.py`

## Основные результаты

Сводная таблица:

`results/benchmark_summary.csv`

Все 5 повторных прогонов:

`results/benchmark_runtime_repeated.csv`

Файлы анализа графов:

`runtime_graphs/`

HTML-дашборд:

`demo/runtime_dashboard.html`

## Окружение замеров

- GPU: NVIDIA GeForce RTX 4070, 12 GB VRAM
- OS: Windows 10 Pro 22H2
- Python: 3.11.7
- PyTorch: 2.12.1+cu126
- CUDA: 12.6
- test split: 1604 images, 4863 helmet objects
- imgsz: 640
- batch: 1

## Методология замеров

Время взято из вывода Ultralytics `model.val()`:

`Speed: preprocess ms, inference ms, postprocess ms per image`

Для сравнения использовался показатель `inference ms`.

Для каждого runtime был сделан warmup, затем 5 повторных прогонов на одном и том же test split.

## Вывод

TensorRT FP16 оказался лучшим практическим вариантом.

TensorRT INT8 уменьшил размер engine, но не дал ускорения относительно FP16 на маленькой YOLO11n при batch=1 и немного снизил качество.

