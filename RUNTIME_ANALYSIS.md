# Анализ runtime и графов YOLO

## Окружение

Измерения проводились на локальной машине:

- GPU: NVIDIA GeForce RTX 4070, 12282 MiB VRAM
- OS: Windows 10 10.0.19045
- Python: 3.11.7
- PyTorch: 2.12.1+cu126
- CUDA: 12.6
- ONNX Runtime: 1.22.0
- TensorRT: использовался через Ultralytics/TensorRT engine

## Методология замеров

Для сравнения использовался один и тот же test split датасета:

- 1604 изображения
- 4863 объекта класса `helmet`
- `imgsz=640`
- `batch=1`

Для каждого runtime выполнялся один warmup-прогон, затем 5 повторов валидации.

Время в таблице — это `inference ms` из вывода Ultralytics `model.val()`:

```text
Speed: X ms preprocess, Y ms inference, 0.0 ms loss, Z ms postprocess per image
```

В итоговую таблицу заносится именно `Y ms inference`, то есть время прямого прохода модели на одно изображение. Это не wall-clock время всей валидации и не включает чтение датасета, подсчёт mAP и сохранение результатов.

Исходные повторные замеры сохраняются в файл:

```text
results/benchmark_runtime_repeated.csv
```

Сводная таблица сохраняется в файл:

```text
results/benchmark_summary.csv
```

## Результаты повторных замеров

| Runtime | Inference ms, повторы | Среднее | Медиана | Качество |
|---|---:|---:|---:|---|
| PyTorch | 6.4 / 6.9 / 4.1 / 3.7 / 3.8 | 4.98 ms | 4.10 ms | mAP50-95 = 0.678 |
| ONNX Runtime CUDA | 3.3 / 3.2 / 2.9 / 2.9 / 2.9 | 3.04 ms | 2.90 ms | mAP50-95 = 0.679 |
| TensorRT FP16 | 1.5 / 1.5 / 1.5 / 1.6 / 1.5 | 1.52 ms | 1.50 ms | mAP50-95 = 0.679 |
| TensorRT INT8 | 1.5 / 1.5 / 1.5 / 1.5 / 1.5 | 1.50 ms | 1.50 ms | mAP50-95 = 0.663 |

## Вывод по INT8 и FP16

Изначальный вывод, что INT8 работает дольше FP16, был некорректным по новым повторным замерам.

По повторной проверке TensorRT INT8 не стал заметно быстрее FP16:

- TensorRT FP16: около 1.5 ms inference на изображение;
- TensorRT INT8: около 1.5 ms inference на изображение.

При этом INT8 engine меньше:

- TensorRT FP16 engine: около 7 MiB;
- TensorRT INT8 engine: около 4 MiB.

Но качество у INT8 ниже:

- FP16: mAP50 = 0.981, mAP50-95 = 0.679;
- INT8: mAP50 = 0.978, mAP50-95 = 0.663.

Поэтому для этой модели и этого режима запуска лучший вариант — TensorRT FP16: он даёт такую же или практически такую же скорость, но сохраняет качество лучше, чем INT8.

Почему INT8 не дал ускорения:

- модель YOLO11n очень маленькая;
- запуск выполнялся с `batch=1`;
- на RTX 4070 FP16 операции уже очень быстрые;
- выигрыш от INT8 на такой маленькой модели может быть съеден накладными расходами TensorRT engine, quantize/dequantize и постобработкой;
- INT8 чаще полезнее на более крупных моделях, больших batch size или при ограничении памяти.

## Анализ графов исполнения

Для анализа графов используются два уровня:

1. ONNX / ONNX Runtime:
   - исходный ONNX-граф;
   - optimized graph, сохранённый ONNX Runtime после применения оптимизаций.

2. TensorRT:
   - layer dump / engine inspector для FP16 engine;
   - layer dump / engine inspector для INT8 engine.

Скрипты:

```text
scripts/save_ort_optimized_graph.py
scripts/inspect_runtime_graphs.py
```

После запуска появляются файлы:

```text
runtime_graphs/onnx_original_ops.txt
runtime_graphs/onnx_ort_optimized_ops.txt
runtime_graphs/tensorrt_fp16_layers.json
runtime_graphs/tensorrt_int8_layers.json
```

Эти файлы показывают, как модель представлена и оптимизирована в разных runtime. ONNX-файлы можно дополнительно открыть в Netron.
