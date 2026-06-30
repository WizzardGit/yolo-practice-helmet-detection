# Runtime profiling evidence

## 1. Что показывает анализ графовых оптимизаций

В проекте есть два уровня анализа ONNX-графа:

* `runtime_graphs/onnx_original_ops.json`
* `runtime_graphs/onnx_ort_optimized_ops.json`

Это structural graph analysis, то есть структурный анализ графа.

Он отвечает на вопрос: какие операции есть в модели до и после оптимизаций ONNX Runtime. Это не замер времени выполнения слоёв, а анализ структуры вычислительного графа.

Исходный ONNX-граф показывает операции после экспорта модели. Оптимизированный ONNX-граф показывает, как ONNX Runtime переписал граф при `GraphOptimizationLevel.ORT_ENABLE_ALL`.

## 2. Выводы из structural graph analysis

Главный вывод: ONNX Runtime действительно меняет граф и применяет graph optimization / fusion.

Примеры:

* часть паттернов `Sigmoid` + `Mul` после оптимизации представляется как `QuickGelu`;
* `MatMul` после оптимизации представляется как `FusedMatMul`.

Что это означает:

* runtime не всегда исполняет граф ровно в том виде, в каком он был экспортирован из PyTorch;
* часть операций может быть заменена или объединена;
* structural-анализ помогает увидеть факт graph fusion / graph rewrite.

Ограничение:

* этот анализ не показывает layer latency;
* сам факт появления `QuickGelu` или `FusedMatMul` не доказывает ускорение конкретного слоя;
* фактическое ускорение доказывается benchmark-таблицей `results/benchmark_summary.csv`.

## 3. ONNX Runtime profiling

После замечания был добавлен ONNX Runtime profiling:

* `scripts/profile_onnx_runtime.py`
* `runtime_graphs/onnx_runtime_profile_summary.json`
* `runtime_graphs/onnx_runtime_profile_sample.json`

Скрипт включает:

* `SessionOptions.enable_profiling = True`
* `GraphOptimizationLevel.ORT_ENABLE_ALL`
* `CUDAExecutionProvider`
* warmup
* 100 inference-итераций
* input shape `1x3x640x640`

В отличие от operator-count JSON, этот profiling JSON уже показывает runtime trace: реально выполненные node/kernel events и их длительности.

## 4. Проверка CUDAExecutionProvider

Профилирование ONNX Runtime было запущено через CUDAExecutionProvider.

В `onnx_runtime_profile_summary.json` зафиксировано:

* `cuda_execution_provider_started: true`
* `active_session_providers: CUDAExecutionProvider, CPUExecutionProvider`
* `avg_wall_time_ms: 6.849980000406504`
* `node_event_count: 24990`

Технический момент:

в `scripts/profile_onnx_runtime.py` сначала импортируется `torch`, чтобы PyTorch загрузил свои CUDA/cuDNN DLL, и только потом импортируется `onnxruntime`.

Это важно для Windows-окружения, потому что без подгруженных CUDA/cuDNN DLL ONNX Runtime может не поднять CUDA provider.

## 5. Почему raw ONNX profile не закоммичен

Raw ONNX Runtime profile получается большим и не коммитится в репозиторий.

Вместо него добавлены:

* компактный summary;
* sample первых profiling events.

Этого достаточно, чтобы показать формат JSON-трассы и основные runtime-выводы, не раздувая GitHub-репозиторий.

## 6. Почему avg_wall_time_ms из ONNX profiling не равен benchmark

В `onnx_runtime_profile_summary.json` есть `avg_wall_time_ms`.

Это значение измерено:

* на random input;
* с включённым profiling;
* в отдельном low-level ONNX Runtime loop.

Поэтому это значение не заменяет основной benchmark.

Основное сравнение runtime остаётся в:

* `results/benchmark_summary.csv`

Методология основного benchmark:

* GPU: NVIDIA GeForce RTX 4070 12 GB
* imgsz: 640
* batch: 1
* warmup + repeated runs
* метрика: `inference ms/image` из Ultralytics validation

## 7. TensorRT runtime profiling

Изначально планировалось использовать `trtexec` / `trt-engine-explorer`, но локально `trtexec` не был доступен в PATH.

После этого был добавлен альтернативный runtime-level profiling через TensorRT Python API:

* `scripts/profile_tensorrt_current.py`
* `runtime_graphs/tensorrt_fp16_current_layer_info.json`
* `runtime_graphs/tensorrt_fp16_current_profile.json`
* `runtime_graphs/tensorrt_fp16_current_times.json`
* `runtime_graphs/tensorrt_fp16_current_runtime_summary.json`
* `runtime_graphs/tensorrt_fp16_current_top_layers.csv`
* `runtime_graphs/tensorrt_fp16_current_python_profile.log`

Старый serialized TensorRT engine оказался несовместим с текущей версией TensorRT. Поэтому engine был пересобран заново из ONNX текущим TensorRT `11.1.0.106` и затем профилирован.

В `tensorrt_fp16_current_runtime_summary.json` зафиксировано:

* `analysis_type: TensorRT current-version build + runtime profiling`
* `tensorrt_version: 11.1.0.106`
* `gpu: NVIDIA GeForce RTX 4070`
* `profiler_attached: true`
* `raw_layer_record_count: 35090`
* `unique_profiled_layers: 319`
* `avg_wall_time_ms: 3.603655000915751`

Это уже runtime-level evidence для TensorRT: есть информация по engine, profile, times и top layers.

## 8. Что показывает TensorRT profile

TensorRT runtime analysis показывает:

* какой engine был построен из ONNX;
* какие tensors есть у engine;
* какую информацию о слоях возвращает Engine Inspector;
* какие слои / fused layers профилируются во время inference;
* какие слои занимают больше всего времени;
* среднее wall-time в отдельном TensorRT inference loop.

Это отличается от ONNX operator-count analysis:

* ONNX operator-count показывает структуру ONNX-графа;
* ONNX Runtime profile показывает runtime trace ONNX Runtime;
* TensorRT runtime profile показывает уже serialized TensorRT engine, то есть граф, который исполняется TensorRT после своих оптимизаций.

## 9. Benchmark summary

Основной benchmark лежит в:

* `results/benchmark_summary.csv`

Результаты:

* PyTorch: `4.973 ms/image`
* ONNX Runtime CUDA: `3.027 ms/image`
* TensorRT FP16: `1.485 ms/image`
* TensorRT INT8: `1.526 ms/image`

Вывод:

TensorRT FP16 — лучший практический вариант для этой модели и этого окружения. Он даёт минимальное inference time при сохранении качества.

INT8 в этом эксперименте не дал выигрыша относительно FP16 и немного снизил качество. Возможная причина: модель маленькая, batch = 1, RTX 4070 хорошо ускоряет FP16, а INT8 overhead/calibration/quantization effects не дают преимущества на этой конкретной задаче.

## 10. Итог

Теперь анализ разделён на три уровня:

1. `onnx_original_ops.json` / `onnx_ort_optimized_ops.json`
   Structural graph analysis: показывает, как меняется структура ONNX-графа после оптимизаций.

2. `onnx_runtime_profile_summary.json` / `onnx_runtime_profile_sample.json`
   ONNX Runtime profiling: показывает runtime trace с реально выполненными событиями и длительностями.

3. `tensorrt_fp16_current_*` artifacts
   TensorRT runtime profiling: показывает runtime-level анализ свежесобранного TensorRT engine.

Таким образом, по helmet-модели закрыты:

* код обучения;
* воспроизводимый pipeline;
* методология benchmark;
* GPU и условия замеров;
* объяснение FP16 vs INT8;
* ONNX graph optimization analysis;
* ONNX Runtime JSON profiling;
* TensorRT runtime profiling.
