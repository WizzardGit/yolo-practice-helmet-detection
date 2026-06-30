# Runtime profiling evidence

## 1. то показывал старый анализ графов

Файлы:

- `runtime_graphs/onnx_original_ops.json`
- `runtime_graphs/onnx_ort_optimized_ops.json`

это **structural graph analysis**, то есть структурный анализ графа.

н показывает не время выполнения слоёв, а то, какие операции есть в графе модели:

- в исходном ONNX после экспорта;
- в ONNX-графе после оптимизаций ONNX Runtime.

лавный вывод из этого анализа:

ONNX Runtime действительно меняет граф и применяет graph optimization / fusion.  
апример:

- часть паттернов `Sigmoid` + `Mul` после оптимизации представляется как `QuickGelu`;
- `MatMul` после оптимизации представляется как `FusedMatMul`.

ажно: это не доказывает ускорение конкретного слоя.  
то доказывает, что структура графа изменилась и runtime применил оптимизации.

Фактическое ускорение доказывается benchmark-таблицей:

- `results/benchmark_summary.csv`

## 2. то добавлено после замечания

осле замечания был добавлен ONNX Runtime profiling:

- `scripts/profile_onnx_runtime.py`
- `runtime_graphs/onnx_runtime_profile_summary.json`
- `runtime_graphs/onnx_runtime_profile_sample.json`

Скрипт включает:

- `SessionOptions.enable_profiling = True`
- `GraphOptimizationLevel.ORT_ENABLE_ALL`
- `CUDAExecutionProvider`
- warmup
- повторные inference-итерации на input shape `1x3x640x640`

 отличие от operator-count JSON, этот profiling JSON уже показывает runtime trace: события, реально выполненные узлы/kernel events и их длительности.

## 3. роверка CUDAExecutionProvider

рофилирование было запущено через ONNX Runtime с CUDAExecutionProvider.

 summary зафиксировано:

- `cuda_execution_provider_started: true`
- `active_session_providers: CUDAExecutionProvider, CPUExecutionProvider`
- `onnxruntime_version: 1.26.0`
- `torch: 2.12.1+cu126`
- `torch_cuda: 12.6`
- `torch_cudnn: 91002`

ажный технический момент:

в скрипте сначала импортируется `torch`, чтобы PyTorch загрузил свои CUDA/cuDNN DLL, и только потом импортируется `onnxruntime`.  
ез этого ONNX Runtime на Windows может не поднять CUDA provider из-за проблем с DLL.

## 4. очему raw JSON не закоммичен

Raw ONNX Runtime profile получился большим:

- `runtime_graphs/onnx_runtime_profile_raw.json`

н не коммитится в репозиторий, потому что весит десятки мегабайт.

место него в репозиторий добавлены:

- компактный summary;
- sample первых events из raw profile.

того достаточно, чтобы показать формат JSON-трассы и основные runtime-выводы.

## 5. очему avg_wall_time_ms из profiling не равен benchmark

 `onnx_runtime_profile_summary.json` есть `avg_wall_time_ms`.

то время измерено:

- на random input;
- с включённым profiling;
- в отдельном low-level ONNX Runtime loop.

оэтому это значение нельзя напрямую сравнивать с основной benchmark-таблицей.

сновное сравнение runtime остаётся здесь:

- `results/benchmark_summary.csv`

етодология основного benchmark:

- GPU: NVIDIA GeForce RTX 4070 12 GB
- imgsz: 640
- batch: 1
- warmup + repeated runs
- метрика: `inference ms/image` из Ultralytics validation

## 6. TensorRT profiling

сновной TensorRT benchmark уже есть:

- TensorRT FP16
- TensorRT INT8

езультаты лежат в:

- `results/benchmark_summary.csv`

о глубокий per-layer TensorRT profiling требует внешние NVIDIA tools:

- `trtexec`
- или `trt-engine-explorer`

окально `trtexec.exe` не найден.

то зафиксировано явно в файле:

- `runtime_graphs/tensorrt_trtexec_unavailable.json`

Также добавлен воспроизводимый скрипт:

- `scripts/profile_tensorrt_trtexec.py`

огда в окружении появится `trtexec.exe`, этот скрипт сможет выгрузить:

- TensorRT layer info JSON;
- TensorRT runtime profile JSON;
- TensorRT inference times JSON;
- trtexec log.

## 7. тоговый вывод

Теперь анализ разделён на три уровня:

1. `onnx_original_ops.json` / `onnx_ort_optimized_ops.json`  
   Structural graph analysis: показывает, как меняется структура ONNX-графа после оптимизаций.

2. `onnx_runtime_profile_summary.json` / `onnx_runtime_profile_sample.json`  
   ONNX Runtime profiling: показывает runtime trace с реально выполненными событиями и длительностями.

3. `profile_tensorrt_trtexec.py` / `tensorrt_trtexec_unavailable.json`  
   одготовленный путь для TensorRT per-layer profiling через trtexec. окально не выполнен, потому что TensorRT tools отсутствуют.

лавный практический вывод по runtime остаётся прежним:

TensorRT FP16 — лучший вариант для этой модели и этого окружения, потому что он даёт минимальное inference time при сохранении качества.  
INT8 в этом эксперименте не дал выигрыша относительно FP16 и немного снизил качество.
