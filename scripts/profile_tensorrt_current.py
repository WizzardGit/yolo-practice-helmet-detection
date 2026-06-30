import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import tensorrt as trt


LOGGER = trt.Logger(trt.Logger.INFO)


def trt_major():
    try:
        return int(str(trt.__version__).split(".", 1)[0])
    except Exception:
        return 0


def parse_shape(text):
    return tuple(int(x) for x in text.replace("x", ",").split(","))


def make_network_flags():
    # TensorRT >= 10 uses explicit batch by default.
    # TensorRT 11 may not expose EXPLICIT_BATCH at all.
    if trt_major() >= 10:
        return 0

    flag_enum = getattr(trt, "NetworkDefinitionCreationFlag", None)
    if flag_enum is not None and hasattr(flag_enum, "EXPLICIT_BATCH"):
        return 1 << int(flag_enum.EXPLICIT_BATCH)

    return 0


def set_workspace(config, gb):
    bytes_count = int(gb * 1024 * 1024 * 1024)
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, bytes_count)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = bytes_count


def try_enable_fp16(builder, config):
    # TensorRT 11 can be strongly typed and may not expose the old FP16 builder flag.
    builder_flag = getattr(trt, "BuilderFlag", None)
    fp16_flag = getattr(builder_flag, "FP16", None) if builder_flag is not None else None

    if fp16_flag is None:
        print("[WARN] TensorRT BuilderFlag.FP16 is not available in this TensorRT version.")
        print("[WARN] Building current-version TensorRT engine without explicit FP16 builder flag.")
        return False

    has_fast_fp16 = getattr(builder, "platform_has_fast_fp16", True)
    if not has_fast_fp16:
        print("[WARN] platform_has_fast_fp16 is False. Building without FP16.")
        return False

    config.set_flag(fp16_flag)
    print("[INFO] FP16 builder flag enabled.")
    return True


def build_engine_from_onnx(onnx_path, engine_path, input_shape, request_fp16=True):
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    builder = trt.Builder(LOGGER)
    network = builder.create_network(make_network_flags())
    parser = trt.OnnxParser(network, LOGGER)
    config = builder.create_builder_config()

    set_workspace(config, 4)

    if hasattr(config, "profiling_verbosity") and hasattr(trt, "ProfilingVerbosity"):
        try:
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            print("[INFO] profiling_verbosity = DETAILED")
        except Exception as exc:
            print("[WARN] Failed to set profiling verbosity:", exc)

    fp16_enabled = False
    if request_fp16:
        fp16_enabled = try_enable_fp16(builder, config)

    print("[INFO] TensorRT version:", trt.__version__)
    print("[INFO] Network flags:", make_network_flags())

    data = onnx_path.read_bytes()
    if not parser.parse(data):
        errors = []
        for i in range(parser.num_errors):
            errors.append(str(parser.get_error(i)))
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    print("[INFO] ONNX parsed.")
    print("[INFO] network inputs:", network.num_inputs)
    print("[INFO] network outputs:", network.num_outputs)

    need_profile = False
    profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = tuple(inp.shape)
        print("[INFO] input", i, inp.name, shape)

        if any(dim < 0 for dim in shape):
            need_profile = True
            profile.set_shape(inp.name, min=input_shape, opt=input_shape, max=input_shape)
            print("[INFO] dynamic input profile set:", inp.name, input_shape)

    if need_profile:
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")

    engine_path.write_bytes(bytes(serialized))
    print("[OK] engine saved:", engine_path)

    return engine_path, fp16_enabled


def load_engine(engine_path):
    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize freshly built engine: {engine_path}")
    return runtime, engine


def trt_dtype_to_torch(dtype):
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.int8:
        return torch.int8
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.bool:
        return torch.bool
    return torch.float32


class CollectingProfiler(trt.IProfiler):
    def __init__(self):
        try:
            trt.IProfiler.__init__(self)
        except Exception:
            pass
        self.records = []

    def report_layer_time(self, layer_name, ms):
        self.records.append({"layer": str(layer_name), "time_ms": float(ms)})


def prepare_context_and_tensors(engine, input_shape):
    context = engine.create_execution_context()
    tensors = []
    bindings = []

    if hasattr(engine, "num_io_tensors"):
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            is_input = mode == trt.TensorIOMode.INPUT
            dtype = engine.get_tensor_dtype(name)
            shape = tuple(engine.get_tensor_shape(name))

            if is_input and any(dim < 0 for dim in shape):
                context.set_input_shape(name, input_shape)

            try:
                final_shape = tuple(context.get_tensor_shape(name))
            except Exception:
                final_shape = shape

            if any(dim < 0 for dim in final_shape):
                if is_input:
                    final_shape = input_shape
                else:
                    if hasattr(context, "infer_shapes"):
                        try:
                            context.infer_shapes()
                            final_shape = tuple(context.get_tensor_shape(name))
                        except Exception:
                            pass

            if any(dim < 0 for dim in final_shape):
                raise RuntimeError(f"Unresolved dynamic shape for tensor {name}: {final_shape}")

            torch_dtype = trt_dtype_to_torch(dtype)

            if is_input:
                tensor = torch.randn(*final_shape, device="cuda", dtype=torch.float32)
                if torch_dtype != torch.float32:
                    tensor = tensor.to(torch_dtype)
            else:
                tensor = torch.empty(*final_shape, device="cuda", dtype=torch_dtype)

            context.set_tensor_address(name, int(tensor.data_ptr()))

            tensors.append({
                "name": name,
                "mode": "input" if is_input else "output",
                "shape": list(final_shape),
                "dtype": str(dtype),
                "torch_dtype": str(torch_dtype)
            })

        return context, tensors, None

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        is_input = engine.binding_is_input(i)
        dtype = engine.get_binding_dtype(i)
        shape = tuple(engine.get_binding_shape(i))

        if is_input and any(dim < 0 for dim in shape):
            context.set_binding_shape(i, input_shape)

        final_shape = tuple(context.get_binding_shape(i))
        if any(dim < 0 for dim in final_shape):
            raise RuntimeError(f"Unresolved dynamic shape for binding {name}: {final_shape}")

        torch_dtype = trt_dtype_to_torch(dtype)

        if is_input:
            tensor = torch.randn(*final_shape, device="cuda", dtype=torch.float32)
            if torch_dtype != torch.float32:
                tensor = tensor.to(torch_dtype)
        else:
            tensor = torch.empty(*final_shape, device="cuda", dtype=torch_dtype)

        bindings.append(int(tensor.data_ptr()))
        tensors.append({
            "name": name,
            "mode": "input" if is_input else "output",
            "shape": list(final_shape),
            "dtype": str(dtype),
            "torch_dtype": str(torch_dtype)
        })

    return context, tensors, bindings


def run_inference(context, bindings=None):
    stream = torch.cuda.current_stream().cuda_stream

    if hasattr(context, "execute_async_v3"):
        ok = context.execute_async_v3(stream_handle=stream)
    elif hasattr(context, "execute_async_v2"):
        ok = context.execute_async_v2(bindings=bindings, stream_handle=stream)
    elif hasattr(context, "execute_v2"):
        ok = context.execute_v2(bindings)
    else:
        raise RuntimeError("No supported TensorRT execute method found")

    if not ok:
        raise RuntimeError("TensorRT execution returned False")


def inspect_engine(engine, context):
    result = {"format": None, "data": None, "error": None}

    try:
        inspector = engine.create_engine_inspector()

        try:
            inspector.context = context
        except Exception:
            pass

        try:
            inspector.execution_context = context
        except Exception:
            pass

        try:
            text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            result["format"] = "JSON"
            try:
                result["data"] = json.loads(text)
            except Exception:
                result["data"] = text
            return result
        except Exception:
            text = inspector.get_engine_information(trt.LayerInformationFormat.ONELINE)
            result["format"] = "ONELINE"
            result["data"] = text
            return result

    except Exception as exc:
        result["error"] = str(exc)
        return result


def aggregate(records):
    totals = defaultdict(float)
    counts = defaultdict(int)

    for r in records:
        layer = r["layer"]
        totals[layer] += float(r["time_ms"])
        counts[layer] += 1

    rows = []
    for layer, total in totals.items():
        calls = counts[layer]
        rows.append({
            "layer": layer,
            "total_time_ms": total,
            "calls": calls,
            "avg_time_ms": total / max(calls, 1)
        })

    rows.sort(key=lambda x: x["total_time_ms"], reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="results/models/helmet_yolo11n.onnx")
    ap.add_argument("--engine", default="results/models/helmet_yolo11n_fp16_current.engine")
    ap.add_argument("--prefix", default="tensorrt_fp16_current")
    ap.add_argument("--out-dir", default="runtime_graphs")
    ap.add_argument("--shape", default="1x3x640x640")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_shape = parse_shape(args.shape)

    print("[INFO] torch:", torch.__version__)
    print("[INFO] cuda available:", torch.cuda.is_available())
    print("[INFO] gpu:", torch.cuda.get_device_name(0))
    print("[INFO] TensorRT:", trt.__version__)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in torch")

    engine_path, fp16_enabled = build_engine_from_onnx(
        onnx_path=args.onnx,
        engine_path=args.engine,
        input_shape=input_shape,
        request_fp16=True,
    )

    runtime, engine = load_engine(engine_path)
    context, tensors, bindings = prepare_context_and_tensors(engine, input_shape)

    profiler = CollectingProfiler()
    profiler_attached = True

    try:
        context.profiler = profiler
    except Exception as exc:
        profiler_attached = False
        print("[WARN] profiler attach failed:", exc)

    if hasattr(context, "enqueue_emits_profile"):
        try:
            context.enqueue_emits_profile = True
        except Exception:
            pass

    for _ in range(args.warmup):
        run_inference(context, bindings)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iterations):
        run_inference(context, bindings)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    if hasattr(context, "report_to_profiler"):
        try:
            context.report_to_profiler()
        except Exception:
            pass

    engine_info = inspect_engine(engine, context)
    top_layers = aggregate(profiler.records)

    total_ms = (t1 - t0) * 1000.0
    avg_ms = total_ms / args.iterations

    precision_note = "FP16 builder flag enabled" if fp16_enabled else "current TensorRT build; explicit FP16 builder flag unavailable or disabled"

    layer_info = {
        "engine_path": str(engine_path),
        "onnx_path": args.onnx,
        "tensorrt_version": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "precision_note": precision_note,
        "tensors": tensors,
        "engine_inspector": engine_info
    }

    profile = {
        "engine_path": str(engine_path),
        "profiler_attached": profiler_attached,
        "raw_records_count": len(profiler.records),
        "raw_records_sample": profiler.records[:300],
        "top_layers": top_layers[:100]
    }

    times = {
        "engine_path": str(engine_path),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "total_wall_time_ms": total_ms,
        "avg_wall_time_ms": avg_ms
    }

    summary = {
        "analysis_type": "TensorRT current-version build + runtime profiling",
        "onnx_path": args.onnx,
        "engine_path": str(engine_path),
        "tensorrt_version": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "precision_note": precision_note,
        "profiler_attached": profiler_attached,
        "raw_layer_record_count": len(profiler.records),
        "unique_profiled_layers": len(top_layers),
        "avg_wall_time_ms": avg_ms,
        "top_layers": top_layers[:30],
        "important_note": (
            "The old serialized TensorRT engine was incompatible with the current TensorRT runtime. "
            "This script rebuilds the engine from ONNX using the currently installed TensorRT version "
            "and then profiles the freshly built engine. TensorRT 10/11 use explicit batch by default, "
            "so EXPLICIT_BATCH flag is not required."
        )
    }

    paths = {
        "layer_info": out_dir / f"{args.prefix}_layer_info.json",
        "profile": out_dir / f"{args.prefix}_profile.json",
        "times": out_dir / f"{args.prefix}_times.json",
        "summary": out_dir / f"{args.prefix}_runtime_summary.json",
        "csv": out_dir / f"{args.prefix}_top_layers.csv",
        "log": out_dir / f"{args.prefix}_python_profile.log",
    }

    paths["layer_info"].write_text(json.dumps(layer_info, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["profile"].write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["times"].write_text(json.dumps(times, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with paths["csv"].open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "layer", "total_time_ms", "calls", "avg_time_ms"])
        w.writeheader()
        for i, row in enumerate(top_layers[:100], 1):
            w.writerow({
                "rank": i,
                "layer": row["layer"],
                "total_time_ms": row["total_time_ms"],
                "calls": row["calls"],
                "avg_time_ms": row["avg_time_ms"],
            })

    paths["log"].write_text(
        "\n".join([
            f"onnx={args.onnx}",
            f"engine={engine_path}",
            f"tensorrt_version={trt.__version__}",
            f"gpu={torch.cuda.get_device_name(0)}",
            f"precision_note={precision_note}",
            f"profiler_attached={profiler_attached}",
            f"raw_layer_record_count={len(profiler.records)}",
            f"unique_profiled_layers={len(top_layers)}",
            f"avg_wall_time_ms={avg_ms}",
        ]),
        encoding="utf-8"
    )

    for name, path in paths.items():
        print(f"[OK] {name}: {path}")

    print("[OK] fresh engine:", engine_path)
    print("[OK] precision note:", precision_note)
    print("[OK] avg wall ms:", avg_ms)
    print("[OK] profiler records:", len(profiler.records))
    print("[OK] unique layers:", len(top_layers))

    if len(top_layers) == 0:
        print("[WARN] TensorRT profiler did not return per-layer records. Engine inspector and timing artifacts were still written.")


if __name__ == "__main__":
    main()
