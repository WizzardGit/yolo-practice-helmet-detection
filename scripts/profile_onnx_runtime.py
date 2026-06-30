import argparse
import json
import os
import pathlib
import shutil
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

# IMPORTANT: import torch before onnxruntime so PyTorch CUDA/cuDNN DLLs are loaded first.
try:
    import torch

    TORCH_VERSION = torch.__version__
    TORCH_CUDA = torch.version.cuda
    TORCH_CUDNN = torch.backends.cudnn.version()

    torch_lib = pathlib.Path(torch.__file__).resolve().parent / "lib"
    if torch_lib.exists():
        os.add_dll_directory(str(torch_lib))
except Exception as exc:
    torch = None
    TORCH_VERSION = None
    TORCH_CUDA = None
    TORCH_CUDNN = None
    print(f"[WARN] torch preload failed: {exc}")

import onnxruntime as ort


def parse_shape(shape_text: str):
    return tuple(int(x) for x in shape_text.lower().replace(",", "x").split("x"))


def load_profile_events(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]

    if isinstance(data, list):
        return data

    return []


def summarize_events(events, top_k=30):
    node_events = []
    op_totals = defaultdict(float)
    provider_totals = defaultdict(float)

    for e in events:
        if not isinstance(e, dict):
            continue

        name = str(e.get("name", ""))
        cat = str(e.get("cat", ""))
        dur_us = float(e.get("dur", 0) or 0)
        args = e.get("args", {}) or {}

        # ONNX Runtime profiling usually marks executed nodes as cat == "Node"
        # and kernel timing events often contain "kernel_time" in the name.
        is_node_event = cat == "Node" or "kernel_time" in name

        if not is_node_event or dur_us <= 0:
            continue

        op_name = args.get("op_name") or args.get("op") or "unknown"
        provider = args.get("provider") or args.get("execution_provider") or "unknown"
        node_name = args.get("node_name") or args.get("node_index") or name

        dur_ms = dur_us / 1000.0

        item = {
            "name": name,
            "node_name": str(node_name),
            "op_name": str(op_name),
            "provider": str(provider),
            "duration_ms": dur_ms,
        }

        node_events.append(item)
        op_totals[str(op_name)] += dur_ms
        provider_totals[str(provider)] += dur_ms

    top_nodes = sorted(node_events, key=lambda x: x["duration_ms"], reverse=True)[:top_k]

    top_ops = [
        {"op_name": op, "total_duration_ms": dur}
        for op, dur in sorted(op_totals.items(), key=lambda x: x[1], reverse=True)[:top_k]
    ]

    providers = [
        {"provider": provider, "total_duration_ms": dur}
        for provider, dur in sorted(provider_totals.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        "node_event_count": len(node_events),
        "top_nodes_by_duration_ms": top_nodes,
        "top_ops_by_total_duration_ms": top_ops,
        "providers_by_total_duration_ms": providers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="results/models/helmet_yolo11n.onnx")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--shape", default="1x3x640x640")
    parser.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--out-dir", default="runtime_graphs")
    parser.add_argument("--prefix", default="onnx_runtime_profile")
    args = parser.parse_args()

    model_path = pathlib.Path(args.model)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"{args.prefix}_raw.json"
    summary_path = out_dir / f"{args.prefix}_summary.json"
    sample_path = out_dir / f"{args.prefix}_sample.json"

    shape = parse_shape(args.shape)
    x = np.random.randn(*shape).astype(np.float32)

    print(f"[INFO] torch: {TORCH_VERSION}")
    print(f"[INFO] torch cuda: {TORCH_CUDA}")
    print(f"[INFO] torch cudnn: {TORCH_CUDNN}")
    print(f"[INFO] onnxruntime: {ort.__version__}")
    print(f"[INFO] available providers: {ort.get_available_providers()}")

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if args.provider == "cuda":
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "cudnn_conv_use_max_workspace": "1",
            }),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=providers,
    )

    active_providers = sess.get_providers()
    print(f"[INFO] session providers: {active_providers}")

    cuda_started = "CUDAExecutionProvider" in active_providers
    if args.require_cuda and not cuda_started:
        raise RuntimeError("CUDAExecutionProvider did not start. Refusing to create CPU fallback profile.")

    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    print(f"[INFO] input name: {input_name}")
    print(f"[INFO] input shape from model: {input_shape}")
    print(f"[INFO] profiling shape: {shape}")

    for _ in range(args.warmup):
        sess.run(None, {input_name: x})

    start = time.perf_counter()
    for _ in range(args.iterations):
        sess.run(None, {input_name: x})
    end = time.perf_counter()

    profile_file = pathlib.Path(sess.end_profiling())

    if raw_path.exists():
        raw_path.unlink()

    shutil.move(str(profile_file), str(raw_path))

    events = load_profile_events(raw_path)
    event_summary = summarize_events(events)

    total_wall_time_sec = end - start
    avg_wall_time_ms = total_wall_time_sec * 1000.0 / args.iterations

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "profile_type": "ONNX Runtime JSON trace",
        "analysis_level": "runtime profiling, not just static operator count",
        "iterations": args.iterations,
        "warmup": args.warmup,
        "input_name": input_name,
        "input_shape_from_model": [str(x) for x in input_shape],
        "profiling_shape": list(shape),
        "total_wall_time_sec": total_wall_time_sec,
        "avg_wall_time_ms": avg_wall_time_ms,
        "onnxruntime_version": ort.__version__,
        "torch_version": TORCH_VERSION,
        "torch_cuda": TORCH_CUDA,
        "torch_cudnn": TORCH_CUDNN,
        "available_providers": ort.get_available_providers(),
        "active_session_providers": active_providers,
        "cuda_execution_provider_started": cuda_started,
        "important_note": (
            "This file is runtime profiling evidence. It complements the structural ONNX "
            "operator-count files. The wall time here is measured with profiling enabled "
            "on random input and should not replace the repeated validation benchmark in "
            "results/benchmark_summary.csv."
        ),
        **event_summary,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    sample = {
        "created_at": summary["created_at"],
        "source_raw_profile": str(raw_path),
        "note": "Small sample of raw ONNX Runtime profiling events for quick review.",
        "events": events[:50],
    }

    with sample_path.open("w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    print(f"[OK] raw profile: {raw_path}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] sample: {sample_path}")
    print(f"[OK] avg wall time with profiling enabled: {avg_wall_time_ms:.4f} ms")
    print(f"[OK] CUDAExecutionProvider started: {cuda_started}")


if __name__ == "__main__":
    main()
