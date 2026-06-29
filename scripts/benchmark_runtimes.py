import csv
import platform
import time
from pathlib import Path

import torch
from ultralytics import YOLO

DATA = r"D:\YOLOPractice\datasets\hardhat_helmet_only\data.yaml"

MODELS = [
    ("PyTorch", r"D:\YOLOPractice\runs\train_helmet\weights\best.pt"),
    ("ONNX Runtime CUDA", r"D:\YOLOPractice\results\models\helmet_yolo11n.onnx"),
    ("TensorRT FP16", r"D:\YOLOPractice\results\models\helmet_yolo11n_fp16.engine"),
    ("TensorRT INT8", r"D:\YOLOPractice\results\models\helmet_yolo11n_int8.engine"),
]

OUT = Path(r"D:\YOLOPractice\results\benchmark_runtime_repeated.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

REPEATS = 5
IMGSZ = 640
BATCH = 1
DEVICE = 0

def get_metric(metrics, key, default=""):
    try:
        return metrics.results_dict.get(key, default)
    except Exception:
        return default

def main():
    rows = []

    print("Hardware:")
    print("OS:", platform.platform())
    print("Python:", platform.python_version())
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)

    for runtime, model_path in MODELS:
        print()
        print("=" * 80)
        print(runtime)
        print(model_path)

        model = YOLO(model_path)

        # warmup
        print("Warmup...")
        model.val(data=DATA, imgsz=IMGSZ, batch=BATCH, device=DEVICE, verbose=False)

        for i in range(1, REPEATS + 1):
            print(f"Repeat {i}/{REPEATS}")

            t0 = time.perf_counter()
            metrics = model.val(
                data=DATA,
                imgsz=IMGSZ,
                batch=BATCH,
                device=DEVICE,
                verbose=False,
                project=r"D:\YOLOPractice\results\benchmark_runs",
                name=f"{runtime.replace(' ', '_').replace('/', '_')}_{i}",
            )
            total_s = time.perf_counter() - t0

            speed = getattr(metrics, "speed", {}) or {}

            inference_ms = speed.get("inference", "")
            fps = ""
            if isinstance(inference_ms, (int, float)) and inference_ms > 0:
                fps = 1000.0 / inference_ms

            rows.append({
                "runtime": runtime,
                "repeat": i,
                "model": model_path,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "imgsz": IMGSZ,
                "batch": BATCH,
                "total_seconds": round(total_s, 4),
                "preprocess_ms": speed.get("preprocess", ""),
                "inference_ms": inference_ms,
                "postprocess_ms": speed.get("postprocess", ""),
                "fps_from_inference_ms": round(fps, 2) if fps != "" else "",
                "precision": get_metric(metrics, "metrics/precision(B)"),
                "recall": get_metric(metrics, "metrics/recall(B)"),
                "mAP50": get_metric(metrics, "metrics/mAP50(B)"),
                "mAP50-95": get_metric(metrics, "metrics/mAP50-95(B)"),
            })

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
