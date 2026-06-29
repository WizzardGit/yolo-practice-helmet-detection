from pathlib import Path
from collections import Counter
import json
import traceback


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "results" / "models"
OUT = ROOT / "runtime_graphs"


def inspect_onnx(model_path, name):
    import onnx

    model_path = Path(model_path)
    model = onnx.load(str(model_path))

    ops = Counter(node.op_type for node in model.graph.node)

    txt = OUT / f"{name}_ops.txt"
    js = OUT / f"{name}_ops.json"

    lines = [
        f"model: {model_path}",
        f"file_size_mb: {model_path.stat().st_size / 1024 / 1024:.2f}",
        f"nodes: {len(model.graph.node)}",
        f"inputs: {len(model.graph.input)}",
        f"outputs: {len(model.graph.output)}",
        "",
        "operators:",
    ]

    for op, count in ops.most_common():
        lines.append(f"{op}: {count}")

    txt.write_text("\n".join(lines), encoding="utf-8")
    js.write_text(json.dumps(dict(ops), indent=2), encoding="utf-8")


def inspect_tensorrt(engine_path, name):
    engine_path = Path(engine_path)

    info_path = OUT / f"{name}_engine_info.txt"
    error_path = OUT / f"{name}_inspector_error.txt"

    info_lines = [
        f"model: {engine_path}",
        f"file_size_mb: {engine_path.stat().st_size / 1024 / 1024:.2f}",
        "",
        "note:",
        "TensorRT .engine files are environment-specific.",
        "They depend on TensorRT version, CUDA version, GPU architecture and driver.",
        "The engine was successfully executed during Ultralytics benchmark.",
        "If TensorRT EngineInspector cannot deserialize it separately, the error is saved below.",
    ]

    info_path.write_text("\n".join(info_lines), encoding="utf-8")

    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        serialized_engine = engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(serialized_engine)

        if engine is None:
            raise RuntimeError("TensorRT returned None while deserializing engine.")

        inspector = engine.create_engine_inspector()

        try:
            layer_info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            layer_path = OUT / f"{name}_layers.json"
        except Exception:
            layer_info = inspector.get_engine_information(trt.LayerInformationFormat.ONELINE)
            layer_path = OUT / f"{name}_layers.txt"

        layer_path.write_text(layer_info, encoding="utf-8")

        if error_path.exists():
            error_path.unlink()

    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    original_onnx = MODELS / "helmet_yolo11n.onnx"
    optimized_onnx = MODELS / "helmet_yolo11n_ort_optimized.onnx"
    fp16_engine = MODELS / "helmet_yolo11n_fp16.engine"
    int8_engine = MODELS / "helmet_yolo11n_int8.engine"

    if original_onnx.exists():
        inspect_onnx(original_onnx, "onnx_original")

    if optimized_onnx.exists():
        inspect_onnx(optimized_onnx, "onnx_ort_optimized")

    if fp16_engine.exists():
        inspect_tensorrt(fp16_engine, "tensorrt_fp16")

    if int8_engine.exists():
        inspect_tensorrt(int8_engine, "tensorrt_int8")

    print("Saved runtime graph analysis to:", OUT)


if __name__ == "__main__":
    main()
