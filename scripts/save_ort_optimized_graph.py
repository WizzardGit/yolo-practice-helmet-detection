from pathlib import Path
import argparse
import os


def add_torch_dll_path():
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.exists():
            os.add_dll_directory(str(torch_lib))
            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def parse_args():
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(root / "results" / "models" / "helmet_yolo11n.onnx"))
    parser.add_argument("--output", default=str(root / "results" / "models" / "helmet_yolo11n_ort_optimized.onnx"))
    return parser.parse_args()


def main():
    add_torch_dll_path()

    import onnxruntime as ort

    args = parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.optimized_model_filepath = str(output)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(
        args.input,
        sess_options=session_options,
        providers=providers,
    )

    provider_info = output.with_suffix(".providers.txt")
    provider_info.write_text(
        "\n".join(session.get_providers()),
        encoding="utf-8",
    )

    print("Input:", args.input)
    print("Output:", output)
    print("Providers:", session.get_providers())


if __name__ == "__main__":
    main()
