from pathlib import Path
import argparse
import shutil
from ultralytics import YOLO


def parse_args():
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(root / "datasets" / "hardhat_helmet_only" / "data.yaml"))
    parser.add_argument("--weights", default=str(root / "runs" / "train_helmet" / "weights" / "best.pt"))
    parser.add_argument("--models-dir", default=str(root / "results" / "models"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def copy_exported(path, target):
    path = Path(path)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return target


def validate(model_path, data, imgsz, device, name):
    root = Path(__file__).resolve().parents[1]

    model = YOLO(str(model_path), task="detect")
    return model.val(
        data=data,
        imgsz=imgsz,
        batch=1,
        device=device,
        project=str(root / "results"),
        name=name,
    )


def main():
    args = parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    base = YOLO(args.weights, task="detect")

    validate(args.weights, args.data, args.imgsz, args.device, "val_pytorch")

    onnx_path = base.export(format="onnx", imgsz=args.imgsz, simplify=True, device=args.device)
    onnx_target = copy_exported(onnx_path, models_dir / "helmet_yolo11n.onnx")
    validate(onnx_target, args.data, args.imgsz, args.device, "val_onnx")

    fp16_path = base.export(format="engine", imgsz=args.imgsz, half=True, device=args.device)
    fp16_target = copy_exported(fp16_path, models_dir / "helmet_yolo11n_fp16.engine")
    validate(fp16_target, args.data, args.imgsz, args.device, "val_tensorrt_fp16")

    int8_path = base.export(format="engine", imgsz=args.imgsz, int8=True, data=args.data, device=args.device)
    int8_target = copy_exported(int8_path, models_dir / "helmet_yolo11n_int8.engine")
    validate(int8_target, args.data, args.imgsz, args.device, "val_tensorrt_int8")


if __name__ == "__main__":
    main()
