# YOLO Practice: helmet detection

## 1. Project description

University practice project.

Topic:

Fine-tuning and optimization of YOLO models for new object classes.

In this project, YOLO11n was fine-tuned for one target class: helmet.

The model was exported and tested with several runtimes:

- PyTorch
- ONNX Runtime CUDA
- TensorRT FP16
- TensorRT INT8

The project includes:

- dataset filtering for one class;
- YOLO fine-tuning;
- ONNX export;
- TensorRT optimization;
- INT8 quantization;
- runtime performance comparison;
- visual prediction comparison;
- ONNX graph analysis in Netron.

## 2. Dataset

Dataset: Hard Hat Workers.

Original classes:

- head
- helmet
- person

For this experiment only one class was kept:

- helmet

After filtering:

- train images: 4832
- validation/test images: 1604
- train helmet objects: 14884
- test helmet objects: 4863

## 3. Training

Base model:

- YOLO11n

Training parameters:

- epochs: 30
- image size: 640
- batch size: 16
- device: CUDA:0
- GPU: NVIDIA GeForce RTX 4070

Training command:

    yolo detect train model=yolo11n.pt data="D:\YOLOPractice\datasets\hardhat_helmet_only\data.yaml" epochs=30 imgsz=640 batch=16 device=0 project="D:\YOLOPractice\runs" name="train_helmet"

## 4. Training results

Final PyTorch metrics:

| Metric | Value |
|---|---:|
| Precision | 0.963 |
| Recall | 0.935 |
| mAP50 | 0.981 |
| mAP50-95 | 0.679 |

## 5. Runtime comparison

| Runtime | Model | Precision | Recall | mAP50 | mAP50-95 | Inference ms | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| PyTorch | best.pt | 0.963 | 0.935 | 0.981 | 0.679 | 1.3 | 769 |
| ONNX Runtime CUDA | helmet_yolo11n.onnx | 0.960 | 0.937 | 0.981 | 0.679 | 3.0 | 333 |
| TensorRT FP16 | helmet_yolo11n_fp16.engine | 0.960 | 0.937 | 0.981 | 0.679 | 1.5 | 667 |
| TensorRT INT8 | helmet_yolo11n_int8.engine | 0.960 | 0.932 | 0.978 | 0.663 | 1.9 | 526 |

## 6. Result

YOLO11n was successfully fine-tuned for the new helmet class.

ONNX export and TensorRT export were completed successfully.

PyTorch, ONNX Runtime CUDA and TensorRT FP16 showed almost the same detection quality.

TensorRT INT8 reduced the engine size, but slightly reduced mAP50-95 and did not improve inference speed compared to FP16 in this experiment.

For this setup, TensorRT FP16 was the best optimization option.

## 7. Demo

On Windows, run:

    start.bat

The script opens an HTML demo with:

- runtime comparison table;
- training plots;
- confusion matrix;
- prediction examples for PyTorch, ONNX Runtime, TensorRT FP16 and TensorRT INT8.

## 8. Repository structure

    .
     README.md
     requirements.txt
     start.bat
     scripts/
        filter_helmet_dataset.py
        make_demo.ps1
     report/
     report_assets/
     demo/
        index.html
        assets/
     results/
         runtime_comparison.csv

## 9. Not included in repository

The repository does not include:

- .venv
- datasets
- training runs
- .pt model files
- .onnx model files
- TensorRT .engine files

These files are not included because they are large and hardware-dependent.

TensorRT engines may depend on GPU, CUDA, TensorRT and driver versions.

For practice review, the report, source scripts, metrics, screenshots and demo are enough.

## 10. Environment setup

Example setup on Windows with NVIDIA GPU:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    python -m pip install --upgrade pip wheel setuptools

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

    pip install -r requirements.txt

Check CUDA:

    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

Check ONNX Runtime:

    python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"

Check TensorRT:

    python -c "import tensorrt as trt; print(trt.__version__)"
