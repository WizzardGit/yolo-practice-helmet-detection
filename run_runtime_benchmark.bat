@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo RUN YOLO RUNTIME BENCHMARK
echo ============================================================
echo.
echo This will run:
echo - PyTorch
echo - ONNX Runtime CUDA
echo - TensorRT FP16
echo - TensorRT INT8
echo.
echo Output:
echo - results\benchmark_runtime_repeated.csv
echo - results\benchmark_summary.csv
echo - demo\runtime_dashboard.html
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found.
    echo Create environment first or run from prepared local project.
    pause
    exit /b 1
)

if not exist "scripts\benchmark_runtimes.py" (
    echo ERROR: scripts\benchmark_runtimes.py not found.
    pause
    exit /b 1
)

if not exist "scripts\summarize_benchmark.py" (
    echo ERROR: scripts\summarize_benchmark.py not found.
    pause
    exit /b 1
)

echo Running benchmark...
echo.

".venv\Scripts\python.exe" scripts\benchmark_runtimes.py

if errorlevel 1 (
    echo.
    echo ERROR: benchmark failed.
    pause
    exit /b 1
)

echo.
echo Building benchmark summary...
".venv\Scripts\python.exe" scripts\summarize_benchmark.py

echo.
echo Rebuilding dashboard...
".venv\Scripts\python.exe" scripts\build_dashboard.py

echo.
echo Opening updated dashboard...
if exist "demo\runtime_dashboard.html" start "" "demo\runtime_dashboard.html"

echo.
echo Done.
pause
