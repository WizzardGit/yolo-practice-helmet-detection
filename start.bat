@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo YOLO Helmet Detection
echo ============================================================
echo.
echo [1] Open results dashboard
echo [2] Run runtime benchmark again
echo [3] Open raw result folders
echo [4] Rebuild dashboard
echo.
echo Recommended for checking: press 1
echo To reproduce runtime measurements: press 2
echo.

choice /C 1234 /N /M "Select action [1/2/3/4]: "

if errorlevel 4 goto rebuild
if errorlevel 3 goto folders
if errorlevel 2 goto benchmark
if errorlevel 1 goto dashboard

:dashboard
if not exist "demo\runtime_dashboard.html" (
    if exist "scripts\build_dashboard.py" (
        if exist ".venv\Scripts\python.exe" (
            ".venv\Scripts\python.exe" scripts\build_dashboard.py
        ) else (
            python scripts\build_dashboard.py
        )
    )
)

if exist "demo\runtime_dashboard.html" start "" "demo\runtime_dashboard.html"
if exist "demo\index.html" start "" "demo\index.html"
exit /b 0

:benchmark
if exist "run_runtime_benchmark.bat" (
    call run_runtime_benchmark.bat
) else (
    echo ERROR: run_runtime_benchmark.bat not found.
    pause
)
exit /b 0

:folders
if exist "results" start "" "results"
if exist "runtime_graphs" start "" "runtime_graphs"
if exist "scripts" start "" "scripts"
exit /b 0

:rebuild
if exist "scripts\build_dashboard.py" (
    if exist ".venv\Scripts\python.exe" (
        ".venv\Scripts\python.exe" scripts\build_dashboard.py
    ) else (
        python scripts\build_dashboard.py
    )
)
if exist "demo\runtime_dashboard.html" start "" "demo\runtime_dashboard.html"
pause
exit /b 0
