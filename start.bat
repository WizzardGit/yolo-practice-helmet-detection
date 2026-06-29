@echo off
cd /d "%~dp0"
if not exist "demo\runtime_dashboard.html" (
    if exist ".venv\Scripts\python.exe" (
        ".venv\Scripts\python.exe" "scripts\build_dashboard.py"
    ) else (
        python "scripts\build_dashboard.py"
    )
)
start "" "%~dp0demo\runtime_dashboard.html"
