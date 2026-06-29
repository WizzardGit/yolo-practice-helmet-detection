@echo off
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -NoProfile -File "scripts\show_live_detection.ps1"
