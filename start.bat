@echo off
chcp 65001 >nul
set ROOT=%~dp0

echo Opening YOLO Practice Demo...

if exist "%ROOT%demo\index.html" (
    start "" "%ROOT%demo\index.html"
) else (
    echo demo\index.html not found.
)

if exist "%ROOT%report_assets" (
    start "" "%ROOT%report_assets"
)

if exist "%ROOT%report\YOLO_Practice_Report.pdf" (
    start "" "%ROOT%report\YOLO_Practice_Report.pdf"
)

if exist "%ROOT%report\YOLO_Practice_Report.docx" (
    start "" "%ROOT%report\YOLO_Practice_Report.docx"
)

echo.
echo Done.
echo If the browser did not open, manually open:
echo %ROOT%demo\index.html
echo.
pause
