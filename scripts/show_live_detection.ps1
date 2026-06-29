Set-Location (Split-Path $PSScriptRoot -Parent)

$ROOT = (Get-Location).Path
$PY = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

$BEST = Get-ChildItem -Recurse -Filter best.pt |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $BEST) {
    Write-Host "[ERROR] best.pt not found"
    pause
    exit 1
}

if (-not (Test-Path "compare_images")) {
    Write-Host "[ERROR] compare_images folder not found"
    pause
    exit 1
}

Write-Host "[INFO] Using model:" $BEST.FullName
Write-Host "[INFO] Running live prediction..."

& $PY -c "from ultralytics import YOLO; model=YOLO(r'$($BEST.FullName)'); model.predict(source=r'$ROOT\compare_images', save=True, conf=0.25, project=r'$ROOT\demo', name='live_predict', exist_ok=True)"

explorer "$ROOT\demo\live_predict"
pause
