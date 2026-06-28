$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $PSScriptRoot
$assets = Join-Path $root "report_assets"
$demoAssets = Join-Path $root "demo\assets"
$demoDir = Join-Path $root "demo"

New-Item -ItemType Directory -Force $assets | Out-Null
New-Item -ItemType Directory -Force $demoAssets | Out-Null
New-Item -ItemType Directory -Force $demoDir | Out-Null

function Copy-Asset {
    param(
        [string]$Source,
        [string]$Name
    )

    if (Test-Path $Source) {
        $dst1 = Join-Path $assets $Name
        $dst2 = Join-Path $demoAssets $Name
        Copy-Item $Source $dst1 -Force
        Copy-Item $Source $dst2 -Force
        Write-Host "OK: $Name"
    } else {
        Write-Host "MISS: $Source" -ForegroundColor Yellow
    }
}

$train = Join-Path $root "runs\train_helmet"

Copy-Asset (Join-Path $train "results.png") "01_training_results.png"
Copy-Asset (Join-Path $train "confusion_matrix.png") "02_confusion_matrix.png"
Copy-Asset (Join-Path $train "PR_curve.png") "03_pr_curve.png"
Copy-Asset (Join-Path $train "F1_curve.png") "04_f1_curve.png"
Copy-Asset (Join-Path $train "labels.jpg") "05_dataset_labels.jpg"

$csv = Join-Path $root "results\runtime_comparison.csv"
Copy-Asset $csv "runtime_comparison.csv"

$predictionDirs = [ordered]@{
    "pytorch"       = "results\predictions\pytorch"
    "onnx"          = "results\predictions\onnx"
    "tensorrt_fp16" = "results\predictions\tensorrt_fp16"
    "tensorrt_int8" = "results\predictions\tensorrt_int8"
}

# Ищем одну общую картинку, которая есть во всех runtime
$commonNames = $null

foreach ($key in $predictionDirs.Keys) {
    $dir = Join-Path $root $predictionDirs[$key]
    if (!(Test-Path $dir)) {
        Write-Host "MISS prediction dir: $dir" -ForegroundColor Yellow
        continue
    }

    $files = Get-ChildItem $dir -File | Where-Object {
        $_.Extension -match "^\.(jpg|jpeg|png)$"
    }

    $names = $files | Select-Object -ExpandProperty BaseName

    if ($null -eq $commonNames) {
        $commonNames = $names
    } else {
        $commonNames = $commonNames | Where-Object { $names -contains $_ }
    }
}

$chosenBase = $null
if ($commonNames -and $commonNames.Count -gt 0) {
    $chosenBase = $commonNames | Select-Object -First 1
    Write-Host "Common prediction image selected: $chosenBase"
} else {
    Write-Host "No common prediction image found. Will copy first image from each runtime." -ForegroundColor Yellow
}

$i = 1
foreach ($key in $predictionDirs.Keys) {
    $dir = Join-Path $root $predictionDirs[$key]
    if (!(Test-Path $dir)) { continue }

    if ($chosenBase) {
        $file = Get-ChildItem $dir -File | Where-Object {
            $_.BaseName -eq $chosenBase -and $_.Extension -match "^\.(jpg|jpeg|png)$"
        } | Select-Object -First 1
    } else {
        $file = Get-ChildItem $dir -File | Where-Object {
            $_.Extension -match "^\.(jpg|jpeg|png)$"
        } | Select-Object -First 1
    }

    if ($file) {
        $ext = $file.Extension.ToLower()
        Copy-Asset $file.FullName ("predict_{0:D2}_{1}{2}" -f $i, $key, $ext)
        $i++
    }
}

# Делаем HTML-демо
$tableHtml = "<p>runtime_comparison.csv not found</p>"
$csvForDemo = Join-Path $demoAssets "runtime_comparison.csv"

if (Test-Path $csvForDemo) {
    try {
        $tableHtml = Import-Csv $csvForDemo | ConvertTo-Html -Fragment
    } catch {
        $tableHtml = "<p>Could not parse runtime_comparison.csv</p>"
    }
}

$imageHtml = ""
$images = Get-ChildItem $demoAssets -File | Where-Object {
    $_.Extension -match "^\.(jpg|jpeg|png)$"
} | Sort-Object Name

foreach ($img in $images) {
    $safeName = $img.Name
    $imageHtml += "<figure><img src='assets/$safeName' alt='$safeName'><figcaption>$safeName</figcaption></figure>`n"
}

$html = @"
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>YOLO Practice Demo</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; line-height: 1.5; }
    h1, h2 { margin-top: 28px; }
    table { border-collapse: collapse; margin: 16px 0; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
    th { background: #f2f2f2; }
    figure { margin: 24px 0; padding: 12px; border: 1px solid #ddd; }
    img { max-width: 100%; height: auto; display: block; }
    figcaption { margin-top: 8px; color: #555; font-size: 14px; }
    .note { background: #fff8dc; padding: 12px; border: 1px solid #ead27a; }
  </style>
</head>
<body>
  <h1>YOLO Practice Demo</h1>

  <p>
    Практика: дообучение и оптимизация YOLO для нового класса объектов.
    Модель YOLO11n была дообучена на классе <b>helmet</b>, затем экспортирована
    и проверена в PyTorch, ONNX Runtime CUDA, TensorRT FP16 и TensorRT INT8.
  </p>

  <div class="note">
    Это демо не запускает обучение заново. Оно открывает готовые результаты:
    графики обучения, метрики и визуальное сравнение предсказаний.
  </div>

  <h2>Runtime comparison</h2>
  $tableHtml

  <h2>Images and plots</h2>
  $imageHtml
</body>
</html>
"@

$htmlPath = Join-Path $demoDir "index.html"
$html | Set-Content -Encoding UTF8 $htmlPath

Write-Host ""
Write-Host "DONE"
Write-Host "Report assets: $assets"
Write-Host "Demo page: $htmlPath"
