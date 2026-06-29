# -*- coding: utf-8 -*-
from pathlib import Path
import os, csv, io, html, shutil, datetime

ROOT = Path("D:/YOLOPractice") if Path("D:/YOLOPractice").exists() else Path.cwd()
os.chdir(ROOT)

def esc(x): return html.escape(str(x))
def code(x): return f"<code>{esc(x)}</code>"
def ok(p): return "✅" if (ROOT / p).exists() else "⚠️"
def read_csv_any(p):
    p = ROOT / p
    if not p.exists(): return [], []
    txt = p.read_text(encoding="utf-8-sig", errors="replace")
    if not txt.strip(): return [], []
    sample = txt[:4000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except Exception:
        dialect = csv.excel
        dialect.delimiter = "," if sample.count(",") >= sample.count(";") else ";"
    rows = list(csv.reader(io.StringIO(txt), dialect))
    return rows[:1], rows[1:]

def table_from_csv(path, limit=120):
    head, rows = read_csv_any(path)
    if not head:
        return f"<p class='muted'>Файл {code(path)} не найден или пустой.</p>"
    header = head[0]
    th = "".join(f"<th>{esc(c)}</th>" for c in header)
    trs = []
    for r in rows[:limit]:
        tds = "".join(f"<td>{esc(r[i]) if i < len(r) else ''}</td>" for i in range(len(header)))
        trs.append(f"<tr>{tds}</tr>")
    return f"<div class='tw'><table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>"

def make_excel_csv(path):
    head, rows = read_csv_any(path)
    if not head: return
    src = ROOT / path
    dst = src.with_name(src.stem + "_excel.csv")
    with dst.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(head[0]); w.writerows(rows)

for p in ["results/benchmark_summary.csv", "results/benchmark_runtime_repeated.csv", "results/runtime_comparison.csv", "report_assets/runtime_comparison.csv"]:
    try: make_excel_csv(p)
    except Exception as e: print("[warn]", p, e)

def first(patterns):
    for pat in patterns:
        xs = sorted(ROOT.glob(pat))
        if xs:
            try: return str(xs[0].relative_to(ROOT)).replace("\\","/")
            except Exception: return str(xs[0]).replace("\\","/")
    return "не найдено"

best_pt = first(["runs/**/train_helmet*/weights/best.pt", "runs/**/weights/best.pt"])
onnx = first(["results/models/*.onnx", "**/helmet*.onnx"])
fp16 = first(["results/models/*fp16*.engine", "**/*fp16*.engine"])
int8 = first(["results/models/*int8*.engine", "**/*int8*.engine"])

checks = [
("Код обучения", "scripts/train_helmet.py", "закрывает замечание «кода обучения нет»"),
("Фильтрация датасета", "scripts/filter_helmet_dataset.py", "доказывает подготовку нового класса helmet"),
("Экспорт/валидация", "scripts/export_validate.py", "PyTorch → ONNX → TensorRT FP16/INT8"),
("Benchmark", "scripts/benchmark_runtimes.py", "warmup + повторные замеры runtime"),
("Сводка benchmark", "scripts/summarize_benchmark.py", "итоговые таблицы"),
("ORT optimized graph", "scripts/save_ort_optimized_graph.py", "сохранение оптимизированного ONNX Runtime graph"),
("Анализ графов", "scripts/inspect_runtime_graphs.py", "ONNX/ORT/TensorRT graph evidence"),
("Raw CSV", "results/benchmark_runtime_repeated.csv", "сырые повторные замеры"),
("Summary CSV", "results/benchmark_summary.csv", "сводка по runtime"),
("ONNX original ops", "runtime_graphs/onnx_original_ops.txt", "операторы исходного ONNX-графа"),
("ORT optimized ops", "runtime_graphs/onnx_ort_optimized_ops.txt", "операторы после оптимизации ORT"),
]
check_rows = "\n".join(f"<tr><td>{ok(p)}</td><td>{esc(a)}</td><td>{code(p)}</td><td>{esc(d)}</td></tr>" for a,p,d in checks)

imgs = [
("Графики обучения", "report_assets/01_training_results.png"),
("Confusion matrix", "report_assets/02_confusion_matrix.png"),
("Dataset labels", "report_assets/05_dataset_labels.jpg"),
("PyTorch predict", "report_assets/predict_01_pytorch.jpg"),
("ONNX predict", "report_assets/predict_02_onnx.jpg"),
("TensorRT FP16 predict", "report_assets/predict_03_tensorrt_fp16.jpg"),
("TensorRT INT8 predict", "report_assets/predict_04_tensorrt_int8.jpg"),
]
img_html = "".join(f"<figure><img src='../{esc(p)}'><figcaption>{esc(t)}</figcaption></figure>" for t,p in imgs if (ROOT/p).exists())

css = """
body{margin:0;background:#0f1117;color:#eef2f7;font-family:Segoe UI,Arial,sans-serif;line-height:1.55}
.wrap{max-width:1150px;margin:auto;padding:26px 20px 70px}
.hero,section{background:#171a22;border:1px solid #303847;border-radius:16px;padding:22px;margin:14px 0}
.hero{background:linear-gradient(135deg,#18202d,#11141b)}
h1{margin:0 0 8px;font-size:31px} h2{margin:0 0 12px;font-size:23px} h3{margin:20px 0 8px}
.muted{color:#aab3c2}.badge{display:inline-block;background:#202838;border:1px solid #303847;border-radius:999px;padding:7px 11px;margin:5px 5px 5px 0}
.nav a{display:inline-block;color:#dff1ff;text-decoration:none;background:#202838;border:1px solid #303847;border-radius:999px;padding:7px 10px;margin:4px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:#171a22;border:1px solid #303847;border-radius:14px;padding:15px}.num{font-size:25px;font-weight:800}.label{color:#aab3c2;font-size:14px}
.tw{overflow:auto;border:1px solid #303847;border-radius:12px;margin:10px 0} table{width:100%;border-collapse:collapse;min-width:760px} th,td{padding:10px 12px;border-bottom:1px solid #303847;vertical-align:top} th{background:#1f2430;text-align:left}
code{background:#0b0e14;border:1px solid #252c3a;border-radius:6px;padding:2px 6px;white-space:nowrap;color:#e6edf7}
pre{background:#0b0e14;border:1px solid #252c3a;border-radius:12px;padding:13px;overflow:auto;color:#e6edf7}
.call{border-left:4px solid #7cc4ff;background:#121823;border-radius:10px;padding:10px 14px;margin:10px 0}
.fig{display:grid;grid-template-columns:repeat(2,1fr);gap:12px} figure{margin:0;background:#10151f;border:1px solid #303847;border-radius:14px;padding:9px} img{width:100%;border-radius:10px} figcaption{color:#aab3c2;font-size:14px;margin-top:6px}
@media(max-width:850px){.grid,.fig{grid-template-columns:1fr}}
"""
html_page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>YOLO Helmet Detection — Dashboard</title><style>{css}</style></head><body><div class="wrap">
<div class="hero">
<h1>YOLO Helmet Detection — единый dashboard</h1>
<p class="muted">Главная страница проверки проекта. Здесь собраны ответы на замечания, методология, результаты и ссылки на raw-файлы, чтобы руководителю не искать всё по .md/.csv/.txt.</p>
<span class="badge">Класс: <b>helmet</b></span><span class="badge">Модель: <b>YOLO11n</b></span><span class="badge">GPU: <b>RTX 4070 12 GB</b></span><span class="badge">Runtime: <b>PyTorch / ONNX / TensorRT</b></span><span class="badge">Итог: <b>TensorRT FP16</b></span>
<div class="nav"><a href="#answers">Замечания</a><a href="#model">Форматы</a><a href="#train">Обучение</a><a href="#method">Методология</a><a href="#res">Результаты</a><a href="#graphs">Графы</a><a href="#files">Файлы</a><a href="#cmd">Команды</a></div>
</div>

<div class="grid">
<div class="card"><div class="num">~0.679</div><div class="label">mAP50-95 у FP16 без заметной потери качества</div></div>
<div class="card"><div class="num">~1.5 ms</div><div class="label">TensorRT inference на изображение</div></div>
<div class="card"><div class="num">5×</div><div class="label">повторные замеры после warmup</div></div>
<div class="card"><div class="num">1 файл</div><div class="label">основная защита через dashboard</div></div>
</div>

<section id="answers"><h2>1. Ответы на замечания Матвея</h2>
<div class="tw"><table><thead><tr><th>Замечание</th><th>Что сделано</th><th>Где смотреть</th></tr></thead><tbody>
<tr><td>Кода обучения нет</td><td>Добавлен отдельный скрипт обучения YOLO11n на helmet.</td><td>{code("scripts/train_helmet.py")}</td></tr>
<tr><td>Как будто взял готовую модель</td><td>Показан pipeline: датасет → train → export → validate → benchmark.</td><td>{code("scripts/filter_helmet_dataset.py")}<br>{code("scripts/train_helmet.py")}<br>{code("scripts/export_validate.py")}</td></tr>
<tr><td>Нет анализа графов исполнения</td><td>Сохранены original ONNX graph и ONNX Runtime optimized graph. Для TensorRT сохранены info/error файлы.</td><td>{code("runtime_graphs/")}</td></tr>
<tr><td>Неясно, откуда миллисекунды</td><td>Описана методология: Ultralytics model.val(), строка Speed, inference ms/image.</td><td>Раздел 4 ниже</td></tr>
<tr><td>Не указана видеокарта</td><td>Указана RTX 4070 12 GB VRAM.</td><td>Раздел 4 ниже</td></tr>
<tr><td>INT8/FP16 неочевидно</td><td>Сделаны warmup + 5 повторов. Вывод: INT8 примерно равен FP16 по скорости, но ниже по качеству.</td><td>{code("results/benchmark_runtime_repeated.csv")}<br>{code("results/benchmark_summary.csv")}</td></tr>
<tr><td>Воспроизвести ничего не могу</td><td>Добавлены скрипты этапов и старт через один файл.</td><td>{code("start.bat")}</td></tr>
</tbody></table></div></section>

<section id="model"><h2>2. Что такое .pt / .onnx / .engine</h2>
<div class="tw"><table><thead><tr><th>Формат</th><th>Файл</th><th>Что значит</th></tr></thead><tbody>
<tr><td>{code(".pt")}</td><td>{code("yolo11n.pt")} / {code(best_pt)}</td><td>PyTorch-веса. yolo11n.pt — стартовая модель, best.pt — результат дообучения.</td></tr>
<tr><td>{code(".onnx")}</td><td>{code(onnx)}</td><td>Универсальный формат для ONNX Runtime.</td></tr>
<tr><td>{code(".engine")}</td><td>{code(fp16)}<br>{code(int8)}</td><td>TensorRT engine для быстрого запуска на NVIDIA GPU. FP16 — половинная точность, INT8 — квантизация.</td></tr>
</tbody></table></div></section>

<section id="train"><h2>3. Датасет и обучение</h2>
<p>Использовался датасет рабочих касок. Из исходных классов оставлен один целевой класс <b>helmet</b>.</p>
<div class="tw"><table><tbody>
<tr><th>Модель</th><td>YOLO11n</td></tr><tr><th>Класс</th><td>helmet</td></tr><tr><th>Train</th><td>4832 images / 14884 helmet objects</td></tr><tr><th>Val/Test</th><td>1604 images / 4863 helmet objects</td></tr><tr><th>Epochs</th><td>30</td></tr><tr><th>imgsz</th><td>640</td></tr><tr><th>batch train</th><td>16</td></tr><tr><th>результат</th><td>{code(best_pt)}</td></tr>
</tbody></table></div>
<div class="call"><b>Метрики:</b> Precision ≈ 0.963, Recall ≈ 0.935, mAP50 ≈ 0.981, mAP50-95 ≈ 0.679.</div>
</section>

<section id="method"><h2>4. Методология замеров</h2>
<div class="call">Миллисекунды — это не время запуска всего скрипта. Для каждого runtime запускался Ultralytics <b>model.val()</b>, из строки <b>Speed</b> бралось <b>inference ms/image</b>.</div>
<div class="tw"><table><tbody>
<tr><th>GPU</th><td>NVIDIA GeForce RTX 4070, 12 GB VRAM</td></tr><tr><th>OS</th><td>Windows 10</td></tr><tr><th>Python</th><td>3.11.7</td></tr><tr><th>CUDA/PyTorch</th><td>CUDA 12.6 / PyTorch 2.12.1+cu126</td></tr><tr><th>imgsz</th><td>640</td></tr><tr><th>batch benchmark</th><td>1</td></tr><tr><th>повторы</th><td>warmup + 5 повторных прогонов</td></tr><tr><th>источник ms</th><td>Ultralytics Speed → inference ms/image</td></tr>
</tbody></table></div></section>

<section id="res"><h2>5. Результаты runtime</h2>
<div class="tw"><table><thead><tr><th>Runtime</th><th>Формат</th><th>Inference avg</th><th>Median</th><th>mAP50-95</th><th>Вывод</th></tr></thead><tbody>
<tr><td>PyTorch</td><td>.pt</td><td>~4.98 ms</td><td>~4.10 ms</td><td>~0.679</td><td>baseline</td></tr>
<tr><td>ONNX Runtime CUDA</td><td>.onnx</td><td>~3.04 ms</td><td>~2.90 ms</td><td>~0.679</td><td>быстрее PyTorch</td></tr>
<tr><td>TensorRT FP16</td><td>.engine</td><td>~1.52 ms</td><td>~1.50 ms</td><td>~0.679</td><td><b>лучший практический вариант</b></td></tr>
<tr><td>TensorRT INT8</td><td>.engine</td><td>~1.50 ms</td><td>~1.50 ms</td><td>~0.663</td><td>меньше engine, но ниже качество</td></tr>
</tbody></table></div>
<p>Вывод: TensorRT дал главное ускорение. INT8 не дал заметного ускорения относительно FP16 на маленькой YOLO11n при batch=1, но снизил качество, поэтому выбирать лучше FP16.</p>
<h3>Сводка из CSV</h3>{table_from_csv("results/benchmark_summary.csv")}
<h3>Повторные замеры из CSV</h3>{table_from_csv("results/benchmark_runtime_repeated.csv")}
<p class="muted">Для Excel созданы копии с разделителем ;: {code("results/benchmark_summary_excel.csv")} и {code("results/benchmark_runtime_repeated_excel.csv")}.</p>
</section>

<section id="graphs"><h2>6. Анализ графов исполнения</h2>
<p>В dashboard вынесена суть, raw-файлы остаются доказательством.</p>
<div class="tw"><table><thead><tr><th>Статус</th><th>Файл</th><th>Что показывает</th></tr></thead><tbody>
<tr><td>{ok("runtime_graphs/onnx_original_ops.txt")}</td><td>{code("runtime_graphs/onnx_original_ops.txt")}</td><td>операторы исходного ONNX-графа</td></tr>
<tr><td>{ok("runtime_graphs/onnx_ort_optimized_ops.txt")}</td><td>{code("runtime_graphs/onnx_ort_optimized_ops.txt")}</td><td>операторы после оптимизации ONNX Runtime</td></tr>
<tr><td>{ok("runtime_graphs/tensorrt_fp16_engine_info.txt")}</td><td>{code("runtime_graphs/tensorrt_fp16_engine_info.txt")}</td><td>информация по TensorRT FP16 engine</td></tr>
<tr><td>{ok("runtime_graphs/tensorrt_int8_engine_info.txt")}</td><td>{code("runtime_graphs/tensorrt_int8_engine_info.txt")}</td><td>информация по TensorRT INT8 engine</td></tr>
</tbody></table></div>
<div class="call">TensorRT EngineInspector мог не открыть .engine из-за версии TensorRT/CUDA/driver/GPU. Это не скрыто: ошибка сохранена в {code("runtime_graphs/tensorrt_*_inspector_error.txt")}. Benchmark при этом запускался.</div>
</section>

<section id="files"><h2>7. Ключевые файлы проекта</h2>
<div class="tw"><table><thead><tr><th>Статус</th><th>Что</th><th>Файл</th><th>Зачем</th></tr></thead><tbody>{check_rows}</tbody></table></div>
<h3>Папки</h3>
<div class="tw"><table><tbody><tr><td>{code("scripts/")}</td><td>код эксперимента</td></tr><tr><td>{code("datasets/")}</td><td>датасет</td></tr><tr><td>{code("runs/")}</td><td>выходы Ultralytics</td></tr><tr><td>{code("results/")}</td><td>финальные результаты и CSV</td></tr><tr><td>{code("runtime_graphs/")}</td><td>анализ графов</td></tr><tr><td>{code("report_assets/")}</td><td>картинки для отчёта</td></tr><tr><td>{code("demo/runtime_dashboard.html")}</td><td>главная страница проверки</td></tr></tbody></table></div>
</section>

<section><h2>8. Визуальные материалы</h2><div class="fig">{img_html or "<p class='muted'>Картинки report_assets не найдены.</p>"}</div></section>

<section id="cmd"><h2>9. Команды</h2>
<p>Открыть проект для проверки:</p><pre>cd /d D:\\YOLOPractice
start.bat</pre>
<p>Пересобрать dashboard:</p><pre>cd /d D:\\YOLOPractice
.venv\\Scripts\\python.exe scripts\\build_dashboard.py
start demo\\runtime_dashboard.html</pre>
<p>Повторить runtime benchmark:</p><pre>cd /d D:\\YOLOPractice
run_runtime_benchmark.bat</pre>
</section>

<p class="muted">Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} · Root: {esc(ROOT)}</p>
</div></body></html>"""

(ROOT/"demo").mkdir(exist_ok=True)
(ROOT/"demo/runtime_dashboard.html").write_text(html_page, encoding="utf-8")

# make script reusable as scripts/build_dashboard.py
(ROOT/"scripts").mkdir(exist_ok=True)
target = ROOT/"scripts/build_dashboard.py"
try:
    cur = Path(__file__).resolve()
    if cur != target.resolve():
        if target.exists():
            shutil.copy2(target, ROOT/f"scripts/build_dashboard_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
        shutil.copy2(cur, target)
except Exception as e:
    print("[warn] build_dashboard copy:", e)

# start.bat opens dashboard directly
start = ROOT/"start.bat"
if start.exists():
    try: shutil.copy2(start, ROOT/f"start_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.bat")
    except Exception: pass
start.write_text(r"""@echo off
cd /d "%~dp0"
if not exist "demo\runtime_dashboard.html" (
    if exist ".venv\Scripts\python.exe" (
        ".venv\Scripts\python.exe" "scripts\build_dashboard.py"
    ) else (
        python "scripts\build_dashboard.py"
    )
)
start "" "%~dp0demo\runtime_dashboard.html"
""", encoding="utf-8")

(ROOT/"REVIEWER_README.md").write_text("# Быстрая проверка\n\nЗапустить `start.bat`. Откроется `demo/runtime_dashboard.html` — единый dashboard с ответами на замечания, методологией, результатами и ссылками на raw-файлы.\n", encoding="utf-8")

print("[OK] created demo/runtime_dashboard.html")
print("[OK] updated scripts/build_dashboard.py")
print("[OK] updated start.bat")
print("[OK] created Excel-friendly CSV copies where possible")
print("Run now: start.bat")
