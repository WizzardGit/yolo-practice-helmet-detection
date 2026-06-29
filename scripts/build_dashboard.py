from pathlib import Path
import csv
import html


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
OUT = DEMO / "runtime_dashboard.html"


CSS = """
html, body {
    margin: 0;
    padding: 0;
    background: #111;
    color: #f2f2f2;
    font-family: Arial, sans-serif;
    font-size: 18px;
    line-height: 1.55;
}

body {
    padding: 40px 64px;
}

main {
    max-width: 1180px;
    margin: 0 auto;
}

h1, h2, h3, p, li, td, th, summary, pre {
    text-indent: 0;
    margin-left: 0;
    clip-path: none;
}

h1 {
    font-size: 40px;
    margin: 0 0 12px 0;
}

h2 {
    font-size: 30px;
    margin: 44px 0 18px 0;
    padding-top: 24px;
    border-top: 1px solid #333;
}

h3 {
    font-size: 22px;
    margin: 0 0 12px 0;
}

.card {
    background: #1b1b1b;
    border: 1px solid #333;
    border-radius: 14px;
    padding: 26px 32px;
    margin: 18px 0;
}

.muted {
    color: #aaa;
}

.good {
    color: #9be28f;
    font-weight: bold;
}

.warn {
    color: #ffd36a;
    font-weight: bold;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 15px;
}

th, td {
    border: 1px solid #333;
    padding: 10px 12px;
    text-align: left;
    vertical-align: top;
}

th {
    background: #242424;
}

tr:nth-child(even) td {
    background: #171717;
}

code, pre {
    font-family: Consolas, monospace;
}

pre {
    background: #080808;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 18px 22px;
    overflow-x: auto;
    white-space: pre-wrap;
    font-size: 14px;
}

a {
    color: #8ab4ff;
}

.links a {
    display: inline-block;
    margin: 8px 18px 8px 0;
}

details {
    background: #171717;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 14px 0;
}

summary {
    cursor: pointer;
    font-weight: bold;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 18px;
}

.image-card {
    background: #1b1b1b;
    border: 1px solid #333;
    border-radius: 14px;
    padding: 16px;
}

.image-card img {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #333;
}

.missing {
    color: #ff8888;
}
"""


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def read_text(path, limit=25000):
    path = Path(path)
    if not path.exists():
        return "Файл не найден: " + str(path.relative_to(ROOT))
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > limit:
        text = text[:limit] + "\n\n... файл обрезан для просмотра ..."
    return text


def make_table(rows):
    if not rows:
        return "<p class='muted'>CSV не найден или пустой.</p>"

    headers = list(rows[0].keys())
    out = ["<table>", "<thead><tr>"]

    for h in headers:
        out.append("<th>" + html.escape(h) + "</th>")

    out.append("</tr></thead><tbody>")

    for row in rows:
        out.append("<tr>")
        for h in headers:
            out.append("<td>" + html.escape(str(row.get(h, ""))) + "</td>")
        out.append("</tr>")

    out.append("</tbody></table>")
    return "\n".join(out)


def file_link(path, title):
    path = Path(path)
    if not path.exists():
        return "<span class='missing'>" + html.escape(title) + " не найден</span>"

    href = "../" + path.relative_to(ROOT).as_posix()
    return "<a href='" + html.escape(href) + "'>" + html.escape(title) + "</a>"


def code_block(text):
    return "<pre>" + html.escape(text) + "</pre>"


def prediction_cards():
    root = ROOT / "results" / "predictions"

    items = [
        ("pytorch", "PyTorch"),
        ("onnx", "ONNX Runtime CUDA"),
        ("tensorrt_fp16", "TensorRT FP16"),
        ("tensorrt_int8", "TensorRT INT8"),
    ]

    cards = []

    for folder, title in items:
        d = root / folder
        img = None

        if d.exists():
            files = sorted(
                list(d.rglob("*.jpg")) +
                list(d.rglob("*.jpeg")) +
                list(d.rglob("*.png"))
            )
            if files:
                img = files[0]

        if img:
            href = "../" + img.relative_to(ROOT).as_posix()
            cards.append(
                "<div class='image-card'>"
                "<h3>" + html.escape(title) + "</h3>"
                "<a href='" + html.escape(href) + "'>"
                "<img src='" + html.escape(href) + "'>"
                "</a>"
                "<p class='muted'>" + html.escape(str(img.relative_to(ROOT))) + "</p>"
                "</div>"
            )

    if not cards:
        return "<p class='muted'>Картинки предсказаний не найдены.</p>"

    return "<div class='grid'>" + "\n".join(cards) + "</div>"


def main():
    DEMO.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(ROOT / "results" / "benchmark_summary.csv")
    repeated_rows = read_csv(ROOT / "results" / "benchmark_runtime_repeated.csv")

    html_parts = []

    html_parts.append("<!doctype html>")
    html_parts.append("<html lang='ru'>")
    html_parts.append("<head>")
    html_parts.append("<meta charset='utf-8'>")
    html_parts.append("<title>YOLO Helmet Detection  результаты</title>")
    html_parts.append("<style>" + CSS + "</style>")
    html_parts.append("</head>")
    html_parts.append("<body>")
    html_parts.append("<main>")

    html_parts.append("<h1>YOLO Helmet Detection</h1>")
    html_parts.append("<p class='muted'>Результаты обучения, экспорта, benchmark и анализа runtime-графов.</p>")

    html_parts.append("<h2>Краткий вывод</h2>")
    html_parts.append("""
    <div class='card'>
        <p>Модель <b>YOLO11n</b> была дообучена для детекции касок, класс <code>helmet</code>.</p>
        <p><span class='good'>Лучший практический вариант  TensorRT FP16.</span></p>
        <p><span class='warn'>TensorRT INT8 уменьшает размер engine, но не даёт заметного ускорения относительно FP16 и снижает качество.</span></p>
    </div>
    """)

    html_parts.append("<h2>Окружение</h2>")
    html_parts.append("""
    <table>
        <tr><th>Параметр</th><th>Значение</th></tr>
        <tr><td>GPU</td><td>NVIDIA GeForce RTX 4070, 12 GB VRAM</td></tr>
        <tr><td>OS</td><td>Windows 10 Pro 22H2</td></tr>
        <tr><td>Python</td><td>3.11.7</td></tr>
        <tr><td>PyTorch</td><td>2.12.1+cu126</td></tr>
        <tr><td>CUDA</td><td>12.6</td></tr>
        <tr><td>Dataset test split</td><td>1604 images, 4863 helmet objects</td></tr>
        <tr><td>imgsz</td><td>640</td></tr>
        <tr><td>batch</td><td>1</td></tr>
    </table>
    """)

    html_parts.append("<h2>Методология замеров</h2>")
    html_parts.append("""
    <div class='card'>
        <p>Время взято из вывода Ultralytics <code>model.val()</code>: <code>Speed: preprocess ms, inference ms, postprocess ms per image</code>.</p>
        <p>Для сравнения использовался показатель <b>inference ms</b>, потому что он отражает прямое время работы модели без загрузки данных и постобработки.</p>
        <p>Для каждого runtime был сделан warmup, затем 5 повторных прогонов на одном и том же test split.</p>
    </div>
    """)

    html_parts.append("<h2>Сводная таблица benchmark</h2>")
    html_parts.append(make_table(summary_rows))

    html_parts.append("<h2>Все повторные замеры</h2>")
    html_parts.append("<details><summary>Показать raw benchmark table</summary>")
    html_parts.append(make_table(repeated_rows))
    html_parts.append("</details>")

    html_parts.append("<h2>FP16 против INT8</h2>")
    html_parts.append("""
    <div class='card'>
        <p>По повторным замерам TensorRT INT8 не оказался быстрее TensorRT FP16: оба варианта дают примерно <b>1.5 ms inference per image</b>.</p>
        <p>Но качество у INT8 ниже: mAP50-95 около <b>0.663</b> против <b>0.679</b> у FP16.</p>
        <p>Вывод: INT8 полезен по размеру файла, но для маленькой YOLO11n на RTX 4070 при batch=1 лучший вариант  <b>TensorRT FP16</b>.</p>
    </div>
    """)

    html_parts.append("<h2>Визуальные предсказания</h2>")
    html_parts.append(prediction_cards())

    html_parts.append("<h2>ONNX graph analysis</h2>")
    html_parts.append("<details open><summary>ONNX original graph operators</summary>")
    html_parts.append(code_block(read_text(ROOT / "runtime_graphs" / "onnx_original_ops.txt")))
    html_parts.append("</details>")

    html_parts.append("<details><summary>ONNX Runtime optimized graph operators</summary>")
    html_parts.append(code_block(read_text(ROOT / "runtime_graphs" / "onnx_ort_optimized_ops.txt")))
    html_parts.append("</details>")

    html_parts.append("<h2>TensorRT engine analysis</h2>")
    html_parts.append("<details open><summary>TensorRT FP16 engine info</summary>")
    html_parts.append(code_block(read_text(ROOT / "runtime_graphs" / "tensorrt_fp16_engine_info.txt")))
    html_parts.append("</details>")

    html_parts.append("<details open><summary>TensorRT INT8 engine info</summary>")
    html_parts.append(code_block(read_text(ROOT / "runtime_graphs" / "tensorrt_int8_engine_info.txt")))
    html_parts.append("</details>")

    html_parts.append("<h2>Файлы результатов</h2>")
    html_parts.append("<div class='card links'>")
    html_parts.append(file_link(ROOT / "results" / "benchmark_summary.csv", "benchmark_summary.csv"))
    html_parts.append(file_link(ROOT / "results" / "benchmark_runtime_repeated.csv", "benchmark_runtime_repeated.csv"))
    html_parts.append(file_link(ROOT / "RUNTIME_ANALYSIS.md", "RUNTIME_ANALYSIS.md"))
    html_parts.append(file_link(ROOT / "runtime_graphs" / "onnx_original_ops.txt", "onnx_original_ops.txt"))
    html_parts.append(file_link(ROOT / "runtime_graphs" / "onnx_ort_optimized_ops.txt", "onnx_ort_optimized_ops.txt"))
    html_parts.append(file_link(ROOT / "runtime_graphs" / "tensorrt_fp16_engine_info.txt", "tensorrt_fp16_engine_info.txt"))
    html_parts.append(file_link(ROOT / "runtime_graphs" / "tensorrt_int8_engine_info.txt", "tensorrt_int8_engine_info.txt"))
    html_parts.append("</div>")

    html_parts.append("</main>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    OUT.write_text("\n".join(html_parts), encoding="utf-8")
    print("Clean dashboard saved:", OUT)


if __name__ == "__main__":
    main()
