from pathlib import Path
from collections import defaultdict
import csv
import statistics


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "benchmark_runtime_repeated.csv"
OUTPUT = ROOT / "results" / "benchmark_summary.csv"


def as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def main():
    rows = list(csv.DictReader(INPUT.open("r", encoding="utf-8")))

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["runtime"]].append(row)

    summary = []

    for runtime, items in grouped.items():
        inference = [as_float(x["inference_ms"]) for x in items]
        inference = [x for x in inference if x is not None]

        precision = [as_float(x["precision"]) for x in items]
        recall = [as_float(x["recall"]) for x in items]
        map50 = [as_float(x["mAP50"]) for x in items]
        map5095 = [as_float(x["mAP50-95"]) for x in items]

        precision = [x for x in precision if x is not None]
        recall = [x for x in recall if x is not None]
        map50 = [x for x in map50 if x is not None]
        map5095 = [x for x in map5095 if x is not None]

        summary.append({
            "runtime": runtime,
            "repeats": len(items),
            "inference_ms_values": " / ".join(str(round(x, 3)) for x in inference),
            "inference_ms_mean": round(statistics.mean(inference), 3),
            "inference_ms_median": round(statistics.median(inference), 3),
            "inference_ms_min": round(min(inference), 3),
            "inference_ms_max": round(max(inference), 3),
            "precision_mean": round(statistics.mean(precision), 4) if precision else "",
            "recall_mean": round(statistics.mean(recall), 4) if recall else "",
            "mAP50_mean": round(statistics.mean(map50), 4) if map50 else "",
            "mAP50-95_mean": round(statistics.mean(map5095), 4) if map5095 else "",
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
