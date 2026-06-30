from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def find_trtexec(user_path: str | None = None) -> str | None:
    if user_path:
        p = Path(user_path)
        if p.exists():
            return str(p)
    return shutil.which("trtexec") or shutil.which("trtexec.exe")


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def summarize_profile(profile_json):
    if not isinstance(profile_json, list):
        return []

    rows = []
    for item in profile_json:
        if not isinstance(item, dict):
            continue

        name = (
            item.get("name")
            or item.get("layerName")
            or item.get("LayerName")
            or item.get("layer")
            or "unknown"
        )

        time_ms = (
            item.get("timeMs")
            or item.get("averageMs")
            or item.get("avgMs")
            or item.get("latencyMs")
            or item.get("medianMs")
            or 0
        )

        try:
            time_ms = float(time_ms)
        except Exception:
            time_ms = 0.0

        rows.append({
            "layer": name,
            "time_ms": round(time_ms, 6),
            "raw": item,
        })

    rows.sort(key=lambda x: x["time_ms"], reverse=True)
    return rows[:30]


def summarize_layer_info(layer_json):
    counts = {}
    precisions = {}

    if not isinstance(layer_json, list):
        return {"layer_type_counts": counts, "precision_counts": precisions}

    for item in layer_json:
        if not isinstance(item, dict):
            continue

        typ = (
            item.get("LayerType")
            or item.get("layerType")
            or item.get("type")
            or item.get("subtype")
            or "unknown"
        )
        counts[typ] = counts.get(typ, 0) + 1

        precision = (
            item.get("precision")
            or item.get("Precision")
            or item.get("computePrecision")
            or ""
        )
        if precision:
            precisions[precision] = precisions.get(precision, 0) + 1

    return {
        "layer_type_counts": dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)),
        "precision_counts": dict(sorted(precisions.items(), key=lambda x: x[1], reverse=True)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="results/models/helmet_yolo11n_fp16.engine")
    ap.add_argument("--prefix", default="tensorrt_fp16")
    ap.add_argument("--outdir", default="runtime_graphs")
    ap.add_argument("--trtexec", default=None)
    args = ap.parse_args()

    engine = Path(args.engine)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trtexec = find_trtexec(args.trtexec)
    if not trtexec:
        print("[ERROR] trtexec not found.")
        print("Install TensorRT tools or pass path:")
        print(r'python scripts\profile_tensorrt_trtexec.py --trtexec "C:\path\to\trtexec.exe"')
        raise SystemExit(2)

    layer_info = outdir / f"{args.prefix}_layer_info.json"
    profile = outdir / f"{args.prefix}_profile.json"
    times = outdir / f"{args.prefix}_times.json"
    log = outdir / f"{args.prefix}_trtexec.log"
    summary_path = outdir / f"{args.prefix}_runtime_summary.json"

    cmd = [
        trtexec,
        f"--loadEngine={engine}",
        "--warmUp=500",
        "--iterations=100",
        "--avgRuns=10",
        "--dumpProfile",
        "--dumpLayerInfo",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={layer_info}",
        f"--exportProfile={profile}",
        f"--exportTimes={times}",
    ]

    print("[INFO] running:")
    print(" ".join(map(str, cmd)))

    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(proc.stdout, encoding="utf-8", errors="ignore")

    profile_json = load_json(profile)
    layer_json = load_json(layer_info)

    summary = {
        "engine": str(engine),
        "trtexec": trtexec,
        "returncode": proc.returncode,
        "files": {
            "layer_info": str(layer_info),
            "profile": str(profile),
            "times": str(times),
            "log": str(log),
        },
        "top_layers_by_profile_time": summarize_profile(profile_json),
        "layer_info_summary": summarize_layer_info(layer_json),
        "note": "Profile is produced by trtexec. It shows TensorRT runtime per-layer execution timing and layer information.",
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] log:", log)
    print("[OK] layer info:", layer_info)
    print("[OK] profile:", profile)
    print("[OK] times:", times)
    print("[OK] summary:", summary_path)

    if proc.returncode != 0:
        print("[WARN] trtexec finished with non-zero code. Check log.")


if __name__ == "__main__":
    main()
