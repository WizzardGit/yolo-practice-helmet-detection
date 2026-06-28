from pathlib import Path
import shutil
import yaml

RAW_ROOT = Path(r"D:\YOLOPractice\datasets\hardhat_raw")
OUT_ROOT = Path(r"D:\YOLOPractice\datasets\hardhat_helmet_only")

yaml_files = list(RAW_ROOT.rglob("data.yaml"))
if not yaml_files:
    raise FileNotFoundError("data.yaml not found inside hardhat_raw")

DATA_YAML = yaml_files[0]
YAML_DIR = DATA_YAML.parent

print(f"Using data.yaml: {DATA_YAML}")

with open(DATA_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data.get("names")

if isinstance(names, dict):
    names_list = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
else:
    names_list = list(names)

print("Source classes:")
for i, name in enumerate(names_list):
    print(f"{i}: {name}")

target_ids = []
for i, name in enumerate(names_list):
    n = str(name).lower().replace("-", "_").replace(" ", "_")
    if "helmet" in n or "hardhat" in n or "hard_hat" in n:
        target_ids.append(i)

if not target_ids:
    raise ValueError("Helmet/hardhat class not found in data.yaml names")

print(f"Keeping source class ids: {target_ids}")

if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

OUT_ROOT.mkdir(parents=True, exist_ok=True)

splits = {
    "train": data.get("train"),
    "valid": data.get("val") or data.get("valid"),
    "test": data.get("test"),
}

image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def find_images_dir(split_name, split_value):
    candidates = []

    if split_value:
        p = Path(split_value)

        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((YAML_DIR / p).resolve())
            candidates.append((RAW_ROOT / p).resolve())

            # Roboflow sometimes writes ../train/images even when train is inside extracted root
            value_str = str(split_value).replace("\\", "/")
            if "train/images" in value_str:
                candidates.append(RAW_ROOT / "train" / "images")
            if "valid/images" in value_str or "val/images" in value_str:
                candidates.append(RAW_ROOT / "valid" / "images")
                candidates.append(RAW_ROOT / "val" / "images")
            if "test/images" in value_str:
                candidates.append(RAW_ROOT / "test" / "images")

    # Fallbacks
    candidates.append(RAW_ROOT / split_name / "images")

    if split_name == "valid":
        candidates.append(RAW_ROOT / "val" / "images")

    for c in candidates:
        if c.exists():
            return c.resolve()

    print(f"Could not find images dir for split={split_name}")
    print("Tried:")
    for c in candidates:
        print(f"  - {c}")

    return None

def find_labels_dir(img_dir):
    candidates = []

    s = str(img_dir)
    candidates.append(Path(s.replace("\\images", "\\labels").replace("/images", "/labels")))
    candidates.append(img_dir.parent / "labels")

    for c in candidates:
        if c.exists():
            return c.resolve()

    print(f"Could not find labels dir for images dir: {img_dir}")
    print("Tried:")
    for c in candidates:
        print(f"  - {c}")

    return None

total_copied = 0

for split, split_value in splits.items():
    img_dir = find_images_dir(split, split_value)

    if img_dir is None:
        continue

    label_dir = find_labels_dir(img_dir)

    if label_dir is None:
        continue

    out_img_dir = OUT_ROOT / split / "images"
    out_lbl_dir = OUT_ROOT / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    split_count = 0
    object_count = 0

    for img_path in img_dir.rglob("*"):
        if img_path.suffix.lower() not in image_exts:
            continue

        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        new_lines = []

        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))

            if cls_id in target_ids:
                parts[0] = "0"
                new_lines.append(" ".join(parts))

        if not new_lines:
            continue

        shutil.copy2(img_path, out_img_dir / img_path.name)
        (out_lbl_dir / f"{img_path.stem}.txt").write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        split_count += 1
        object_count += len(new_lines)

    print(f"{split}: copied {split_count} images, {object_count} helmet objects")
    total_copied += split_count

if total_copied == 0:
    raise RuntimeError("No helmet images were copied. Check dataset structure and class names.")

data_out = {
    "path": str(OUT_ROOT).replace("\\", "/"),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": {0: "helmet"},
}

with open(OUT_ROOT / "data.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(data_out, f, allow_unicode=True, sort_keys=False)

print()
print("DONE")
print(f"New dataset: {OUT_ROOT}")
print(f"Total images with helmet: {total_copied}")
print(f"New data.yaml: {OUT_ROOT / 'data.yaml'}")
