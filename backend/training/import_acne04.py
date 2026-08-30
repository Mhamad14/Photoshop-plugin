"""Import the ACNE04-v2 dataset (expert acne circle annotations) into the
local YOLO training dataset under backend/training/data/retouch_skin/.

Source layout expected (produced by the earlier download steps):
  backend/training/_raw_acne04/acne04v2-main/Acne04-v2_annotations.json
  backend/training/_raw_acne04/images/Classification/JPEGImages/levleX_N.jpg

Each acne circle (center + radius) becomes a 16-vertex YOLO segmentation
polygon with class 0 (heal_blemish).

NOTE: ACNE04 is released for ACADEMIC USE ONLY (Wu et al., ICCV 2019).
Do not ship a model trained on it inside a commercial product without
permission from the dataset authors.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
RAW_ROOT = TRAINING_ROOT / "_raw_acne04"
ANNOTATIONS = RAW_ROOT / "acne04v2-main" / "Acne04-v2_annotations.json"
IMAGES_DIR = RAW_ROOT / "images" / "Classification" / "JPEGImages"
DATASET_ROOT = TRAINING_ROOT / "data" / "retouch_skin"

CLASS_ID = 0          # heal_blemish
POLYGON_VERTICES = 16
MIN_RADIUS_PX = 2.0   # skip speckle annotations


def circle_to_polygon(cx: float, cy: float, r: float, w: int, h: int) -> list[float]:
    points: list[float] = []
    for i in range(POLYGON_VERTICES):
        angle = (2.0 * math.pi * i) / POLYGON_VERTICES
        x = min(max(cx + r * math.cos(angle), 0.0), float(w))
        y = min(max(cy + r * math.sin(angle), 0.0), float(h))
        points.extend((x / w, y / h))
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description="Import ACNE04-v2 into the retouch_skin YOLO dataset")
    parser.add_argument("--train", type=float, default=0.7, help="train fraction")
    parser.add_argument("--val", type=float, default=0.15, help="val fraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear", action="store_true", help="clear existing imported images first")
    args = parser.parse_args()

    if not ANNOTATIONS.exists():
        print(f"Missing annotations: {ANNOTATIONS}")
        return 1
    if not IMAGES_DIR.exists():
        print(f"Missing images: {IMAGES_DIR}")
        return 1

    with ANNOTATIONS.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    by_image: dict[int, list] = {}
    for ann in data["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    available = [img for img in images.values() if (IMAGES_DIR / img["file_name"]).exists()]
    print(f"annotated images: {len(images)} | present on disk: {len(available)}")

    rng = random.Random(args.seed)
    rng.shuffle(available)
    n = len(available)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    splits = {
        "train": available[:n_train],
        "val": available[n_train:n_train + n_val],
        "test": available[n_train + n_val:],
    }

    for split in splits:
        (DATASET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / "metadata" / split).mkdir(parents=True, exist_ok=True)

    stats = {s: {"images": 0, "labels": 0, "skipped": 0} for s in splits}
    for split, imgs in splits.items():
        for img in imgs:
            w, h = int(img["width"]), int(img["height"])
            lines = []
            for ann in by_image.get(img["id"], []):
                r = float(ann.get("radius", 0))
                if r < MIN_RADIUS_PX:
                    stats[split]["skipped"] += 1
                    continue
                cx, cy = ann["coordinates"][0], ann["coordinates"][1]
                poly = circle_to_polygon(float(cx), float(cy), r, w, h)
                lines.append(" ".join([str(CLASS_ID)] + [f"{v:.6f}" for v in poly]))

            src = IMAGES_DIR / img["file_name"]
            shutil.copy2(src, DATASET_ROOT / "images" / split / src.name)
            (DATASET_ROOT / "labels" / split / f"{src.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            (DATASET_ROOT / "metadata" / split / f"{src.stem}.json").write_text(json.dumps({
                "source": "acne04_v2",
                "file_name": src.name,
                "width": w,
                "height": h,
                "instances": len(lines),
            }), encoding="utf-8")
            stats[split]["images"] += 1
            stats[split]["labels"] += len(lines)

    for split, s in stats.items():
        print(f"{split}: {s['images']} images | {s['labels']} acne labels | {s['skipped']} speckles skipped")
    total_labels = sum(s["labels"] for s in stats.values())
    print(f"TOTAL: {sum(s['images'] for s in stats.values())} images, {total_labels} labels")
    print(f"dataset ready at: {DATASET_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
