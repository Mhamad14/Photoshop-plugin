"""Validate a portrait-retouch YOLO segmentation dataset before training."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = ("heal_blemish", "tone_irregularity", "preserve_mark")


def parse_label(label_path: Path) -> tuple[Counter, list[str]]:
    counts: Counter = Counter()
    errors: list[str] = []

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = raw_line.split()
        if not values:
            continue
        if len(values) < 7 or (len(values) - 1) % 2:
            errors.append(f"{label_path}:{line_number}: expected class plus at least 3 polygon points")
            continue
        try:
            class_id = int(values[0])
            points = [float(value) for value in values[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_number}: non-numeric label value")
            continue
        if class_id not in range(len(CLASS_NAMES)):
            errors.append(f"{label_path}:{line_number}: unknown class {class_id}")
        if any(point < 0.0 or point > 1.0 for point in points):
            errors.append(f"{label_path}:{line_number}: polygon coordinates must be normalized 0..1")
        counts[class_id] += 1

    return counts, errors


def image_stems(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {path.stem for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}


def validate_split(root: Path, split: str) -> tuple[Counter, list[str]]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    errors: list[str] = []
    counts: Counter = Counter()

    images = image_stems(image_dir)
    labels = {path.stem for path in label_dir.glob("*.txt")} if label_dir.exists() else set()
    if not images:
        errors.append(f"{split}: no images found in {image_dir}")
    if not label_dir.exists():
        errors.append(f"{split}: missing label directory {label_dir}")
        return counts, errors

    for stem in sorted(images - labels):
        errors.append(f"{split}: missing label for image '{stem}'")
    for stem in sorted(labels - images):
        errors.append(f"{split}: label without image '{stem}'")
    for label_path in sorted(label_dir.glob("*.txt")):
        parsed_counts, parsed_errors = parse_label(label_path)
        counts.update(parsed_counts)
        errors.extend(parsed_errors)

    return counts, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate YOLO segmentation labels for portrait retouching.")
    parser.add_argument("--data", type=Path, default=Path("data/retouch_skin"), help="Dataset root")
    args = parser.parse_args()

    counts: Counter = Counter()
    errors: list[str] = []
    for split in ("train", "val", "test"):
        split_counts, split_errors = validate_split(args.data, split)
        counts.update(split_counts)
        errors.extend(split_errors)

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {counts[class_id]} instances")

    if errors:
        print("\nDataset validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1

    if counts[0] < 100:
        print("\nWarning: fewer than 100 heal_blemish instances; expect poor generalization.")
    print("\nDataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
