"""Fine-tune YOLOv8-seg for photographer-grade portrait retouch detection."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a portrait-retouch YOLOv8 segmentation model.")
    parser.add_argument("--data", type=Path, default=Path("dataset.yaml"), help="YOLO dataset YAML")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Pretrained YOLOv8 segmentation checkpoint")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=1024, help="Use 1024px to retain small facial spots")
    parser.add_argument("--batch", type=int, default=-1, help="-1 uses automatic GPU batch sizing")
    parser.add_argument("--device", default=None, help="CUDA device, 'cpu', or omit for Ultralytics auto-select")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="retouch_yolov8_seg")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-onnx", action="store_true", help="Export best checkpoint to ONNX after training")
    args = parser.parse_args()

    validator = Path(__file__).with_name("validate_dataset.py")
    validation = subprocess.run([sys.executable, str(validator), "--data", str(args.data.parent / "data" / "retouch_skin")])
    if validation.returncode:
        return validation.returncode

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Missing training dependency. Run: pip install -r backend/training/requirements.txt")
        return 1

    model = YOLO(args.model)
    train_options = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "pretrained": True,
        "resume": args.resume,
        # Conservative augmentation: portraits must remain anatomically realistic.
        "degrees": 4.0,
        "translate": 0.04,
        "scale": 0.20,
        "fliplr": 0.5,
        "mosaic": 0.10,
        "close_mosaic": 15,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.01,
        "hsv_s": 0.20,
        "hsv_v": 0.15,
        "patience": 30,
        "seed": 42,
    }
    if args.device is not None:
        train_options["device"] = args.device

    model.train(**train_options)
    best = args.project / args.name / "weights" / "best.pt"
    if not best.exists():
        print(f"Training finished but {best} was not found.")
        return 1

    metrics = YOLO(str(best)).val(data=str(args.data), imgsz=args.imgsz)
    print(f"Validation complete: {metrics}")

    if args.export_onnx:
        onnx_path = YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, dynamic=True, simplify=True)
        print(f"ONNX export: {onnx_path}")

    print(f"Best checkpoint: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
