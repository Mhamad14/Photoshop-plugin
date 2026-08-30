# YOLOv8-Seg Fine-Tuning: Portrait Retouching

This is a training scaffold, not a production detector. Do not wire a model into the plugin until it passes the held-out photographer test set.

## Product Goal

For a photographer, the model must find localized skin issues that are typically retouched quickly, then leave a toggleable mask for human approval:

- `heal_blemish` (class `0`): acne, pustules, whiteheads, blackheads, milia, razor bumps, and isolated temporary spots.
- `tone_irregularity` (class `1`): localized PIH, acne scars, red patches, or dark spots appropriate for spot healing or local tone work.
- `preserve_mark` (class `2`): beauty moles, intentional freckles, tattoos, or other identity marks that must default to **ignored** by the plugin.

Annotate only small, local regions. Global uneven tone and normal pore texture belong to the Smooth Skin and Tone Lift stages, not this detector.

## Dataset Layout

Create this structure under `backend/training/data/retouch_skin/`:

```text
images/train/photo_001.jpg
images/val/photo_101.jpg
images/test/photo_151.jpg
labels/train/photo_001.txt
labels/val/photo_101.txt
labels/test/photo_151.txt
```

Each label file uses YOLO segmentation format:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

Coordinates are normalized `0..1`. Use polygon masks around the actual imperfection, not a large generic circle. Empty `.txt` files are valid for clean portraits.

## Collection and Annotation Rules

1. Start with 500-1,000 portraits and aim for 5,000+ labelled instances. Include RAW exports, JPEGs, phone shots, studio light, hard flash, backlight, makeup/no-makeup, facial hair, all skin tones, skin conditions, and face angles.
2. Split by **person**, not by image. The same model/person must never appear in train and test.
3. Reserve at least 15% of people for `test` and never tune thresholds against that set.
4. Add clean photos with empty labels. They are essential for lowering false positives.
5. Label beauty marks/freckles as `preserve_mark`; do not silently omit them. This teaches the model what to leave alone.
6. Reject labels that touch eyes, eyebrows, lips, nostrils, hair, beard edges, or face boundaries.

## Run

From `backend/training/`:

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 validate_dataset.py --data data/retouch_skin
py -3.13 train_yolo_seg.py --epochs 120 --imgsz 1024 --device 0 --export-onnx
```

For CPU-only experimentation, use `--device cpu`; training will be slow. An NVIDIA GPU with 8 GB+ VRAM is recommended.

## ACNE04 Quick Start (already imported)

The ACNE04-v2 dataset (1,204 images / 32,443 expert acne annotations, MICCAI 2024) is
imported into `data/retouch_skin/`. To re-import or re-split:

```powershell
py -3.13 import_acne04.py            # requires _raw_acne04/ (annotations + JPEGImages)
```

Full training on CPU (no GPU on this machine) — recommended first run:

```powershell
cd backend\training
py -3.13 train_yolo_seg.py --model yolov8n-seg.pt --epochs 60 --imgsz 640 --batch 16 --device cpu --name acne04_v1
```

Roughly 3-6 hours on an i7 CPU. When finished, deploy with:

```powershell
copy backend\training\runs\segment\runs\acne04_v1\weights\best.pt backend\models\retouch_yolov8_seg.pt
```

The server automatically uses the model at `backend/models/retouch_yolov8_seg.pt` as
the primary detector (classical CV stays as fallback). No server code changes needed.
Higher `sensitivity` in the panel lowers the YOLO confidence threshold.

**License note:** ACNE04 is released for ACADEMIC USE ONLY (Wu et al., ICCV 2019).
Fine for testing and learning; a commercial plugin needs the authors' permission or
a commercially-licensed / self-collected dataset (e.g. via the panel's Training Mode).

## Acceptance Gate

Do not call the model “photographer ready” based on mAP alone. Review its predictions with a retoucher across the locked test set:

- Fewer than 5% of `preserve_mark` instances should be enabled as removable by default.
- At least 85% recall for clearly retouchable, visible localized blemishes.
- All predicted masks must stay inside skin and exclude eye/lip/hair boundaries.
- Review difficult photos separately: dark skin under flash, heavy makeup, facial hair, severe acne, freckles, and motion blur.
- The Photoshop panel must still let the photographer toggle each detection before Apply.

## Deployment Handoff

After acceptance, export `best.pt` to ONNX and copy it to `backend/models/retouch_yolov8_seg.onnx`. Then integrate it as the local proposal stage in `pimple_detector_v2.py`:

1. Run YOLO only inside `skin_mask`.
2. Convert `heal_blemish` and approved `tone_irregularity` instances into existing blob objects.
3. Convert `preserve_mark` instances to inactive blobs or use them to suppress overlapping heuristic candidates.
4. Keep the existing click-to-add, click-to-ignore, and Gemini verification paths as refinement tools.

This keeps inference local and fast while the photographer retains the final decision.
