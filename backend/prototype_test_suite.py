import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from face_segmenter import segment_face_skin
from pimple_detector_v2 import detect_pimple_candidates, blobs_to_mask
from skin_toner import calculate_tone_lift, create_lightened_rgba_patch

def run_prototype_benchmark(image_path: str, output_dir: str = "benchmark_results"):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"\n=======================================================")
    print(f" [PROTOTYPE BENCHMARK] Processing: {image_path}")
    print(f"=======================================================")
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    print(f"Image Resolution: {w}x{h}")
    
    # 1. Benchmark Layer 1: Face/Skin Segmentation
    t0 = time.time()
    skin_mask, skin_meta = segment_face_skin(img_rgb, include_neck=True, feather_radius=3)
    t_layer1 = (time.time() - t0) * 1000
    print(f"[Layer 1] Face/Skin Segmentation completed in {t_layer1:.1f}ms")
    print(f"         Skin coverage: {skin_meta['skin_percentage']:.1f}% ({skin_meta['skin_pixel_count']} px)")
    print(f"         Sampled Base Tone RGB: {skin_meta['base_tone_rgb']}, LAB: {[round(x,1) for x in skin_meta['base_tone_lab']]}")
    
    # 2. Benchmark Layer 2: Hybrid Pimple Detection
    t0 = time.time()
    blobs, pimple_mask = detect_pimple_candidates(img_rgb, skin_mask, sensitivity=0.5)
    t_layer2 = (time.time() - t0) * 1000
    print(f"[Layer 2] Pimple Detection completed in {t_layer2:.1f}ms")
    print(f"         Detected Candidate Blobs: {len(blobs)}")
    for b in blobs[:5]:
        print(f"           - Blob #{b['id']}: center={b['centroid']}, radius={b['radius']}px, conf={b['confidence']}")
    if len(blobs) > 5:
        print(f"           ... and {len(blobs) - 5} more blobs.")

    # 3. Benchmark Layer 5 (Action 2): Skin Lightening Tone Lift
    t0 = time.time()
    lightened_rgb, lightened_alpha = calculate_tone_lift(
        img_rgb, skin_mask, strength=0.35, base_tone_lab=skin_meta['base_tone_lab'], feather_radius=4
    )
    t_layer5 = (time.time() - t0) * 1000
    print(f"[Layer 5] Skin Lightening calculation completed in {t_layer5:.1f}ms")

    # 4. Generate Visual Overlay Inspection Sheet (4-panel comparison)
    # Panel 1: Original Image
    p1 = img_rgb.copy()
    
    # Panel 2: Skin Mask Overlay (Cyan tint over excluded eyes/brows/lips)
    p2 = img_rgb.copy().astype(np.float32)
    skin_norm = (skin_mask.astype(np.float32) / 255.0)[:, :, None]
    cyan_overlay = np.array([0, 180, 255], dtype=np.float32)
    p2 = (p2 * (1.0 - skin_norm * 0.35) + cyan_overlay * (skin_norm * 0.35)).clip(0, 255).astype(np.uint8)

    # Panel 3: Pimple Mask Overlay (Red glowing circles + bounding boxes)
    p3 = img_rgb.copy()
    for b in blobs:
        cx, cy = int(b["centroid"][0]), int(b["centroid"][1])
        r = int(b["radius"])
        conf = b["confidence"]
        # Draw red circle
        cv2.circle(p3, (cx, cy), r, (255, 40, 40), 2)
        # Draw small center point
        cv2.circle(p3, (cx, cy), 1, (255, 255, 0), -1)

    # Panel 4: Lightened Skin Preview
    p4 = lightened_rgb.copy()

    # Combine into a 2x2 grid
    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bot_row])

    # Convert to PIL for labeled export
    pil_grid = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil_grid)
    
    # Add panel labels
    labels = [
        ("1. Original Portrait", 15, 15),
        ("2. Layer 1: Skin Mask (BiSeNet)", w + 15, 15),
        (f"3. Layer 2: Pimple Detections ({len(blobs)} blobs)", 15, h + 15),
        ("4. Layer 5: Relative Skin Lightening", w + 15, h + 15)
    ]
    for text, lx, ly in labels:
        draw.rectangle([lx - 5, ly - 3, lx + len(text)*9 + 10, ly + 20], fill=(0, 0, 0, 180))
        draw.text((lx, ly), text, fill=(255, 255, 255))

    out_file = os.path.join(output_dir, f"{basename}_benchmark_grid.png")
    pil_grid.save(out_file)
    print(f"\nSaved Visual Benchmark Grid -> {out_file}")
    print(f"Total Pipeline Analysis Latency: {t_layer1 + t_layer2 + t_layer5:.1f}ms")
    print(f"=======================================================\n")


if __name__ == "__main__":
    candidates = ["test_healed_full_portrait.png", "../test_healed_full_portrait.png"]
    found = False
    for img_p in candidates:
        if os.path.exists(img_p):
            run_prototype_benchmark(img_p)
            found = True
            break
    if not found:
        print("Sample image not found.")
