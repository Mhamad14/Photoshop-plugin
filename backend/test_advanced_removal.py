import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from face_segmenter import segment_face_skin
from pimple_detector_v2 import detect_pimple_candidates, blobs_to_mask
from server import neutralize_erythema, blend_skin_texture, inpaint_with_context_tiling

def test_advanced_removal_pipeline(image_path: str = "test_healed_full_portrait.png", output_dir: str = "benchmark_results"):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\n=======================================================")
    print(f" [ADVANCED PIMPLE DETECTION & REMOVAL TEST]")
    print(f" Image: {image_path}")
    print(f"=======================================================")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not read image at {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    print(f"Resolution: {w}x{h} px")

    # 1. Skin segmentation
    t0 = time.time()
    skin_mask, skin_meta = segment_face_skin(img_rgb, include_neck=True, feather_radius=3)
    t_skin = (time.time() - t0) * 1000
    print(f"[1/4] Skin segmentation: {t_skin:.1f}ms (Coverage: {skin_meta['skin_percentage']:.1f}%)")

    # 2. Advanced Multi-Scale Adaptive Pimple Detection
    t0 = time.time()
    blobs, pimple_mask = detect_pimple_candidates(img_rgb, skin_mask, sensitivity=0.55)
    t_detect = (time.time() - t0) * 1000
    print(f"[2/4] Advanced pimple detection: {t_detect:.1f}ms ({len(blobs)} blemish candidates)")

    # 3. Dermatological Erythema Pre-Neutralization
    t0 = time.time()
    clean_rgb = neutralize_erythema(img_rgb, pimple_mask)
    t_neut = (time.time() - t0) * 1000
    print(f"[3/4] Erythema neutralization: {t_neut:.1f}ms")

    # 4. Inpainting with Simple-LaMa (or fallback)
    t0 = time.time()
    try:
        from simple_lama_inpainting import SimpleLama
        lama_model = SimpleLama()
        print("      Simple-LaMa model loaded successfully.")
    except Exception as e:
        print(f"      LaMa model load error: {e}. Using CV Inpainting fallback.")
        lama_model = None

    if lama_model is not None and np.sum(pimple_mask > 0) > 0:
        inpainted_np = inpaint_with_context_tiling(
            model=lama_model,
            img_rgb=clean_rgb,
            mask_gray=pimple_mask,
            max_tile_size=768,
            context_pad=80
        )
    else:
        inpainted_np = cv2.inpaint(clean_rgb, pimple_mask, 5, cv2.INPAINT_TELEA)

    # 5. Pore-Preserving Texture Synthesis & Illumination Matching
    final_healed = blend_skin_texture(
        original_img=img_rgb,
        inpainted_img=inpainted_np,
        mask_gray=pimple_mask,
        texture_blend=0.35,
        grain_intensity=0.03
    )
    t_heal = (time.time() - t0) * 1000
    print(f"[4/4] Inpainting & healthy pore texture synthesis: {t_heal:.1f}ms")

    # 6. Generate Treatment Mode Variations
    # Mode B: Calmed Redness (Erythema soothing only)
    calmed_rgb = clean_rgb.copy()

    # Mode C: Bump Flattened (Frequency separation smoothing of raised blemish)
    orig_f = img_rgb.astype(np.float32)
    blurred_low = cv2.GaussianBlur(orig_f, (15, 15), 0)
    high_freq = orig_f - blurred_low
    clean_f = clean_rgb.astype(np.float32)
    mask_f = (pimple_mask.astype(np.float32) / 255.0)[:, :, None]
    flattened_rgb = np.clip(clean_f + high_freq * (1.0 - mask_f * 0.7), 0, 255).astype(np.uint8)

    # 7. Create Comprehensive Visual Comparison Grid
    # Panel 1: Original Image
    p1 = img_rgb.copy()

    # Panel 2: Organic Detected Blemishes Overlay
    p2 = img_rgb.copy()
    mask_overlay = (pimple_mask > 0).astype(np.float32)[:, :, None]
    red_tint = np.array([255, 40, 40], dtype=np.float32)
    p2 = (p2.astype(np.float32) * (1.0 - mask_overlay * 0.45) + red_tint * (mask_overlay * 0.45)).clip(0, 255).astype(np.uint8)
    for b in blobs:
        cx, cy = int(b["centroid"][0]), int(b["centroid"][1])
        r = int(b["radius"])
        cv2.circle(p2, (cx, cy), r, (255, 60, 60), 2)
        cv2.circle(p2, (cx, cy), 1, (255, 255, 0), -1)

    # Panel 3: Calmed Redness Mode
    p3 = calmed_rgb.copy()

    # Panel 4: Full AI Healed with Pore Restoration
    p4 = final_healed.copy()

    # Combine 2x2 grid
    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bot_row])

    pil_grid = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil_grid)

    labels = [
        ("1. Original Input", 15, 15),
        (f"2. Organic Blemish Detections ({len(blobs)} spots)", w + 15, 15),
        ("3. Calmed Redness (Soothe Inflammation)", 15, h + 15),
        ("4. Full AI Inpaint + Pore Preservation", w + 15, h + 15)
    ]
    for text, lx, ly in labels:
        draw.rectangle([lx - 5, ly - 3, lx + len(text)*10 + 12, ly + 22], fill=(0, 0, 0, 200))
        draw.text((lx, ly), text, fill=(255, 255, 255))

    out_file = os.path.join(output_dir, f"{basename}_advanced_removal_grid.png")
    pil_grid.save(out_file)
    print(f"\n[SUCCESS] Visual Comparison Grid exported -> {out_file}")
    print(f"Total processing latency: {t_skin + t_detect + t_neut + t_heal:.1f}ms")
    print(f"=======================================================\n")

if __name__ == "__main__":
    test_path = "test_healed_full_portrait.png"
    if os.path.exists(test_path):
        test_advanced_removal_pipeline(test_path)
    else:
        print(f"File {test_path} not found in workspace.")
