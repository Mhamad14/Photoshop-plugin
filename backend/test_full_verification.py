import os
import sys
sys.path.append('backend')
import cv2
import numpy as np
from PIL import Image

from face_segmenter import segment_face_skin
from pimple_detector_v2 import detect_pimple_candidates
from spot_healer import spot_healing_brush_inpaint
from server import anchor_corners_rgba

im_path = r"C:\Users\Mhamad\.gemini\antigravity-ide\brain\25b7ced4-5e26-40dc-92bc-5c3a987aaf3b\.user_uploaded\media_1788725460256.jpg"
img_bgr = cv2.imread(im_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w, _ = img_rgb.shape

print(f"[TEST 1] Processing couple image {w}x{h}...")

# 1. Skin segmentation
skin_mask, meta = segment_face_skin(img_rgb, feather_radius=3)
faces_count = meta.get("faces_detected", 0)
skin_pct = meta.get("skin_percentage", 0.0)
print(f" -> Faces detected: {faces_count}")
print(f" -> Skin percentage: {skin_pct:.2f}%")
print(f" -> Classes present: {meta.get('classes_present')}")

# Verify no skin on woman's dress (dress is in bottom half y > 500, x in 100..400)
dress_region = skin_mask[550:800, 100:350]
dress_skin_pixels = np.sum(dress_region > 128)
print(f" -> Dress region false skin pixels: {dress_skin_pixels} (expected: 0)")
assert dress_skin_pixels == 0, "Error: Dress region contains false skin pixels!"

# 2. Blemish detection
blobs, pimple_mask = detect_pimple_candidates(img_rgb, skin_mask, sensitivity=0.5)
print(f"[TEST 2] Detected {len(blobs)} blemish spots:")
for i, b in enumerate(blobs):
    cx, cy = b["centroid"]
    r = b["radius"]
    conf = b.get("confidence", 0)
    print(f"    Spot #{i+1}: ({cx:.1f}, {cy:.1f}), r={r}, conf={conf:.2f}")
    # Verify all spots are strictly within facial skin
    assert skin_mask[int(cy), int(cx)] > 0, f"Error: Spot at ({cx}, {cy}) is outside verified skin!"

# 3. Spot Healing Inpainting
lama = None
try:
    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()
except Exception as e:
    print("LaMa note:", e)

healed_rgb, alpha = spot_healing_brush_inpaint(
    img_rgb=img_rgb,
    blobs=blobs,
    lama_model=lama,
    heal_mode="full_inpaint",
    texture_blend=0.30,
    feather_radius=3,
    grain_intensity=0.03,
    skin_mask=skin_mask
)

print(f"[TEST 3] Spot healing completed successfully.")

# 4. RGBA Transparent Patch Generation with Photoshop Corner Anchors
r = Image.fromarray(healed_rgb[:, :, 0])
g = Image.fromarray(healed_rgb[:, :, 1])
b = Image.fromarray(healed_rgb[:, :, 2])
a = Image.fromarray(alpha)
rgba_raw = Image.merge("RGBA", (r, g, b, a))
rgba_anchored = anchor_corners_rgba(rgba_raw)

# Verify corner anchors
pix = rgba_anchored.load()
assert pix[0, 0][3] >= 1, "Top-left anchor missing"
assert pix[w-1, 0][3] >= 1, "Top-right anchor missing"
assert pix[0, h-1][3] >= 1, "Bottom-left anchor missing"
assert pix[w-1, h-1][3] >= 1, "Bottom-right anchor missing"

# Check bounding box
bbox = rgba_anchored.getbbox()
print(f"[TEST 4] Anchored RGBA bounding box: {bbox} (full image: (0, 0, {w}, {h}))")
assert bbox == (0, 0, w, h), f"BBox mismatch: {bbox} != (0, 0, {w}, {h})"

# Save test outputs
cv2.imwrite("test_verified_healed.png", cv2.cvtColor(healed_rgb, cv2.COLOR_RGB2BGR))
rgba_anchored.save("test_verified_patch.png", format="PNG")
print("\nALL VERIFICATION TESTS PASSED PERFECTLY!")
