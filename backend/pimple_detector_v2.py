import io
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger("pimple_detector_v2")


def estimate_clicked_blemish_radius(
    img_rgb: np.ndarray,
    cx: int,
    cy: int,
    min_r: int = 3,
    max_r: int = 24,
    default_r: int = 8
) -> int:
    """
    Intelligent Click-to-Heal Auto-Radius Estimator:
    
    When a user clicks a point on the canvas, this function measures the radial color/gradient
    inflection from (cx, cy) outward along 8 compass rays to dynamically fit the exact diameter
    of the clicked blemish without requiring manual slider adjustment.
    """
    h, w, _ = img_rgb.shape
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return default_r

    half = 24
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half + 1), min(h, cy + half + 1)
    
    patch_rgb = img_rgb[y1:y2, x1:x2]
    if patch_rgb.size == 0 or patch_rgb.shape[0] < 7 or patch_rgb.shape[1] < 7:
        return default_r

    patch_lab = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    center_px = patch_lab[cy - y1, cx - x1]

    delta_e = np.sqrt(np.sum((patch_lab - center_px) ** 2, axis=2))
    
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    detected_radii = []
    
    for ang in angles:
        rad = math.radians(ang)
        dx = math.cos(rad)
        dy = math.sin(rad)
        
        last_diff = 0.0
        found_r = default_r
        for step in range(2, half - 1):
            px = int(round((cx - x1) + dx * step))
            py = int(round((cy - y1) + dy * step))
            if px < 0 or py < 0 or px >= patch_rgb.shape[1] or py >= patch_rgb.shape[0]:
                break
                
            cur_diff = delta_e[py, px]
            if step >= 3 and cur_diff > 14.0 and (cur_diff - last_diff) < 1.0:
                found_r = step
                break
            last_diff = cur_diff
            
        detected_radii.append(found_r)

    if detected_radii:
        estimated_r = int(np.percentile(detected_radii, 75)) + 2
        return max(min_r, min(max_r, estimated_r))
    
    return default_r


def compute_multi_channel_blemish_energy(
    img_rgb: np.ndarray,
    skin_binary: np.ndarray,
    sensitivity: float = 0.5
) -> np.ndarray:
    """
    Multi-Channel Dermatological Blemish Energy Extractor:
    
    Combines 5 clinical signatures of acne blemishes with Hair & Beard Stubble Rejection:
    1. Erythema Signature: Elevated localized delta a* redness (papules, inflamed acne, rosacea).
    2. Whitehead Core Signature: Focal micro-specular luminance peak (L*) with steep radial gradient.
    3. Blackhead / Clogged Pore Signature: Focal dark comedone core with high surrounding gradient.
    4. Post-Inflammatory Hyperpigmentation (PIH): Melanin scar spots on Fitzpatrick Types III-VI.
    5. Micro-Texture Disorder: High local variance in high-pass spatial frequency.
    6. Linear Structure Tensor Filter: Distinguishes linear hair filaments & beard stubble from circular acne.
    """
    h, w, _ = img_rgb.shape
    skin_float = skin_binary.astype(np.float32)

    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(img_lab)

    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)

    # 1. Baseline healthy skin estimation (broad Gaussian over skin region)
    sigma_broad = max(18.0, min(h, w) * 0.06)
    norm_w = cv2.GaussianBlur(skin_float, (0, 0), sigma_broad) + 1e-5
    
    baseline_a = cv2.GaussianBlur(a_chan * skin_float, (0, 0), sigma_broad) / norm_w
    baseline_l = cv2.GaussianBlur(l_chan * skin_float, (0, 0), sigma_broad) / norm_w

    # 2. Channel 1: Erythema Delta (Redness Inflammation)
    norm_redness = (r - g) / (r + g + 12.0)
    delta_a = np.maximum(0.0, a_chan - baseline_a)
    erythema_energy = (delta_a * 0.65) + (norm_redness * 45.0 * 0.35)

    # 3. Channel 2: Whitehead Specular Core (Local peak in L*)
    sigma_micro = 2.0
    l_micro = cv2.GaussianBlur(l_chan * skin_float, (0, 0), sigma_micro) / (cv2.GaussianBlur(skin_float, (0, 0), sigma_micro) + 1e-5)
    whitehead_energy = np.maximum(0.0, l_micro - baseline_l)
    whitehead_energy = np.clip(whitehead_energy * 0.45, 0.0, 30.0)

    # 4. Channel 3: Post-Inflammatory Hyperpigmentation (PIH) & Comedones (Focal L* depression)
    pih_energy = np.maximum(0.0, baseline_l - l_micro)
    pih_energy = np.clip(pih_energy * 0.40, 0.0, 25.0)

    # 5. Channel 4: Local Texture Disorder (High-pass variance)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    high_pass = np.abs(gray - cv2.GaussianBlur(gray, (5, 5), 0))
    texture_disorder = cv2.GaussianBlur(high_pass, (3, 3), 0)

    # 6. Multi-Scale Difference of Gaussians (DoG) for isolated focal spots (2px to 20px)
    scales = [(1.2, 2.4), (2.0, 4.2), (3.5, 7.5), (5.5, 12.0), (8.0, 16.0)]
    dog_maps = []
    
    combined_signal = (erythema_energy * 0.50) + (whitehead_energy * 0.20) + (pih_energy * 0.15) + (texture_disorder * 0.15)

    for s1, s2 in scales:
        g1 = cv2.GaussianBlur(combined_signal, (0, 0), s1)
        g2 = cv2.GaussianBlur(combined_signal, (0, 0), s2)
        dog = np.maximum(0.0, g1 - g2)
        dog_maps.append(dog)

    spot_energy = np.maximum.reduce(dog_maps)

    # 7. Structure Tensor Linearity Filter: Suppresses linear beard stubble & hair roots
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    j_xx = cv2.GaussianBlur(gx * gx, (5, 5), 0)
    j_yy = cv2.GaussianBlur(gy * gy, (5, 5), 0)
    j_xy = cv2.GaussianBlur(gx * gy, (5, 5), 0)
    trace = j_xx + j_yy
    disc = np.sqrt(np.maximum(0.0, (j_xx - j_yy) ** 2 + 4.0 * (j_xy ** 2)))
    lambda1 = (trace + disc) / 2.0
    lambda2 = (trace - disc) / 2.0
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-4)
    hair_attenuation = np.clip(1.0 - (coherence - 0.38) * 2.2, 0.05, 1.0)
    spot_energy = spot_energy * hair_attenuation

    # 8. Erode skin margin to protect jawline/hairline boundaries
    kernel_skin_margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_inner = cv2.erode(skin_binary, kernel_skin_margin)
    spot_energy[skin_inner == 0] = 0.0

    # 9. Structural Edge Exclusion (protect sharp boundaries: lips, nose tip, eyelids)
    edges = cv2.Canny(gray.astype(np.uint8), 35, 95)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_exclusion = cv2.dilate(edges, kernel_edge)
    spot_energy[edge_exclusion > 0] *= 0.12

    return spot_energy


def detect_pimple_candidates(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    sensitivity: float = 0.5,
    min_radius: int = 2,
    max_radius: int = 25,
    gemini_api_key: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Precision Hybrid Pimple & Blemish Detection Pipeline.
    
    Detects:
    - Inflammatory acne (papules, pustules)
    - Whiteheads & blackheads
    - Post-inflammatory hyperpigmentation (PIH)
    - Razor bumps and focal redness
    
    Returns:
        blobs: List of candidate blob dicts {id, bbox, centroid, radius, confidence, active, label}
        binary_pimple_mask: uint8 2D array (0 or 255)
    """
    h, w, _ = img_rgb.shape
    skin_binary = (skin_mask > 128).astype(np.uint8)

    # Resolution-adaptive radius bounds: at high camera resolutions (4000px+)
    # a real blemish spans 100px+, so fixed pixel caps would only mask the
    # pimple core and leave its inflamed halo unhealed.
    min_dim = float(min(h, w))
    min_radius = max(min_radius, int(min_dim * 0.0015))
    max_radius = max(max_radius, int(min_dim * 0.02))
    
    if np.sum(skin_binary) < 100:
        logger.warning("Skin mask is nearly empty. Detection skipped.")
        return [], np.zeros((h, w), dtype=np.uint8)

    # Stage A: Multi-channel blemish energy with hair rejection
    spot_energy = compute_multi_channel_blemish_energy(img_rgb, skin_binary, sensitivity=sensitivity)

    # Stage B: Adaptive Thresholding
    skin_spots = spot_energy[skin_binary > 0]
    if len(skin_spots) == 0:
        return [], np.zeros((h, w), dtype=np.uint8)

    mean_val = float(np.mean(skin_spots))
    std_val = float(np.std(skin_spots))

    # Dynamic k threshold scaling: sensitivity 0.1 (strict) to 1.0 (sensitive)
    k = (1.0 - max(0.05, min(1.0, sensitivity))) * 1.8 + 0.15
    threshold = mean_val + (k * std_val)

    candidate_mask = (spot_energy > threshold).astype(np.uint8) * 255

    # Stage C: Morphological Blob Extraction & Shape Verification
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask)
    blobs: List[Dict[str, Any]] = []
    final_mask = np.zeros((h, w), dtype=np.uint8)
    blob_id = 1

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        if bw == 0 or bh == 0 or area < 2:
            continue

        aspect_ratio = float(bw) / float(bh)
        if aspect_ratio < 0.22 or aspect_ratio > 4.5:
            continue

        eff_radius = int(math.ceil(math.sqrt(area / math.pi)))
        if eff_radius < min_radius or eff_radius > max_radius:
            continue

        blob_pixels = spot_energy[labels == i]
        peak_val = float(np.max(blob_pixels)) if len(blob_pixels) > 0 else threshold
        conf = min(0.99, max(0.40, float((peak_val - threshold) / (std_val * 2.5 + 1e-5) * 0.45 + 0.55)))

        pad = max(3, int(eff_radius * 0.40))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        radius_with_margin = eff_radius + max(3, int(eff_radius * 0.55))

        blob_data = {
            "id": blob_id,
            "bbox": [x1, y1, x2, y2],
            "centroid": [round(cx, 1), round(cy, 1)],
            "radius": radius_with_margin,
            "area": area,
            "confidence": round(conf, 2),
            "active": True,
            "label": "pimple",
            "source": "auto_cv"
        }
        blobs.append(blob_data)

        cv2.circle(final_mask, (int(round(cx)), int(round(cy))), radius_with_margin, 255, -1)
        blob_id += 1

    # Morphological cluster fusion on candidate mask
    k_fuse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, k_fuse)

    # Optional Stage D: Gemini Vision VLM Verification / Augmentation
    if gemini_api_key:
        try:
            from gemini_detector import detect_blemishes_gemini
            logger.info("Running Gemini Vision VLM verification/augmentation...")
            pil_img = Image.fromarray(img_rgb)
            vlm_mask_pil = detect_blemishes_gemini(pil_img, api_key=gemini_api_key, dilate_pixels=3)
            if vlm_mask_pil is not None:
                vlm_np = np.array(vlm_mask_pil.convert("L"))
                vlm_num, vlm_labels, vlm_stats, vlm_centroids = cv2.connectedComponentsWithStats((vlm_np > 128).astype(np.uint8))
                for v in range(1, vlm_num):
                    vc_area = int(vlm_stats[v, cv2.CC_STAT_AREA])
                    if vc_area < 4:
                        continue
                    vcx, vcy = float(vlm_centroids[v][0]), float(vlm_centroids[v][1])
                    vr = int(math.ceil(math.sqrt(vc_area / math.pi))) + 3
                    
                    already_covered = any(
                        math.hypot(b["centroid"][0] - vcx, b["centroid"][1] - vcy) < (b["radius"] * 1.2)
                        for b in blobs
                    )
                    if not already_covered and skin_binary[int(vcy), int(vcx)] > 0:
                        vpad = max(2, int(vr * 0.3))
                        blobs.append({
                            "id": blob_id,
                            "bbox": [max(0, int(vcx - vr - vpad)), max(0, int(vcy - vr - vpad)),
                                     min(w, int(vcx + vr + vpad)), min(h, int(vcy + vr + vpad))],
                            "centroid": [round(vcx, 1), round(vcy, 1)],
                            "radius": vr,
                            "area": vc_area,
                            "confidence": 0.92,
                            "active": True,
                            "label": "pimple",
                            "source": "vlm_gemini"
                        })
                        cv2.circle(final_mask, (int(round(vcx)), int(round(vcy))), vr, 255, -1)
                        blob_id += 1
        except Exception as e:
            logger.warning(f"VLM verification warning: {e}")

    return blobs, final_mask


def blobs_to_mask(blobs: List[Dict[str, Any]], shape: Tuple[int, int], dilate_px: int = 0) -> np.ndarray:
    """
    Converts active blemish blobs into a 2D uint8 mask with Morphological Cluster Fusion.
    Fuses neighboring breakout acne spots into unified continuous organic contours.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for blob in blobs:
        if blob.get("active", True) is False:
            continue
        cx = int(round(blob.get("centroid", [blob.get("x", 0), blob.get("y", 0)])[0]))
        cy = int(round(blob.get("centroid", [blob.get("x", 0), blob.get("y", 0)])[1]))
        r = int(blob.get("radius", blob.get("r", 6)))
        # Full Spot Healing Brush coverage (envelopes the complete inflamed redness margin)
        expanded_r = max(4, int(r * 1.35) + 2)
        cv2.circle(mask, (cx, cy), expanded_r, 255, -1)

    # Morphological Cluster Fusion: bridges close breakout spots so no patchy un-healed gaps remain
    k_fuse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_fuse)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, kernel)

    return mask


def add_blob_at_point(
    blobs: Optional[List[Dict[str, Any]]] = None,
    point: Optional[Tuple[int, int]] = None,
    radius: int = 8,
    shape: Optional[Tuple[int, int]] = None,
    existing_blobs: Optional[List[Dict[str, Any]]] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    default_radius: int = 8,
    img_rgb: Optional[np.ndarray] = None,
    skin_mask: Optional[np.ndarray] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Adds a new manual blemish blob with intelligent auto-radius estimation."""
    target_blobs = blobs if blobs is not None else (existing_blobs if existing_blobs is not None else [])
    
    if point is not None:
        cx, cy = point
    elif x is not None and y is not None:
        cx, cy = x, y
    else:
        cx, cy = 0, 0

    if img_rgb is not None:
        eff_radius = estimate_clicked_blemish_radius(img_rgb, cx, cy, default_r=default_radius)
    else:
        eff_radius = radius if radius != 8 else default_radius

    next_id = max([b.get("id", 0) for b in target_blobs], default=0) + 1
    
    w = shape[1] if shape else (img_rgb.shape[1] if img_rgb is not None else 9999)
    h = shape[0] if shape else (img_rgb.shape[0] if img_rgb is not None else 9999)

    pad = max(2, int(eff_radius * 0.35))
    new_blob = {
        "id": next_id,
        "bbox": [max(0, int(cx - eff_radius - pad)), max(0, int(cy - eff_radius - pad)),
                 min(w, int(cx + eff_radius + pad)), min(h, int(cy + eff_radius + pad))],
        "centroid": [float(cx), float(cy)],
        "radius": int(eff_radius),
        "area": int(math.pi * eff_radius * eff_radius),
        "confidence": 1.0,
        "active": True,
        "label": "manual_pimple",
        "source": "manual_click"
    }
    target_blobs.append(new_blob)
    return target_blobs, new_blob


def toggle_blob_at_point(
    blobs: Optional[List[Dict[str, Any]]] = None,
    point: Optional[Tuple[int, int]] = None,
    existing_blobs: Optional[List[Dict[str, Any]]] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    hit_radius_multiplier: float = 1.6
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Toggles active state of a blob clicked by the user."""
    target_blobs = blobs if blobs is not None else (existing_blobs if existing_blobs is not None else [])
    
    if point is not None:
        px, py = point
    elif x is not None and y is not None:
        px, py = x, y
    else:
        return target_blobs, None

    for blob in target_blobs:
        bx = blob.get("centroid", [blob.get("x", 0), blob.get("y", 0)])[0]
        by = blob.get("centroid", [blob.get("x", 0), blob.get("y", 0)])[1]
        r = blob.get("radius", blob.get("r", 8))
        dist = math.hypot(bx - px, by - py)
        if dist <= (r * hit_radius_multiplier):
            blob["active"] = not blob.get("active", True)
            return target_blobs, blob
    return target_blobs, None
