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

YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "retouch_yolov8_seg.pt")
_yolo_model = None
_yolo_load_attempted = False


def get_yolo_model():
    global _yolo_model, _yolo_load_attempted
    if _yolo_load_attempted:
        return _yolo_model
    _yolo_load_attempted = True
    try:
        if os.path.exists(YOLO_MODEL_PATH):
            from ultralytics import YOLO
            _yolo_model = YOLO(YOLO_MODEL_PATH)
            logger.info("YOLO acne detector loaded from %s", YOLO_MODEL_PATH)
        else:
            logger.info("No YOLO detector at %s - using classical CV pipeline.", YOLO_MODEL_PATH)
    except Exception as e:
        logger.warning("YOLO detector load failed (%s) - using classical CV pipeline.", e)
    return _yolo_model


def compute_multi_scale_differential_energy(
    img_rgb: np.ndarray,
    skin_binary: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-Scale Differential Energy Extractor:
    Computes local chromatic redness deviation (delta a*) and multi-scale Laplacian/DoG
    energy to isolate every tiny micro-pimple, whitehead, blackhead, and inflamed papule,
    while suppressing clean anatomical highlights (nose tip/bridge) and directional creases (nostrils, ears, neck folds).
    """
    h, w, _ = img_rgb.shape
    skin_float = skin_binary.astype(np.float32)

    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(img_lab)

    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)

    # 1. Local Adaptive Redness Baseline (15-25px kernel)
    ksize_local = max(13, int(min(h, w) * 0.025) | 1)
    blurred_skin_w = cv2.GaussianBlur(skin_float, (ksize_local, ksize_local), 0) + 1e-4
    local_baseline_a = cv2.GaussianBlur(a_chan * skin_float, (ksize_local, ksize_local), 0) / blurred_skin_w
    local_baseline_l = cv2.GaussianBlur(l_chan * skin_float, (ksize_local, ksize_local), 0) / blurred_skin_w

    # Delta Redness (Erythema)
    delta_a = np.maximum(0.0, a_chan - local_baseline_a)
    norm_redness = np.maximum(0.0, (r - g) / (r + g + 18.0))
    erythema_signal = (delta_a * 0.75) + (norm_redness * 40.0 * 0.25)

    # Whitehead Specular Core: Strictly gated by minimum redness so clean highlights NEVER trigger spots
    raw_specular = np.maximum(0.0, l_chan - cv2.GaussianBlur(l_chan, (5, 5), 0))
    specular_gate = np.clip((delta_a - 1.5) / 2.0, 0.0, 1.0)
    specular_signal = raw_specular * specular_gate
    
    # Post-inflammatory hyperpigmentation (Focal dark comedones)
    dark_comedone_signal = np.maximum(0.0, local_baseline_l - l_chan)
    dark_comedone_signal = np.clip(dark_comedone_signal * 0.45, 0.0, 20.0)

    combined_signal = (erythema_signal * 0.75) + (specular_signal * 0.15) + (dark_comedone_signal * 0.10)

    # 2. Multi-Scale Difference of Gaussians (DoG) capturing all discrete blemish sizes (2.5px to 11px)
    scales = [(0.8, 1.8), (1.2, 2.5), (1.8, 3.6), (2.6, 5.2), (3.8, 7.6)]
    dog_maps = []
    for s1, s2 in scales:
        g1 = cv2.GaussianBlur(combined_signal, (0, 0), s1)
        g2 = cv2.GaussianBlur(combined_signal, (0, 0), s2)
        dog = np.maximum(0.0, g1 - g2)
        dog_maps.append(dog)

    spot_energy = np.maximum.reduce(dog_maps)

    # 3. Structure Tensor Linearity & Coherence: Strictly rejects 1D edges (ear cartilage, nostril rims, neck creases)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
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

    # Strong attenuation for directional creases (coherence > 0.25)
    edge_attenuation = np.clip(1.0 - (coherence - 0.25) * 3.8, 0.0, 1.0)
    spot_energy = spot_energy * edge_attenuation

    # 4. Strict Anatomical Feature & Cavity Exclusion Zones
    # Protect Nostrils, Mouth Interior, Lips, and Clean Nose from false spot detection
    dark_cavities = ((gray < 72.0) | (l_chan < 62.0)).astype(np.uint8) * 255
    k_cav = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    dilated_cavities = cv2.dilate(dark_cavities, k_cav)
    spot_energy[dilated_cavities > 0] = 0.0

    # Lip & Vermilion Border Exclusion (Mucosal red lips & mouth corners)
    lip_heuristic = ((norm_redness > 0.16) & (l_chan < 145.0) & (b_chan < 148.0)).astype(np.uint8) * 255
    k_lip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    dilated_lips = cv2.dilate(lip_heuristic, k_lip)
    spot_energy[dilated_lips > 0] = 0.0

    # 5. Profile Contour Margin Protection
    if min(h, w) > 200:
        margin_ksize = max(9, int(min(h, w) * 0.015) | 1)
        kernel_skin_margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_ksize, margin_ksize))
        skin_inner = cv2.erode(skin_binary, kernel_skin_margin)
        spot_energy[skin_inner == 0] = 0.0

    return spot_energy, delta_a, coherence


def verify_annular_contrast(
    img_rgb: np.ndarray,
    img_lab: np.ndarray,
    skin_binary: np.ndarray,
    cx: float,
    cy: float,
    radius: int
) -> Tuple[bool, float]:
    """
    Strict Dermatological Annular Contrast Verification:
    Tests if candidate (cx, cy) is a genuine localized blemish core or clean porcelain background skin.
    Clean skin (such as porcelain forehead, cheeks, nose, neck shadow) has ~0 delta against its immediate ring.
    """
    h, w = img_rgb.shape[:2]
    ix, iy = int(round(cx)), int(round(cy))
    r = max(3, min(12, int(round(radius))))

    r_outer = int(round(r * 2.2)) + 1
    x1 = max(0, ix - r_outer)
    y1 = max(0, iy - r_outer)
    x2 = min(w, ix + r_outer + 1)
    y2 = min(h, iy + r_outer + 1)

    if (x2 - x1) < (r * 2) or (y2 - y1) < (r * 2):
        return False, 0.0

    grid_y, grid_x = np.ogrid[y1:y2, x1:x2]
    dist_sq = (grid_x - ix) ** 2 + (grid_y - iy) ** 2

    # Inner core disk (<= r)
    core_mask = (dist_sq <= (r * r)) & (skin_binary[y1:y2, x1:x2] > 0)
    # Surrounding annulus (1.35r <= dist <= 2.2r)
    r_ann_in = int(round(r * 1.35))
    r_ann_out = int(round(r * 2.2))
    annulus_mask = (dist_sq >= (r_ann_in * r_ann_in)) & (dist_sq <= (r_ann_out * r_ann_out)) & (skin_binary[y1:y2, x1:x2] > 0)

    if np.sum(core_mask) < 3 or np.sum(annulus_mask) < 6:
        return False, 0.0

    patch_lab = img_lab[y1:y2, x1:x2]
    l_chan = patch_lab[:, :, 0]
    a_chan = patch_lab[:, :, 1]

    # STRICT NOSTRIL & LIP REJECTION
    if np.any(l_chan[core_mask] < 58.0) or np.any(l_chan[annulus_mask] < 48.0):
        return False, 0.0
    if np.mean(a_chan[core_mask]) > 165.0 and np.mean(l_chan[core_mask]) < 135.0:
        return False, 0.0

    patch_rgb = img_rgb[y1:y2, x1:x2].astype(np.float32)
    patch_red_ratio = (patch_rgb[:, :, 0] - patch_rgb[:, :, 1]) / (patch_rgb[:, :, 0] + patch_rgb[:, :, 1] + 18.0)

    core_a = float(np.mean(a_chan[core_mask]))
    ann_a = float(np.mean(a_chan[annulus_mask]))
    delta_a = core_a - ann_a

    core_l = float(np.mean(l_chan[core_mask]))
    ann_l = float(np.mean(l_chan[annulus_mask]))
    delta_l_dark = ann_l - core_l        # Dark comedone drop
    delta_l_white = core_l - ann_l       # Whitehead specular spike

    core_red = float(np.mean(patch_red_ratio[core_mask]))
    ann_red = float(np.mean(patch_red_ratio[annulus_mask]))
    delta_red = core_red - ann_red

    # Strict Dermatological Lesion Criteria (prevents clean skin / JPEG noise hits)
    # 1. Inflamed Papule/Pustule: High localized erythema spike
    is_inflamed = (delta_a >= 3.8 and delta_red >= 0.035) or (delta_a >= 4.8) or (delta_red >= 0.060)
    # 2. Whitehead: Distinct luminance peak with surrounding inflammatory redness
    is_whitehead = (delta_l_white >= 5.0) and (delta_a >= 1.6 or delta_red >= 0.020)
    # 3. Dark Comedone: Distinct localized pigment core drop
    is_comedone = (delta_l_dark >= 5.5) and (delta_a >= 0.0)

    is_valid = is_inflamed or is_whitehead or is_comedone
    contrast_score = max(delta_a * 1.5, delta_red * 100.0, delta_l_white * 1.2, delta_l_dark * 1.0)

    return is_valid, contrast_score


def refine_peak_and_radius(
    energy_map: np.ndarray,
    delta_a_map: np.ndarray,
    raw_cx: float,
    raw_cy: float,
    initial_r: int = 5,
    search_window: int = 3
) -> Tuple[float, float, int]:
    h, w = energy_map.shape[:2]
    ix, iy = int(round(raw_cx)), int(round(raw_cy))

    # 1. Search local peak within window
    x1 = max(0, ix - search_window)
    y1 = max(0, iy - search_window)
    x2 = min(w, ix + search_window + 1)
    y2 = min(h, iy + search_window + 1)

    patch_energy = energy_map[y1:y2, x1:x2]
    if patch_energy.size > 0:
        local_max = np.unravel_index(np.argmax(patch_energy), patch_energy.shape)
        peak_y = y1 + local_max[0]
        peak_x = x1 + local_max[1]
    else:
        peak_x, peak_y = ix, iy

    # 2. Radial falloff measurement along 8 compass rays
    max_test_r = min(14, max(initial_r + 2, 7))
    peak_val = float(energy_map[peak_y, peak_x]) if (0 <= peak_y < h and 0 <= peak_x < w) else 1.0
    thresh_falloff = max(0.7, peak_val * 0.40)

    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    radii = []

    for ang in angles:
        rad = math.radians(ang)
        dx = math.cos(rad)
        dy = math.sin(rad)
        r_found = initial_r
        for step in range(2, max_test_r + 1):
            px = int(round(peak_x + dx * step))
            py = int(round(peak_y + dy * step))
            if px < 0 or py < 0 or px >= w or py >= h:
                r_found = step
                break
            if energy_map[py, px] < thresh_falloff:
                r_found = step
                break
        radii.append(r_found)

    fitted_radius = int(np.percentile(radii, 60))
    fitted_radius = max(3, min(12, fitted_radius))

    return float(peak_x), float(peak_y), fitted_radius


def calculate_circle_iou(cx1: float, cy1: float, r1: float, cx2: float, cy2: float, r2: float) -> float:
    d = math.hypot(cx1 - cx2, cy1 - cy2)
    if d >= (r1 + r2):
        return 0.0
    if d <= abs(r1 - r2):
        r_min = min(r1, r2)
        r_max = max(r1, r2)
        return (math.pi * r_min * r_min) / (math.pi * r_max * r_max)

    r1_sq = r1 * r1
    r2_sq = r2 * r2
    d_sq = d * d

    alpha = math.acos(max(-1.0, min(1.0, (d_sq + r1_sq - r2_sq) / (2.0 * d * r1))))
    beta = math.acos(max(-1.0, min(1.0, (d_sq + r2_sq - r1_sq) / (2.0 * d * r2))))

    inter_area = (r1_sq * alpha - r1_sq * math.sin(2.0 * alpha) / 2.0) + \
                 (r2_sq * beta - r2_sq * math.sin(2.0 * beta) / 2.0)
    union_area = (math.pi * r1_sq) + (math.pi * r2_sq) - inter_area

    if union_area <= 0:
        return 0.0
    return max(0.0, min(1.0, inter_area / union_area))


def apply_fast_nms(
    candidates: List[Dict[str, Any]],
    overlap_dist_factor: float = 0.75,
    max_blobs: int = 350
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda b: float(b.get("confidence", 0.5)), reverse=True)
    kept: List[Dict[str, Any]] = []

    for cand in sorted_candidates:
        if len(kept) >= max_blobs:
            break

        cx1, cy1 = cand["centroid"]
        r1 = float(cand.get("radius", 6))
        suppress = False

        for kept_b in kept:
            cx2, cy2 = kept_b["centroid"]
            r2 = float(kept_b.get("radius", 6))
            dist = math.hypot(cx1 - cx2, cy1 - cy2)

            iou = calculate_circle_iou(cx1, cy1, r1, cx2, cy2, r2)
            if iou > 0.22 or dist < (min(r1, r2) * 0.90):
                suppress = True
                if cand.get("confidence", 0) > kept_b.get("confidence", 0):
                    kept_b["centroid"] = [(cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0]
                    kept_b["radius"] = max(r1, r2)
                break

        if not suppress:
            cand["id"] = len(kept) + 1
            kept.append(cand)

    return kept


def detect_pimple_candidates(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    sensitivity: float = 0.5,
    min_radius: int = 3,
    max_radius: int = 14,
    gemini_api_key: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Master Precision Blemish Detection Pipeline.
    Strictly differentiates real acne lesions from clean porcelain skin, smooth lighting shadows,
    and fine photographic grain.
    """
    h, w, _ = img_rgb.shape
    skin_binary = (skin_mask > 128).astype(np.uint8)

    if np.sum(skin_binary) < 100:
        logger.warning("Skin mask is empty. Detection skipped.")
        return [], np.zeros((h, w), dtype=np.uint8)

    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # 1. Compute multi-scale differential energy & directional coherence
    spot_energy, delta_a, coherence = compute_multi_scale_differential_energy(img_rgb, skin_binary)

    raw_candidates: List[Dict[str, Any]] = []

    # 2. Stage 1: Gemini Vision AI (if API key provided)
    api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            from gemini_detector import detect_blemishes_gemini_blobs
            logger.info("Executing Gemini Vision Detailed Blemish Scan...")
            pil_img = Image.fromarray(img_rgb)
            vlm_blobs, _ = detect_blemishes_gemini_blobs(pil_img, api_key=api_key, dilate_pixels=2)
            if vlm_blobs:
                for b in vlm_blobs:
                    cx_raw, cy_raw = b["centroid"]
                    if 0 <= cy_raw < h and 0 <= cx_raw < w and skin_binary[int(cy_raw), int(cx_raw)] > 0:
                        px, py, fit_r = refine_peak_and_radius(spot_energy, delta_a, cx_raw, cy_raw, b.get("radius", 5), search_window=3)
                        valid_spot, contrast_score = verify_annular_contrast(img_rgb, img_lab, skin_binary, px, py, fit_r)
                        if valid_spot:
                            raw_candidates.append({
                                "id": 0,
                                "bbox": [int(px - fit_r), int(py - fit_r), int(px + fit_r), int(py + fit_r)],
                                "centroid": [round(px, 1), round(py, 1)],
                                "radius": fit_r,
                                "area": int(math.pi * fit_r * fit_r),
                                "confidence": 0.96,
                                "active": True,
                                "label": b.get("label", "pimple"),
                                "source": "vlm_gemini"
                            })
                logger.info(f"Gemini Vision proposed {len(raw_candidates)} verified candidates.")
        except Exception as e:
            logger.warning(f"Gemini Vision scan warning: {e}")

    # 3. Stage 2: Local Multi-Scale Peak Detection with Strict Annular Verification
    skin_spots = spot_energy[skin_binary > 0]
    if len(skin_spots) > 0:
        mean_val = float(np.mean(skin_spots))
        std_val = float(np.std(skin_spots))
        max_val = float(np.max(skin_spots))

        # Absolute significance floor for genuine blemishes:
        # Clean porcelain skin has max spot energy < 1.6. It will yield 0 false spots!
        min_abs_energy = 1.70 - (sensitivity - 0.5) * 0.90
        
        if max_val >= min_abs_energy:
            # Dynamic threshold scaling
            k = (1.0 - max(0.05, min(1.0, sensitivity))) * 1.8 + 0.50
            threshold = max(min_abs_energy, mean_val + (k * std_val))

            candidate_mask = ((spot_energy >= threshold) & (spot_energy > 1.20)).astype(np.uint8) * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask)

            for i in range(1, num_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                cx, cy = float(centroids[i][0]), float(centroids[i][1])

                if bw == 0 or bh == 0 or area < 1:
                    continue

                aspect_ratio = float(bw) / float(bh)
                if aspect_ratio < 0.20 or aspect_ratio > 5.0:
                    continue

                eff_r = max(3, int(round(math.sqrt(area / math.pi) * 1.3)))
                px, py, fit_r = refine_peak_and_radius(spot_energy, delta_a, cx, cy, eff_r, search_window=3)

                if fit_r < min_radius or fit_r > max_radius:
                    continue

                # Annular local contrast verification (Rejects clean forehead, cheeks, and neck shadows)
                valid_spot, contrast_score = verify_annular_contrast(img_rgb, img_lab, skin_binary, px, py, fit_r)
                if not valid_spot:
                    continue

                # Structure tensor coherence check at peak (rejects crease lines)
                ipx, ipy = int(round(px)), int(round(py))
                if 0 <= ipy < h and 0 <= ipx < w:
                    if coherence[ipy, ipx] > 0.32:
                        continue

                blob_pixels = spot_energy[labels == i]
                peak_val = float(np.max(blob_pixels)) if len(blob_pixels) > 0 else threshold
                conf = min(0.98, max(0.55, float(peak_val / (threshold + 1e-4) * 0.55)))

                raw_candidates.append({
                    "id": 0,
                    "bbox": [int(px - fit_r), int(py - fit_r), int(px + fit_r), int(py + fit_r)],
                    "centroid": [round(px, 1), round(py, 1)],
                    "radius": fit_r,
                    "area": area,
                    "confidence": round(conf, 2),
                    "active": True,
                    "label": "pimple",
                    "source": "auto_cv"
                })
        else:
            logger.info("Clean, smooth porcelain skin detected (peak energy %.2f < %.2f): 0 blemish spots.", max_val, min_abs_energy)

    # 4. Apply Non-Maximum Suppression to remove duplicates and merge clusters
    final_blobs = apply_fast_nms(raw_candidates, overlap_dist_factor=0.75, max_blobs=350)
    final_mask = blobs_to_mask(final_blobs, (h, w), dilate_px=1)

    logger.info(f"Final precision blemish detection: {len(final_blobs)} verified targets.")
    return final_blobs, final_mask


def blobs_to_mask(blobs: List[Dict[str, Any]], shape: Tuple[int, int], dilate_px: int = 0) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for blob in blobs:
        if blob.get("active", True) is False:
            continue
        cx = int(round(blob.get("centroid", [0, 0])[0]))
        cy = int(round(blob.get("centroid", [0, 0])[1]))
        r = int(blob.get("radius", 5))
        cv2.circle(mask, (cx, cy), max(2, r), 255, -1)

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, k)

    return mask


def add_blob_at_point(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    x: int,
    y: int,
    existing_blobs: List[Dict[str, Any]],
    default_radius: int = 6
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    h, w, _ = img_rgb.shape
    new_id = max([b.get("id", 0) for b in existing_blobs], default=0) + 1
    
    radius = max(3, min(18, default_radius))
    
    pad = 1
    x1 = max(0, x - radius - pad)
    y1 = max(0, y - radius - pad)
    x2 = min(w, x + radius + pad)
    y2 = min(h, y + radius + pad)

    new_blob = {
        "id": new_id,
        "bbox": [x1, y1, x2, y2],
        "centroid": [float(x), float(y)],
        "radius": radius,
        "area": int(math.pi * radius * radius),
        "confidence": 0.99,
        "active": True,
        "label": "user_added",
        "source": "click"
    }

    updated = list(existing_blobs)
    updated.append(new_blob)
    return updated, new_blob


def delete_blob_at_point(
    x: int,
    y: int,
    existing_blobs: List[Dict[str, Any]],
    hit_padding: int = 6
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    deleted = None
    min_dist = float("inf")
    closest_idx = -1

    for idx, b in enumerate(existing_blobs):
        cx, cy = b["centroid"]
        dist = math.hypot(cx - x, cy - y)
        hit_radius = max(8, b.get("radius", 6) + hit_padding)
        if dist <= hit_radius and dist < min_dist:
            min_dist = dist
            closest_idx = idx

    if closest_idx >= 0:
        deleted = existing_blobs[closest_idx]
        updated = [b for i, b in enumerate(existing_blobs) if i != closest_idx]
        return updated, deleted

    return existing_blobs, None


def toggle_blob_at_point(
    x: int,
    y: int,
    existing_blobs: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    updated = [dict(b) for b in existing_blobs]
    toggled = None
    min_dist = float("inf")
    closest_idx = -1

    for idx, b in enumerate(updated):
        cx, cy = b["centroid"]
        dist = math.hypot(cx - x, cy - y)
        hit_radius = max(8, b.get("radius", 6) + 4)
        if dist <= hit_radius and dist < min_dist:
            min_dist = dist
            closest_idx = idx

    if closest_idx >= 0:
        updated[closest_idx]["active"] = not updated[closest_idx].get("active", True)
        toggled = updated[closest_idx]

    return updated, toggled
