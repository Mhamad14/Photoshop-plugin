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


def compute_local_stats(signal: np.ndarray, mask: np.ndarray, win_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes spatially local mean and standard deviation over masked skin area.
    """
    win_size = win_size | 1  # ensure odd
    mask_f = (mask > 0).astype(np.float32)
    
    blurred_signal = cv2.GaussianBlur(signal * mask_f, (win_size, win_size), 0)
    norm_w = cv2.GaussianBlur(mask_f, (win_size, win_size), 0) + 1e-5
    local_mean = blurred_signal / norm_w
    
    sq_signal = cv2.GaussianBlur((signal ** 2) * mask_f, (win_size, win_size), 0)
    local_var = np.maximum(0, (sq_signal / norm_w) - (local_mean ** 2))
    local_std = np.sqrt(local_var)
    
    return local_mean, local_std


def detect_pimple_candidates(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    sensitivity: float = 0.5,
    min_radius: int = 2,
    max_radius: int = 35,
    gemini_api_key: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Layer 2 — Advanced Multi-Scale Adaptive Blemish Detection & Organic Masking.
    
    Features:
    1. Multi-Spectral Erythema Fusion (CIELAB a*, Normalized Erythema Index, Melanin/Lightness Depression, Pustule Apex).
    2. Multi-Scale Morphological Top-Hat & Difference of Gaussians (DoG) with Resolution Scaling.
    3. Spatially Local Adaptive Contrast Windowing (handles shadow & highlight areas equally).
    4. Organic Contour Growth (accurately encompasses blemish core + erythema halo).
    5. Structural Edge Exclusion (preserves lips, nostrils, eyelid creases, facial contours).
    
    Returns:
        blobs: List of candidate blob dicts {id, bbox, centroid, radius, confidence, active, label, contour}
        binary_pimple_mask: uint8 2D array (0 or 255)
    """
    h, w, _ = img_rgb.shape
    skin_binary = (skin_mask > 128).astype(np.uint8)
    
    if np.sum(skin_binary) < 100:
        logger.warning("Skin mask is nearly empty. Detection skipped.")
        return [], np.zeros((h, w), dtype=np.uint8)

    scale_factor = max(1.0, min(h, w) / 1000.0)

    # -------------------------------------------------------------
    # STAGE A: Multi-Spectral & Morphological Saliency Extraction
    # -------------------------------------------------------------
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(img_lab)
    
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)

    # 1. Normalized Erythema Index & Redness
    norm_redness = (r - g) / (r + g + 12.0)
    erythema_idx = 100.0 * (np.log10(r + 1.0) - 0.5 * np.log10(g + 1.0) - 0.5 * np.log10(b + 1.0))
    erythema_idx = np.clip(erythema_idx, 0, 100.0)

    # 2. Local Baseline in LAB space
    blur_kernel_size = max(25, int(min(h, w) * 0.08)) | 1
    local_mean_a, local_std_a = compute_local_stats(a_chan, skin_binary, blur_kernel_size)
    local_mean_l, local_std_l = compute_local_stats(l_chan, skin_binary, blur_kernel_size)

    # Delta signals
    delta_redness = np.maximum(0, a_chan - local_mean_a)
    delta_darkness = np.maximum(0, local_mean_l - l_chan)  # hyperpigmentation, scabs, blackheads
    delta_apex = np.maximum(0, l_chan - local_mean_l)      # pustule white centers

    # 3. Multi-Scale Difference of Gaussians (DoG)
    base_scales = [(1.0, 2.2), (1.8, 3.8), (3.0, 6.5), (5.0, 11.0), (8.0, 18.0), (12.0, 26.0)]
    scales = [(s1 * scale_factor, s2 * scale_factor) for s1, s2 in base_scales]
    
    # Combined raw signal map
    fused_signal = (
        (delta_redness * 0.55) +
        (norm_redness * 35.0 * 0.20) +
        (erythema_idx * 0.15) +
        (delta_darkness * 0.40) +
        (delta_apex * 0.20)
    )

    dog_maps = []
    for s1, s2 in scales:
        g1 = cv2.GaussianBlur(fused_signal, (0, 0), s1)
        g2 = cv2.GaussianBlur(fused_signal, (0, 0), s2)
        dog = np.maximum(0, g1 - g2)
        dog_maps.append(dog)
        
    dog_energy = np.maximum.reduce(dog_maps)

    # 4. Multi-Scale Morphological Top-Hat Transforms (Curvature / Bump Detector)
    tophat_maps = []
    for r_k in [3, 7, 13, 21]:
        rad_px = max(2, int(r_k * scale_factor))
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad_px * 2 + 1, rad_px * 2 + 1))
        # White top-hat on redness (raised inflamed bumps)
        w_th = cv2.morphologyEx(a_chan.astype(np.uint8), cv2.MORPH_TOPHAT, kernel_morph).astype(np.float32)
        # Black top-hat on lightness (dark comedones / scabs)
        b_th = cv2.morphologyEx(l_chan.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel_morph).astype(np.float32)
        tophat_maps.append(w_th * 0.6 + b_th * 0.4)

    tophat_energy = np.maximum.reduce(tophat_maps)

    # Combined spot energy
    spot_energy = (dog_energy * 0.65) + (tophat_energy * 0.35)

    # 5. Boundary & Structural Feature Protection
    margin_px = max(4, int(5 * scale_factor))
    kernel_skin_margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1))
    skin_inner = cv2.erode(skin_binary, kernel_skin_margin)
    spot_energy[skin_inner == 0] = 0.0

    # Protect facial structural edges (eyes, lips, nostrils, jawline)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 110)
    edge_ksize = max(3, int(4 * scale_factor)) | 1
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_ksize, edge_ksize))
    edge_exclusion = cv2.dilate(edges, kernel_edge)
    spot_energy[edge_exclusion > 0] *= 0.08

    # 6. Local Dynamic Contrast Thresholding
    win_stat = max(31, int(min(h, w) * 0.10)) | 1
    loc_mean_spot, loc_std_spot = compute_local_stats(spot_energy, skin_inner, win_stat)

    # Sensitivity mapping: 0.1 (strict) to 1.0 (sensitive)
    k_thresh = (1.0 - max(0.05, min(1.0, sensitivity))) * 3.4 + 0.45
    adaptive_threshold = loc_mean_spot + (k_thresh * loc_std_spot)
    adaptive_threshold = np.maximum(adaptive_threshold, 1.2)  # noise floor

    candidate_mask = (spot_energy > adaptive_threshold).astype(np.uint8) * 255

    # -------------------------------------------------------------
    # STAGE B: Connected Components & Organic Contour Growth
    # -------------------------------------------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask)
    blobs: List[Dict[str, Any]] = []
    final_mask = np.zeros((h, w), dtype=np.uint8)
    blob_id = 1

    # Scaled limits
    eff_min_r = max(1, int(min_radius * scale_factor * 0.6))
    eff_max_r = max(15, int(max_radius * scale_factor))

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        if bw == 0 or bh == 0 or area < 3:
            continue

        aspect_ratio = float(bw) / float(bh)
        # Exclude long linear features (hairs, wrinkles)
        if aspect_ratio < 0.25 or aspect_ratio > 4.0:
            continue

        eff_radius = int(math.ceil(math.sqrt(area / math.pi)))
        if eff_radius < eff_min_r or eff_radius > eff_max_r:
            continue

        # Confidence calculation
        blob_pixels = spot_energy[labels == i]
        peak_val = float(np.max(blob_pixels)) if len(blob_pixels) > 0 else 1.0
        local_ref_std = float(loc_std_spot[int(cy), int(cx)]) if 0 <= int(cy) < h and 0 <= int(cx) < w else 1.0
        conf = min(0.99, max(0.40, float((peak_val / (local_ref_std * 3.0 + 1e-5)) * 0.35 + 0.60)))

        # Adaptive radius covering blemish core + erythema halo margin
        halo_margin = max(3, int(eff_radius * 0.45 + 2 * scale_factor))
        radius_with_margin = eff_radius + halo_margin

        # Bounding box with context
        pad = halo_margin + 2
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + bw + pad)
        y2 = min(h, by + bh + pad)

        # Generate organic local mask for this blob
        # Extract local ROI around the blob
        roi_y1 = max(0, int(cy - radius_with_margin - 3))
        roi_y2 = min(h, int(cy + radius_with_margin + 4))
        roi_x1 = max(0, int(cx - radius_with_margin - 3))
        roi_x2 = min(w, int(cx + radius_with_margin + 4))

        local_seed = (labels[roi_y1:roi_y2, roi_x1:roi_x2] == i).astype(np.uint8) * 255
        # Dilate organically based on local erythema / spot energy
        k_grow = max(3, halo_margin * 2 + 1)
        kernel_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_grow, k_grow))
        local_organic = cv2.dilate(local_seed, kernel_grow)

        # Smooth boundary
        local_organic = cv2.GaussianBlur(local_organic, (5, 5), 0)
        local_organic = (local_organic > 70).astype(np.uint8) * 255

        # Place onto final mask
        final_mask[roi_y1:roi_y2, roi_x1:roi_x2] = np.maximum(
            final_mask[roi_y1:roi_y2, roi_x1:roi_x2],
            local_organic
        )

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
        blob_id += 1

    # Optional Stage B2: Gemini Vision VLM Verification / Augmentation
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
                    vcx, vcy = float(vlm_centroids[v][0]), float(vlm_centroids[v][1])
                    varea = int(vlm_stats[v, cv2.CC_STAT_AREA])
                    v_rad = int(math.ceil(math.sqrt(varea / math.pi))) + max(3, int(3 * scale_factor))

                    # Check if already covered by CV blob
                    already_covered = False
                    for b in blobs:
                        bcx, bcy = b["centroid"]
                        dist = math.hypot(vcx - bcx, vcy - bcy)
                        if dist <= (b["radius"] + 4):
                            already_covered = True
                            b["confidence"] = min(0.99, b["confidence"] + 0.15)
                            b["source"] = "cv+vlm"
                            break

                    if not already_covered:
                        vx1 = max(0, int(vlm_stats[v, cv2.CC_STAT_LEFT]))
                        vy1 = max(0, int(vlm_stats[v, cv2.CC_STAT_TOP]))
                        vx2 = min(w, vx1 + int(vlm_stats[v, cv2.CC_STAT_WIDTH]))
                        vy2 = min(h, vy1 + int(vlm_stats[v, cv2.CC_STAT_HEIGHT]))

                        blobs.append({
                            "id": blob_id,
                            "bbox": [vx1, vy1, vx2, vy2],
                            "centroid": [round(vcx, 1), round(vcy, 1)],
                            "radius": v_rad,
                            "area": varea,
                            "confidence": 0.94,
                            "active": True,
                            "label": "pimple",
                            "source": "vlm"
                        })
                        cv2.circle(final_mask, (int(round(vcx)), int(round(vcy))), v_rad, 255, -1)
                        blob_id += 1
        except Exception as e:
            logger.warning(f"VLM verification error: {e}")

    logger.info(f"Advanced pimple detector produced {len(blobs)} candidate blobs.")
    return blobs, final_mask


def blobs_to_mask(blobs: List[Dict[str, Any]], shape: Tuple[int, int], soft_falloff: bool = False) -> np.ndarray:
    """
    Renders active blobs into a 2D uint8 mask (h, w).
    Supports anti-aliased organic boundaries.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in blobs:
        if not b.get("active", True):
            continue
        cx, cy = b["centroid"]
        r = b.get("radius", 6)
        cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    
    if soft_falloff:
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def add_blob_at_point(
    img_rgb: np.ndarray,
    skin_mask: np.ndarray,
    x: int,
    y: int,
    existing_blobs: List[Dict[str, Any]],
    default_radius: int = 8
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Refinement Layer 4: Adds a new pimple blob at (x, y) with adaptive radius
    computed from local color gradient / erythema bounds.
    """
    h, w = img_rgb.shape[:2]
    x = max(0, min(w - 1, int(x)))
    y = max(0, min(h - 1, int(y)))

    # Estimate local spot radius via radial color gradient analysis
    crop_r = 25
    y1, y2 = max(0, y - crop_r), min(h, y + crop_r + 1)
    x1, x2 = max(0, x - crop_r), min(w, x + crop_r + 1)

    patch = img_rgb[y1:y2, x1:x2]
    patch_lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    center_a = patch_lab[y - y1, x - x1, 1]
    center_l = patch_lab[y - y1, x - x1, 0]

    # Measure radial distance where redness / darkness drops to background
    ph, pw, _ = patch.shape
    dist_map = np.zeros((ph, pw), dtype=np.float32)
    for py in range(ph):
        for px in range(pw):
            dist_map[py, px] = math.hypot(px - (x - x1), py - (y - y1))

    # Ring background estimate (radius 15-22)
    outer_ring = (dist_map >= 15) & (dist_map <= 22)
    if np.sum(outer_ring) > 10:
        bg_a = float(np.median(patch_lab[outer_ring, 1]))
        bg_l = float(np.median(patch_lab[outer_ring, 0]))
        # Pixels with significant deviation from background
        deviation = np.abs(patch_lab[:, :, 1] - bg_a) + 0.5 * np.maximum(0, bg_l - patch_lab[:, :, 0])
        spot_pixels = (deviation > 3.5) & (dist_map <= 20)
        if np.sum(spot_pixels) >= 5:
            calc_radius = int(math.ceil(np.max(dist_map[spot_pixels]))) + 2
            calc_radius = max(3, min(30, calc_radius))
        else:
            calc_radius = default_radius
    else:
        calc_radius = default_radius

    new_id = max([b["id"] for b in existing_blobs], default=0) + 1
    new_blob = {
        "id": new_id,
        "bbox": [max(0, x - calc_radius - 3), max(0, y - calc_radius - 3), min(w, x + calc_radius + 3), min(h, y + calc_radius + 3)],
        "centroid": [float(x), float(y)],
        "radius": calc_radius,
        "area": int(math.pi * calc_radius * calc_radius),
        "confidence": 1.0,
        "active": True,
        "label": "pimple",
        "source": "manual_click"
    }

    updated_blobs = list(existing_blobs)
    updated_blobs.append(new_blob)
    return updated_blobs, new_blob


def toggle_blob_at_point(
    x: int,
    y: int,
    existing_blobs: List[Dict[str, Any]],
    hit_radius: int = 14
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Refinement Layer 4: Toggles the active state of an existing blob if clicked.
    """
    updated_blobs = []
    toggled_blob = None
    min_dist = float("inf")
    target_id = None

    for b in existing_blobs:
        cx, cy = b["centroid"]
        dist = math.hypot(x - cx, y - cy)
        r = b.get("radius", 8)
        effective_hit = max(hit_radius, r + 5)
        if dist <= effective_hit and dist < min_dist:
            min_dist = dist
            target_id = b["id"]

    for b in existing_blobs:
        b_copy = dict(b)
        if b["id"] == target_id:
            b_copy["active"] = not b.get("active", True)
            toggled_blob = b_copy
        updated_blobs.append(b_copy)

    return updated_blobs, toggled_blob
